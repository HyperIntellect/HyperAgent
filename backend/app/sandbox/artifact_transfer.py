"""Cross-sandbox artifact transfer for agent handoffs.

Collects files from a source sandbox and restores them into a target sandbox
during agent-to-agent handoffs. Uses the existing storage backend (R2/local)
as an intermediary to avoid direct sandbox-to-sandbox transfers.
"""

from __future__ import annotations

import uuid
from io import BytesIO
from typing import Any

from app.core.logging import get_logger
from app.sandbox.runtime import SandboxRuntime

logger = get_logger(__name__)

# Default file patterns to collect during handoff
DEFAULT_ARTIFACT_PATTERNS = [
    "*.py",
    "*.csv",
    "*.json",
    "*.txt",
    "*.md",
    "*.html",
    "*.js",
    "*.ts",
    "*.yaml",
    "*.yml",
    "*.xml",
    "*.sql",
    "*.sh",
]

# Maximum constraints
MAX_ARTIFACT_FILES = 10
MAX_ARTIFACT_SIZE_MB = 50
MAX_ARTIFACT_SIZE_BYTES = MAX_ARTIFACT_SIZE_MB * 1024 * 1024


async def collect_artifacts(
    runtime: SandboxRuntime,
    patterns: list[str] | None = None,
    max_files: int = MAX_ARTIFACT_FILES,
    max_size_mb: int = MAX_ARTIFACT_SIZE_MB,
) -> list[dict[str, Any]]:
    """Collect files from a sandbox for handoff transfer.

    Searches the sandbox working directory for files matching the given
    patterns, uploads them to the storage backend, and returns artifact
    metadata for inclusion in HandoffInfo.

    Args:
        runtime: Source sandbox runtime instance
        patterns: Glob patterns to match (defaults to DEFAULT_ARTIFACT_PATTERNS)
        max_files: Maximum number of files to collect
        max_size_mb: Maximum total size in MB

    Returns:
        List of artifact dicts: [{path, storage_key, size}]
    """
    from app.services.file_storage import file_storage_service

    if patterns is None:
        patterns = DEFAULT_ARTIFACT_PATTERNS

    max_size_bytes = max_size_mb * 1024 * 1024

    # Build find command for all patterns
    find_parts = []
    for pat in patterns:
        find_parts.append(f"-name '{pat}'")
    find_expr = " -o ".join(find_parts)

    # Search in /home/user (standard sandbox working dir)
    find_cmd = (
        f"find /home/user -maxdepth 4 -type f \\( {find_expr} \\) "
        f"-not -path '*/node_modules/*' -not -path '*/.git/*' "
        f"-not -path '*/venv/*' -not -path '*/__pycache__/*' "
        f"2>/dev/null | head -n {max_files * 2}"
    )

    try:
        result = await runtime.run_command(find_cmd, timeout=15)
    except Exception as e:
        logger.warning("artifact_find_failed", error=str(e))
        return []

    if result.exit_code != 0 or not result.stdout.strip():
        logger.debug("artifact_find_no_results", exit_code=result.exit_code)
        return []

    candidate_paths = [
        p.strip() for p in result.stdout.strip().split("\n") if p.strip()
    ]

    # Get file sizes to respect budget
    artifacts: list[dict[str, Any]] = []
    total_size = 0

    for path in candidate_paths:
        if len(artifacts) >= max_files:
            break

        # Get file size
        try:
            stat_result = await runtime.run_command(
                f"stat -c '%s' '{path}' 2>/dev/null || stat -f '%z' '{path}' 2>/dev/null",
                timeout=5,
            )
            if stat_result.exit_code != 0 or not stat_result.stdout.strip():
                continue
            file_size = int(stat_result.stdout.strip())
        except (ValueError, Exception):
            continue

        # Check size budget
        if total_size + file_size > max_size_bytes:
            logger.debug(
                "artifact_size_budget_exceeded",
                path=path,
                file_size=file_size,
                total_so_far=total_size,
                max_bytes=max_size_bytes,
            )
            continue

        # Read file content from sandbox
        try:
            content = await runtime.read_file(path)
            if isinstance(content, str):
                content = content.encode("utf-8")
        except Exception as e:
            logger.warning("artifact_read_failed", path=path, error=str(e))
            continue

        # Upload to storage backend with deterministic key
        storage_key = f"handoff_artifacts/{uuid.uuid4().hex}/{path.lstrip('/')}"
        try:
            file_buffer = BytesIO(content)
            if file_storage_service.backend == "local":
                from pathlib import Path as FilePath

                from app.config import settings

                dest = FilePath(settings.local_storage_path) / storage_key
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(file_buffer.read())
            else:
                file_storage_service.client.upload_fileobj(
                    file_buffer,
                    file_storage_service._get_bucket(),
                    storage_key,
                )
        except Exception as e:
            logger.warning("artifact_upload_failed", path=path, error=str(e))
            continue

        artifacts.append({
            "path": path,
            "storage_key": storage_key,
            "size": file_size,
        })
        total_size += file_size

        logger.debug(
            "artifact_collected",
            path=path,
            size=file_size,
            storage_key=storage_key,
        )

    logger.info(
        "artifacts_collected",
        count=len(artifacts),
        total_size=total_size,
        sandbox_id=runtime.sandbox_id,
    )

    return artifacts


async def restore_artifacts(
    runtime: SandboxRuntime,
    artifacts: list[dict[str, Any]],
) -> list[str]:
    """Restore previously collected artifacts into a target sandbox.

    Downloads artifacts from the storage backend and writes them into
    the target sandbox at their original paths.

    Args:
        runtime: Target sandbox runtime instance
        artifacts: List of artifact dicts from collect_artifacts

    Returns:
        List of paths that were successfully restored
    """
    from app.services.file_storage import file_storage_service

    restored_paths: list[str] = []

    for artifact in artifacts:
        path = artifact.get("path", "")
        storage_key = artifact.get("storage_key", "")

        if not path or not storage_key:
            continue

        try:
            # Download from storage
            if file_storage_service.backend == "local":
                from pathlib import Path as FilePath

                from app.config import settings

                local_path = FilePath(settings.local_storage_path) / storage_key
                if not local_path.exists():
                    logger.warning("artifact_not_found_local", storage_key=storage_key)
                    continue
                content = local_path.read_bytes()
            else:
                buffer = await file_storage_service.download_file(storage_key)
                content = buffer.read()

            # Ensure parent directory exists in target sandbox
            parent_dir = "/".join(path.split("/")[:-1])
            if parent_dir:
                await runtime.run_command(f"mkdir -p '{parent_dir}'", timeout=10)

            # Write file to target sandbox
            await runtime.write_file(path, content)
            restored_paths.append(path)

            logger.debug(
                "artifact_restored",
                path=path,
                size=len(content),
                sandbox_id=runtime.sandbox_id,
            )

        except Exception as e:
            logger.warning(
                "artifact_restore_failed",
                path=path,
                storage_key=storage_key,
                error=str(e),
            )
            continue

    logger.info(
        "artifacts_restored",
        count=len(restored_paths),
        total=len(artifacts),
        sandbox_id=runtime.sandbox_id,
    )

    return restored_paths


async def cleanup_artifacts(artifacts: list[dict[str, Any]]) -> int:
    """Clean up transferred artifact files from storage.

    Should be called after the target agent completes to free storage space.

    Args:
        artifacts: List of artifact dicts to clean up

    Returns:
        Number of artifacts successfully cleaned up
    """
    from app.services.file_storage import file_storage_service

    cleaned = 0
    for artifact in artifacts:
        storage_key = artifact.get("storage_key", "")
        if not storage_key:
            continue

        try:
            if file_storage_service.backend == "local":
                from pathlib import Path as FilePath

                from app.config import settings

                local_path = FilePath(settings.local_storage_path) / storage_key
                if local_path.exists():
                    local_path.unlink()
                    cleaned += 1
            else:
                await file_storage_service.delete_file(storage_key)
                cleaned += 1
        except Exception as e:
            logger.warning(
                "artifact_cleanup_failed",
                storage_key=storage_key,
                error=str(e),
            )

    logger.info("artifacts_cleaned_up", count=cleaned, total=len(artifacts))
    return cleaned


def format_artifact_summary(artifacts: list[dict[str, Any]]) -> str:
    """Format a human-readable summary of transferred artifacts.

    Args:
        artifacts: List of artifact dicts

    Returns:
        Formatted summary string for inclusion in handoff context
    """
    if not artifacts:
        return ""

    total_size = sum(a.get("size", 0) for a in artifacts)
    size_str = (
        f"{total_size / 1024:.1f} KB"
        if total_size < 1024 * 1024
        else f"{total_size / (1024 * 1024):.1f} MB"
    )

    lines = [f"Transferred {len(artifacts)} file(s) ({size_str}):"]
    for artifact in artifacts:
        path = artifact.get("path", "unknown")
        size = artifact.get("size", 0)
        lines.append(f"  - {path} ({size:,} bytes)")

    return "\n".join(lines)
