"""Sandbox Snapshot Service.

Handles saving and restoring sandbox workspace snapshots. Snapshots persist
sandbox filesystem state so it can be recovered after SSE disconnect or timeout.

Storage: Uses the configured backend (R2 or local filesystem).
Database: SandboxSnapshot model tracks metadata and expiry.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import settings
from app.core.logging import get_logger
from app.sandbox.runtime import SandboxRuntime

logger = get_logger(__name__)

# Maximum snapshot size (100MB)
MAX_SNAPSHOT_SIZE = settings.snapshot_max_size_bytes

# Default retention period
DEFAULT_RETENTION_HOURS = settings.snapshot_retention_hours


async def save_snapshot(
    runtime: SandboxRuntime,
    user_id: str,
    task_id: str,
    sandbox_type: str,
    paths: list[str] | None = None,
) -> dict | None:
    """Save a snapshot of sandbox workspace to storage.

    Args:
        runtime: Active sandbox runtime
        user_id: User identifier
        task_id: Task identifier
        sandbox_type: Type of sandbox ("execution" or "app")
        paths: Paths to snapshot (defaults to configured paths for sandbox type)

    Returns:
        Dict with snapshot metadata, or None if snapshot failed
    """
    if paths is None:
        if sandbox_type == "execution":
            paths = list(settings.snapshot_default_paths_execution)
        else:
            paths = list(settings.snapshot_default_paths_app)

    snapshot_id = str(uuid.uuid4())
    storage_key = f"snapshots/{user_id}/{task_id}/{snapshot_id}.tar.gz"

    try:
        # Create snapshot archive via runtime
        snapshot_data = await runtime.save_snapshot(paths, snapshot_id)

        if len(snapshot_data) > MAX_SNAPSHOT_SIZE:
            logger.warning(
                "snapshot_too_large",
                size=len(snapshot_data),
                max_size=MAX_SNAPSHOT_SIZE,
                user_id=user_id,
                task_id=task_id,
            )
            return None

        if len(snapshot_data) == 0:
            logger.info(
                "snapshot_empty_no_data",
                user_id=user_id,
                task_id=task_id,
                sandbox_type=sandbox_type,
            )
            return None

        # Store snapshot data
        await _store_snapshot(storage_key, snapshot_data)

        # Record in database
        expires_at = datetime.now(timezone.utc) + timedelta(hours=DEFAULT_RETENTION_HOURS)
        snapshot_record = await _record_snapshot(
            snapshot_id=snapshot_id,
            user_id=user_id,
            task_id=task_id,
            sandbox_type=sandbox_type,
            storage_key=storage_key,
            paths_included=paths,
            size_bytes=len(snapshot_data),
            expires_at=expires_at,
        )

        logger.info(
            "snapshot_saved",
            snapshot_id=snapshot_id,
            user_id=user_id,
            task_id=task_id,
            sandbox_type=sandbox_type,
            size_bytes=len(snapshot_data),
            paths=paths,
        )

        return snapshot_record

    except Exception as e:
        logger.error(
            "snapshot_save_failed",
            user_id=user_id,
            task_id=task_id,
            sandbox_type=sandbox_type,
            error=str(e),
        )
        return None


async def restore_snapshot(
    runtime: SandboxRuntime,
    user_id: str,
    task_id: str,
    sandbox_type: str,
    snapshot_id: str | None = None,
) -> bool:
    """Restore a snapshot into a sandbox runtime.

    If snapshot_id is not provided, restores the latest snapshot for the
    given user/task/sandbox_type combination.

    Args:
        runtime: Active sandbox runtime to restore into
        user_id: User identifier
        task_id: Task identifier
        sandbox_type: Type of sandbox ("execution" or "app")
        snapshot_id: Specific snapshot ID to restore (latest if None)

    Returns:
        True if restore succeeded, False otherwise
    """
    try:
        # Find the snapshot record
        snapshot_record = await _find_snapshot(
            user_id=user_id,
            task_id=task_id,
            sandbox_type=sandbox_type,
            snapshot_id=snapshot_id,
        )

        if not snapshot_record:
            logger.info(
                "no_snapshot_found",
                user_id=user_id,
                task_id=task_id,
                sandbox_type=sandbox_type,
            )
            return False

        # Check expiry
        if snapshot_record["expires_at"] < datetime.now(timezone.utc):
            logger.info(
                "snapshot_expired",
                snapshot_id=snapshot_record["id"],
                expires_at=snapshot_record["expires_at"].isoformat(),
            )
            return False

        # Retrieve snapshot data from storage
        snapshot_data = await _retrieve_snapshot(snapshot_record["storage_key"])
        if not snapshot_data:
            logger.warning(
                "snapshot_data_not_found",
                storage_key=snapshot_record["storage_key"],
            )
            return False

        # Restore into runtime
        success = await runtime.restore_snapshot(snapshot_data, "/")

        if success:
            logger.info(
                "snapshot_restored",
                snapshot_id=snapshot_record["id"],
                user_id=user_id,
                task_id=task_id,
                sandbox_type=sandbox_type,
            )
        else:
            logger.warning(
                "snapshot_restore_failed",
                snapshot_id=snapshot_record["id"],
                user_id=user_id,
                task_id=task_id,
            )

        return success

    except Exception as e:
        logger.error(
            "snapshot_restore_error",
            user_id=user_id,
            task_id=task_id,
            sandbox_type=sandbox_type,
            error=str(e),
        )
        return False


async def _store_snapshot(storage_key: str, data: bytes) -> None:
    """Store snapshot data to configured storage backend.

    Uses asyncio.to_thread for blocking I/O to avoid blocking the event loop.
    """
    import asyncio

    if settings.storage_backend == "local":
        file_path = Path(settings.local_storage_path) / storage_key
        file_path.parent.mkdir(parents=True, exist_ok=True)

        def _write_local() -> None:
            with open(file_path, "wb") as f:
                f.write(data)

        await asyncio.to_thread(_write_local)
    else:
        # R2 storage - use asyncio.to_thread for synchronous boto3 calls
        import boto3
        from io import BytesIO

        def _upload_r2() -> None:
            client = boto3.client(
                "s3",
                endpoint_url=settings.r2_endpoint_url,
                aws_access_key_id=settings.r2_access_key_id,
                aws_secret_access_key=settings.r2_secret_access_key,
                region_name="auto",
            )
            client.upload_fileobj(
                BytesIO(data),
                settings.r2_bucket_name,
                storage_key,
                ExtraArgs={"ContentType": "application/gzip"},
            )

        await asyncio.to_thread(_upload_r2)


async def _retrieve_snapshot(storage_key: str) -> bytes | None:
    """Retrieve snapshot data from configured storage backend.

    Uses asyncio.to_thread for blocking I/O to avoid blocking the event loop.
    """
    import asyncio

    try:
        if settings.storage_backend == "local":
            file_path = Path(settings.local_storage_path) / storage_key
            if not file_path.exists():
                return None

            def _read_local() -> bytes:
                with open(file_path, "rb") as f:
                    return f.read()

            return await asyncio.to_thread(_read_local)
        else:
            # R2 storage
            import boto3
            from io import BytesIO

            def _download_r2() -> bytes:
                client = boto3.client(
                    "s3",
                    endpoint_url=settings.r2_endpoint_url,
                    aws_access_key_id=settings.r2_access_key_id,
                    aws_secret_access_key=settings.r2_secret_access_key,
                    region_name="auto",
                )
                buf = BytesIO()
                client.download_fileobj(
                    settings.r2_bucket_name,
                    storage_key,
                    buf,
                )
                buf.seek(0)
                return buf.read()

            return await asyncio.to_thread(_download_r2)
    except Exception as e:
        logger.error("snapshot_retrieve_failed", storage_key=storage_key, error=str(e))
        return None


async def _record_snapshot(
    snapshot_id: str,
    user_id: str,
    task_id: str,
    sandbox_type: str,
    storage_key: str,
    paths_included: list[str],
    size_bytes: int,
    expires_at: datetime,
) -> dict:
    """Record snapshot metadata in the database."""
    try:
        from app.db.base import get_db_session
        from app.db.models import SandboxSnapshot

        async with get_db_session() as session:
            snapshot = SandboxSnapshot(
                id=snapshot_id,
                user_id=user_id,
                task_id=task_id,
                sandbox_type=sandbox_type,
                storage_key=storage_key,
                paths_included=paths_included,
                size_bytes=size_bytes,
                expires_at=expires_at,
            )
            session.add(snapshot)
            await session.commit()

            logger.info("snapshot_recorded", snapshot_id=snapshot_id)

            return snapshot.to_dict()

    except Exception as e:
        logger.warning(
            "snapshot_db_record_failed",
            snapshot_id=snapshot_id,
            error=str(e),
        )
        # Return dict even if DB write fails (snapshot data is stored)
        return {
            "id": snapshot_id,
            "user_id": user_id,
            "task_id": task_id,
            "sandbox_type": sandbox_type,
            "storage_key": storage_key,
            "paths_included": paths_included,
            "size_bytes": size_bytes,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": expires_at.isoformat(),
        }


async def _find_snapshot(
    user_id: str,
    task_id: str,
    sandbox_type: str,
    snapshot_id: str | None = None,
) -> dict | None:
    """Find a snapshot record in the database."""
    try:
        from sqlalchemy import select

        from app.db.base import get_db_session
        from app.db.models import SandboxSnapshot

        async with get_db_session() as session:
            if snapshot_id:
                stmt = select(SandboxSnapshot).where(SandboxSnapshot.id == snapshot_id)
            else:
                # Get the latest snapshot for this user/task/type
                stmt = (
                    select(SandboxSnapshot)
                    .where(
                        SandboxSnapshot.user_id == user_id,
                        SandboxSnapshot.task_id == task_id,
                        SandboxSnapshot.sandbox_type == sandbox_type,
                    )
                    .order_by(SandboxSnapshot.created_at.desc())
                    .limit(1)
                )

            result = await session.execute(stmt)
            snapshot = result.scalar_one_or_none()

            if snapshot:
                return {
                    "id": snapshot.id,
                    "user_id": snapshot.user_id,
                    "task_id": snapshot.task_id,
                    "sandbox_type": snapshot.sandbox_type,
                    "storage_key": snapshot.storage_key,
                    "paths_included": snapshot.paths_included,
                    "size_bytes": snapshot.size_bytes,
                    "created_at": snapshot.created_at,
                    "expires_at": snapshot.expires_at,
                }
            return None

    except Exception as e:
        logger.error(
            "snapshot_find_failed",
            user_id=user_id,
            task_id=task_id,
            error=str(e),
        )
        return None
