"""Sandbox package for sandbox management and operations.

This package contains:
- Execution sandbox management (code execution sandboxes)
- Desktop sandbox management (desktop/browser sandboxes)
- App sandbox management (web app development sandboxes)
- Sandbox file operations
- Unified metrics for all sandbox managers
- Provider-aware availability checks for graceful degradation
- Support for multiple providers: E2B (cloud) and BoxLite (local)
"""

from typing import Any

from app.sandbox import file_operations
from app.sandbox.app_sandbox_manager import (
    APP_TEMPLATES,
    AppSandboxManager,
    AppSandboxSession,
    get_app_sandbox_manager,
)
from app.sandbox.desktop_sandbox_manager import (
    DesktopSandboxManager,
    DesktopSandboxSession,
    get_desktop_sandbox_manager,
)
from app.sandbox.execution_sandbox_manager import (
    ExecutionSandboxManager,
    ExecutionSandboxSession,
    get_execution_sandbox_manager,
)
from app.sandbox.unified_sandbox_manager import (
    UnifiedSandboxManager,
    UnifiedSandboxSession,
    get_unified_sandbox_manager,
)
from app.sandbox.file import (
    sandbox_file,
    sandbox_file_with_context,
)


def get_sandbox_metrics() -> dict[str, Any]:
    """Get unified metrics from all sandbox managers.

    Returns:
        Dict with metrics from execution, desktop, app, and unified sandbox managers:
            - execution: Metrics from execution sandbox manager
            - desktop: Metrics from desktop sandbox manager
            - app: Metrics from app sandbox manager
            - unified: Metrics from unified sandbox manager
            - totals: Aggregated totals across all managers
    """
    execution_manager = get_execution_sandbox_manager()
    desktop_manager = get_desktop_sandbox_manager()
    app_manager = get_app_sandbox_manager()
    unified_manager = get_unified_sandbox_manager()

    execution_metrics = execution_manager.get_metrics()
    desktop_metrics = desktop_manager.get_metrics()
    app_metrics = app_manager.get_metrics()
    unified_metrics = unified_manager.get_metrics()

    return {
        "execution": execution_metrics,
        "desktop": desktop_metrics,
        "app": app_metrics,
        "unified": unified_metrics,
        "totals": {
            "active_sessions": (
                execution_metrics["active_sessions"]
                + desktop_metrics["active_sessions"]
                + app_metrics["active_sessions"]
                + unified_metrics["active_sessions"]
            ),
            "total_created": (
                execution_metrics["total_created"]
                + desktop_metrics["total_created"]
                + app_metrics["total_created"]
                + unified_metrics["total_created"]
            ),
            "total_cleaned": (
                execution_metrics["total_cleaned"]
                + desktop_metrics["total_cleaned"]
                + app_metrics["total_cleaned"]
                + unified_metrics["total_cleaned"]
            ),
            "total_reused": (
                execution_metrics["total_reused"]
                + desktop_metrics["total_reused"]
                + app_metrics["total_reused"]
                + unified_metrics["total_reused"]
            ),
            "health_check_failures": (
                execution_metrics["health_check_failures"]
                + desktop_metrics["health_check_failures"]
                + app_metrics["health_check_failures"]
                + unified_metrics["health_check_failures"]
            ),
        },
    }


def is_execution_sandbox_available() -> bool:
    """Check if execution sandbox functionality is available.

    Returns:
        True if the configured sandbox provider is available for code execution
    """
    from app.sandbox.provider import is_provider_available

    available, _ = is_provider_available("execution")
    return available


def is_desktop_sandbox_available() -> bool:
    """Check if desktop sandbox functionality is available.

    Returns:
        True if the configured sandbox provider supports desktop/browser automation
    """
    from app.sandbox.provider import is_provider_available

    available, _ = is_provider_available("desktop")
    return available


def is_app_sandbox_available() -> bool:
    """Check if app sandbox functionality is available.

    Returns:
        True if the configured sandbox provider is available for app development
    """
    from app.sandbox.provider import is_provider_available

    available, _ = is_provider_available("app")
    return available


def get_sandbox_availability() -> dict[str, Any]:
    """Get detailed availability status for all sandbox types.

    Returns:
        Dict with availability information:
            - provider: Current sandbox provider name
            - execution_sandbox: Whether execution sandbox is available
            - desktop_sandbox: Whether desktop sandbox is available
            - app_sandbox: Whether app sandbox is available
            - issues: List of issues preventing sandbox usage
    """
    from app.sandbox.provider import get_sandbox_provider, is_provider_available

    issues = []

    exec_available, exec_issue = is_provider_available("execution")
    desktop_available, desktop_issue = is_provider_available("desktop")
    app_available, app_issue = is_provider_available("app")

    if exec_issue:
        issues.append(exec_issue)
    if desktop_issue and desktop_issue not in issues:
        issues.append(desktop_issue)
    if app_issue and app_issue not in issues:
        issues.append(app_issue)

    return {
        "provider": get_sandbox_provider(),
        "execution_sandbox": exec_available,
        "desktop_sandbox": desktop_available,
        "app_sandbox": app_available,
        "desktop_available": is_desktop_sandbox_available(),
        "issues": issues,
    }


async def cleanup_sandboxes_for_task(user_id: str, task_id: str) -> dict[str, bool]:
    """Clean up all sandbox sessions associated with a user/task.

    Call this when an SSE connection drops to free up sandbox resources.

    Args:
        user_id: User identifier
        task_id: Task/conversation identifier

    Returns:
        Dict indicating which sandbox types were cleaned up
    """
    from app.core.logging import get_logger

    logger = get_logger(__name__)

    results: dict[str, bool] = {
        "desktop": False,
        "execution": False,
        "app": False,
        "unified": False,
    }
    cleanup_errors: dict[str, str] = {}

    # Cleanup unified sandbox first (it owns the shared runtime)
    try:
        unified_manager = get_unified_sandbox_manager()
        results["unified"] = await unified_manager.cleanup_session(
            user_id=user_id, task_id=task_id
        )
    except Exception as e:
        cleanup_errors["unified"] = str(e)
        logger.warning(
            "unified_sandbox_cleanup_failed",
            user_id=user_id,
            task_id=task_id,
            error=str(e),
        )

    # Cleanup desktop sandbox
    try:
        desktop_manager = get_desktop_sandbox_manager()
        results["desktop"] = await desktop_manager.cleanup_session(user_id=user_id, task_id=task_id)
    except Exception as e:
        cleanup_errors["desktop"] = str(e)
        logger.warning(
            "desktop_sandbox_cleanup_failed",
            user_id=user_id,
            task_id=task_id,
            error=str(e),
        )

    # Cleanup execution sandbox
    try:
        execution_manager = get_execution_sandbox_manager()
        results["execution"] = await execution_manager.cleanup_session(
            user_id=user_id, task_id=task_id
        )
    except Exception as e:
        cleanup_errors["execution"] = str(e)
        logger.warning(
            "execution_sandbox_cleanup_failed",
            user_id=user_id,
            task_id=task_id,
            error=str(e),
        )

    # Cleanup app sandbox
    try:
        app_manager = get_app_sandbox_manager()
        results["app"] = await app_manager.cleanup_session(user_id=user_id, task_id=task_id)
    except Exception as e:
        cleanup_errors["app"] = str(e)
        logger.warning(
            "app_sandbox_cleanup_failed",
            user_id=user_id,
            task_id=task_id,
            error=str(e),
        )

    if cleanup_errors:
        logger.error(
            "sandbox_cleanup_partial_failure",
            user_id=user_id,
            task_id=task_id,
            failed_types=list(cleanup_errors.keys()),
            errors=cleanup_errors,
        )

    if any(results.values()):
        logger.info(
            "sandboxes_cleaned_on_disconnect",
            user_id=user_id,
            task_id=task_id,
            results=results,
        )

    return results


__all__ = [
    # Execution sandbox management
    "ExecutionSandboxSession",
    "ExecutionSandboxManager",
    "get_execution_sandbox_manager",
    # Desktop sandbox management
    "DesktopSandboxSession",
    "DesktopSandboxManager",
    "get_desktop_sandbox_manager",
    # App sandbox management
    "AppSandboxSession",
    "AppSandboxManager",
    "get_app_sandbox_manager",
    "APP_TEMPLATES",
    # Unified sandbox management
    "UnifiedSandboxManager",
    "UnifiedSandboxSession",
    "get_unified_sandbox_manager",
    # Sandbox file operations
    "sandbox_file",
    "sandbox_file_with_context",
    "file_operations",
    # Metrics
    "get_sandbox_metrics",
    # Cleanup
    "cleanup_sandboxes_for_task",
    # Availability checks
    "is_execution_sandbox_available",
    "is_desktop_sandbox_available",
    "is_app_sandbox_available",
    "get_sandbox_availability",
]
