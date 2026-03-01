"""Unified Sandbox Manager.

Provides a single shared SandboxRuntime for both code execution and app
development within one agent run.  Desktop sandboxes stay separate because
they require a different VM image (Xvfb).

Usage:
    manager = get_unified_sandbox_manager()
    runtime = await manager.get_or_create_runtime(user_id, task_id)
    executor = await manager.get_code_executor(user_id, task_id)
    app_session = await manager.get_app_session(user_id, task_id, template="react")
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.logging import get_logger
from app.sandbox.runtime import SandboxRuntime

logger = get_logger(__name__)

# Default unified session timeout (30 minutes)
DEFAULT_UNIFIED_TIMEOUT = timedelta(minutes=30)

# Cleanup interval for expired sessions (60 seconds)
CLEANUP_INTERVAL_SECONDS = 60

# Skip health checks if last successful check was less than this many seconds ago
HEALTH_CHECK_SKIP_SECONDS = 30


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


@dataclass
class UnifiedSandboxSession:
    """Tracks a unified sandbox session that shares a single runtime."""

    runtime: SandboxRuntime
    session_key: str
    created_at: datetime = field(default_factory=_utcnow)
    last_accessed: datetime = field(default_factory=_utcnow)
    timeout: timedelta = field(default_factory=lambda: DEFAULT_UNIFIED_TIMEOUT)

    # Timestamp of last successful health check (monotonic clock)
    last_health_check: float = 0.0

    @property
    def sandbox_id(self) -> str | None:
        """Get the underlying sandbox ID."""
        if self.runtime:
            return self.runtime.sandbox_id
        return None

    @property
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return datetime.now(timezone.utc) > (self.last_accessed + self.timeout)

    def touch(self) -> None:
        """Update last accessed time to prevent expiry."""
        self.last_accessed = datetime.now(timezone.utc)


class UnifiedSandboxManager:
    """Manages a single shared SandboxRuntime per user+task pair.

    Both code execution and app development use the same underlying sandbox,
    avoiding the overhead of spinning up separate VMs for the same agent run.
    Desktop sandboxes remain separate due to different image requirements.
    """

    _instance: "UnifiedSandboxManager | None" = None

    def __init__(self) -> None:
        """Initialize the unified sandbox manager."""
        self._sessions: dict[str, UnifiedSandboxSession] = {}
        self._cleanup_task: asyncio.Task[None] | None = None
        self._session_lock = asyncio.Lock()
        # Metrics counters
        self._total_created: int = 0
        self._total_cleaned: int = 0
        self._total_reused: int = 0
        self._health_check_failures: int = 0

    @classmethod
    def get_instance(cls) -> "UnifiedSandboxManager":
        """Get the singleton instance of UnifiedSandboxManager."""
        if cls._instance is None:
            cls._instance = UnifiedSandboxManager()
        return cls._instance

    @staticmethod
    def make_session_key(user_id: str | None, task_id: str | None) -> str:
        """Create a session key from user and task IDs.

        Args:
            user_id: User identifier
            task_id: Task identifier

        Returns:
            Session key string with unified prefix
        """
        user = user_id or "anonymous"
        task = task_id or "default"
        return f"unified:{user}:{task}"

    async def get_or_create_runtime(
        self,
        user_id: str | None = None,
        task_id: str | None = None,
        timeout: timedelta | None = None,
    ) -> SandboxRuntime:
        """Get or create the shared SandboxRuntime for a user+task pair.

        Args:
            user_id: User identifier
            task_id: Task identifier
            timeout: Session timeout (defaults to 30 minutes)

        Returns:
            Shared SandboxRuntime instance

        Raises:
            ValueError: If sandbox provider is not available
        """
        session = await self._get_or_create_session(user_id, task_id, timeout)
        return session.runtime

    async def get_code_executor(
        self,
        user_id: str | None = None,
        task_id: str | None = None,
        timeout: timedelta | None = None,
    ) -> "BaseCodeExecutor":
        """Get a code executor backed by the shared runtime.

        Args:
            user_id: User identifier
            task_id: Task identifier
            timeout: Session timeout

        Returns:
            BaseCodeExecutor wrapping the shared runtime
        """
        from app.sandbox.execution_sandbox_manager import ExecutionSandboxSession

        session = await self._get_or_create_session(user_id, task_id, timeout)

        # Create an ExecutionSandboxSession wrapping the shared runtime
        exec_session = await self._get_or_create_execution_session(
            user_id, task_id, session.runtime
        )
        return exec_session.executor

    async def get_app_session(
        self,
        user_id: str | None = None,
        task_id: str | None = None,
        template: str = "react",
        timeout: timedelta | None = None,
    ) -> "AppSandboxSession":
        """Get an app sandbox session backed by the shared runtime.

        Args:
            user_id: User identifier
            task_id: Task identifier
            template: App template to use
            timeout: Session timeout

        Returns:
            AppSandboxSession wrapping the shared runtime
        """
        from app.sandbox.app_sandbox_manager import AppSandboxSession

        session = await self._get_or_create_session(user_id, task_id, timeout)

        app_session = AppSandboxSession(
            sandbox=session.runtime,
            session_key=f"unified-app:{user_id or 'anonymous'}:{task_id or 'default'}",
            template=template,
        )
        # Mark as healthy since the runtime was just verified
        app_session.last_health_check = time.monotonic()

        return app_session

    async def _get_or_create_session(
        self,
        user_id: str | None,
        task_id: str | None,
        timeout: timedelta | None,
    ) -> UnifiedSandboxSession:
        """Get an existing session or create a new one.

        Args:
            user_id: User identifier
            task_id: Task identifier
            timeout: Session timeout

        Returns:
            UnifiedSandboxSession with active runtime

        Raises:
            ValueError: If sandbox provider is not available
        """
        from app.sandbox.provider import is_provider_available

        available, issue = is_provider_available("execution")
        if not available:
            raise ValueError(issue)

        session_key = self.make_session_key(user_id, task_id)
        session_timeout = timeout or DEFAULT_UNIFIED_TIMEOUT

        async with self._session_lock:
            # Check for existing valid session
            if session_key in self._sessions:
                session = self._sessions[session_key]
                if not session.is_expired and await self._is_sandbox_healthy(session):
                    session.touch()
                    self._total_reused += 1
                    logger.info(
                        "unified_sandbox_session_reused",
                        session_key=session_key,
                        sandbox_id=session.sandbox_id,
                    )
                    return session
                else:
                    # Clean up unhealthy or expired session
                    await self._cleanup_session_internal(session_key)

            # Create new shared runtime via provider factory
            from app.sandbox.provider import create_app_runtime

            runtime = await create_app_runtime()

            session = UnifiedSandboxSession(
                runtime=runtime,
                session_key=session_key,
                timeout=session_timeout,
            )
            # Mark as healthy since we just created it
            session.last_health_check = time.monotonic()
            self._sessions[session_key] = session
            self._total_created += 1

            logger.info(
                "unified_sandbox_session_created",
                session_key=session_key,
                sandbox_id=session.sandbox_id,
            )

            # Start cleanup task if not running
            self._ensure_cleanup_task()

            return session

    async def _get_or_create_execution_session(
        self,
        user_id: str | None,
        task_id: str | None,
        runtime: SandboxRuntime,
    ) -> "ExecutionSandboxSession":
        """Create an execution session wrapping the shared runtime.

        Uses the execution sandbox manager's from_runtime factory method
        to create a code executor that operates on the shared runtime.

        Args:
            user_id: User identifier
            task_id: Task identifier
            runtime: Shared SandboxRuntime

        Returns:
            ExecutionSandboxSession backed by the shared runtime
        """
        from app.sandbox.execution_sandbox_manager import (
            ExecutionSandboxManager,
            ExecutionSandboxSession,
        )

        manager = ExecutionSandboxManager.get_instance()
        return await manager.get_or_create_sandbox_with_runtime(
            user_id=user_id,
            task_id=task_id,
            runtime=runtime,
        )

    async def get_session(
        self,
        user_id: str | None = None,
        task_id: str | None = None,
    ) -> UnifiedSandboxSession | None:
        """Get an existing session without creating one.

        Args:
            user_id: User identifier
            task_id: Task identifier

        Returns:
            UnifiedSandboxSession if exists and valid, None otherwise
        """
        session_key = self.make_session_key(user_id, task_id)

        async with self._session_lock:
            session = self._sessions.get(session_key)
            if session and not session.is_expired and await self._is_sandbox_healthy(session):
                session.touch()
                return session
            return None

    async def _is_sandbox_healthy(self, session: UnifiedSandboxSession) -> bool:
        """Check if a sandbox session is still healthy and responsive.

        Skips the actual health check command if the last successful check
        was less than HEALTH_CHECK_SKIP_SECONDS ago.

        Args:
            session: The session to check

        Returns:
            True if sandbox is healthy, False otherwise
        """
        if not session.runtime:
            return False

        # Skip health check if last successful check was recent
        now = time.monotonic()
        recently_checked = (
            session.last_health_check > 0
            and (now - session.last_health_check) < HEALTH_CHECK_SKIP_SECONDS
        )
        if recently_checked:
            return True

        try:
            result = await session.runtime.run_command(
                "echo 'health_check'",
                timeout=5,
            )
            healthy = result.exit_code == 0 and "health_check" in result.stdout
            if healthy:
                session.last_health_check = now
            return healthy
        except Exception as e:
            self._health_check_failures += 1
            logger.warning(
                "unified_sandbox_health_check_failed",
                session_key=session.session_key,
                sandbox_id=session.sandbox_id,
                error=str(e),
            )
            return False

    async def cleanup_session(
        self,
        user_id: str | None = None,
        task_id: str | None = None,
    ) -> bool:
        """Explicitly clean up a sandbox session.

        Args:
            user_id: User identifier
            task_id: Task identifier

        Returns:
            True if session was cleaned up, False if not found
        """
        session_key = self.make_session_key(user_id, task_id)

        async with self._session_lock:
            return await self._cleanup_session_internal(session_key)

    async def _cleanup_session_internal(self, session_key: str) -> bool:
        """Internal cleanup method (assumes lock is held).

        Args:
            session_key: Session key to clean up

        Returns:
            True if session was cleaned up, False if not found
        """
        session = self._sessions.get(session_key)
        if session is None:
            return False

        try:
            await session.runtime.kill()
            self._total_cleaned += 1
            logger.info(
                "unified_sandbox_session_cleaned",
                session_key=session_key,
                sandbox_id=session.sandbox_id,
            )
        except Exception as e:
            self._total_cleaned += 1
            logger.error(
                "unified_sandbox_session_cleanup_failed_sandbox_may_be_leaked",
                session_key=session_key,
                sandbox_id=session.sandbox_id,
                error=str(e),
            )
        finally:
            self._sessions.pop(session_key, None)

        return True

    async def cleanup_expired(self) -> int:
        """Clean up all expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        cleaned = 0

        async with self._session_lock:
            expired_keys = [
                key for key, session in self._sessions.items() if session.is_expired
            ]
            for key in expired_keys:
                if await self._cleanup_session_internal(key):
                    cleaned += 1

        if cleaned > 0:
            logger.info("unified_expired_sessions_cleaned", count=cleaned)

        return cleaned

    async def cleanup_all(self) -> int:
        """Clean up all sessions.

        Returns:
            Number of sessions cleaned up
        """
        cleaned = 0

        async with self._session_lock:
            keys = list(self._sessions.keys())
            for key in keys:
                if await self._cleanup_session_internal(key):
                    cleaned += 1

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        logger.info("unified_all_sessions_cleaned", count=cleaned)
        return cleaned

    def _ensure_cleanup_task(self) -> None:
        """Ensure the background cleanup task is running."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """Background loop that periodically cleans up expired sessions."""
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("unified_cleanup_loop_error", error=str(e))

    @property
    def active_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self._sessions)

    def get_metrics(self) -> dict[str, Any]:
        """Get sandbox manager metrics.

        Returns:
            Dict with metrics
        """
        return {
            "active_sessions": len(self._sessions),
            "total_created": self._total_created,
            "total_cleaned": self._total_cleaned,
            "total_reused": self._total_reused,
            "health_check_failures": self._health_check_failures,
        }

    def get_session_info(self) -> list[dict[str, Any]]:
        """Get information about all active sessions for debugging.

        Returns:
            List of session info dicts
        """
        return [
            {
                "session_key": session.session_key,
                "sandbox_id": session.sandbox_id,
                "created_at": session.created_at.isoformat(),
                "last_accessed": session.last_accessed.isoformat(),
                "is_expired": session.is_expired,
            }
            for session in self._sessions.values()
        ]


def get_unified_sandbox_manager() -> UnifiedSandboxManager:
    """Get the global UnifiedSandboxManager instance.

    Returns:
        UnifiedSandboxManager singleton
    """
    return UnifiedSandboxManager.get_instance()
