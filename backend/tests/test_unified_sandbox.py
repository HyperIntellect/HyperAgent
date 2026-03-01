"""Tests for the Unified Sandbox Manager."""

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.sandbox.runtime import CommandResult
from app.sandbox.unified_sandbox_manager import (
    UnifiedSandboxManager,
    UnifiedSandboxSession,
    get_unified_sandbox_manager,
)


def _make_mock_runtime(sandbox_id: str = "test-sandbox-123") -> MagicMock:
    """Create a mock SandboxRuntime."""
    runtime = MagicMock()
    runtime.sandbox_id = sandbox_id
    runtime.run_command = AsyncMock(
        return_value=CommandResult(exit_code=0, stdout="health_check\n", stderr="")
    )
    runtime.kill = AsyncMock()
    runtime.read_file = AsyncMock(return_value=b"file content")
    runtime.write_file = AsyncMock()
    runtime.get_host_url = AsyncMock(return_value="http://localhost:3000")
    return runtime


class TestUnifiedSandboxSession:
    """Tests for UnifiedSandboxSession dataclass."""

    def test_session_key(self):
        runtime = _make_mock_runtime()
        session = UnifiedSandboxSession(
            runtime=runtime,
            session_key="unified:user1:task1",
        )
        assert session.session_key == "unified:user1:task1"
        assert session.sandbox_id == "test-sandbox-123"

    def test_session_not_expired(self):
        runtime = _make_mock_runtime()
        session = UnifiedSandboxSession(
            runtime=runtime,
            session_key="test",
            timeout=timedelta(minutes=30),
        )
        assert not session.is_expired

    def test_session_expired(self):
        runtime = _make_mock_runtime()
        session = UnifiedSandboxSession(
            runtime=runtime,
            session_key="test",
            timeout=timedelta(seconds=0),
        )
        assert session.is_expired

    def test_touch_updates_last_accessed(self):
        runtime = _make_mock_runtime()
        session = UnifiedSandboxSession(
            runtime=runtime,
            session_key="test",
        )
        original = session.last_accessed
        session.touch()
        assert session.last_accessed >= original


class TestUnifiedSandboxManager:
    """Tests for UnifiedSandboxManager."""

    def setup_method(self):
        """Reset singleton before each test."""
        UnifiedSandboxManager._instance = None
        UnifiedSandboxManager._lock = None

    def test_make_session_key(self):
        key = UnifiedSandboxManager.make_session_key("user1", "task1")
        assert key == "unified:user1:task1"

    def test_make_session_key_defaults(self):
        key = UnifiedSandboxManager.make_session_key(None, None)
        assert key == "unified:anonymous:default"

    def test_singleton(self):
        m1 = UnifiedSandboxManager.get_instance()
        m2 = UnifiedSandboxManager.get_instance()
        assert m1 is m2

    def test_get_unified_sandbox_manager(self):
        m = get_unified_sandbox_manager()
        assert isinstance(m, UnifiedSandboxManager)

    def test_initial_metrics(self):
        manager = UnifiedSandboxManager()
        metrics = manager.get_metrics()
        assert metrics["active_sessions"] == 0
        assert metrics["total_created"] == 0
        assert metrics["total_cleaned"] == 0
        assert metrics["total_reused"] == 0
        assert metrics["health_check_failures"] == 0

    @pytest.mark.asyncio
    async def test_get_or_create_runtime(self):
        """Test creating a new runtime."""
        manager = UnifiedSandboxManager()
        mock_runtime = _make_mock_runtime()

        with (
            patch(
                "app.sandbox.provider.is_provider_available",
                return_value=(True, ""),
            ),
            patch(
                "app.sandbox.provider.create_app_runtime",
                new_callable=AsyncMock,
                return_value=mock_runtime,
            ),
        ):
            runtime = await manager.get_or_create_runtime("user1", "task1")
            assert runtime is mock_runtime
            assert manager.active_session_count == 1
            assert manager.get_metrics()["total_created"] == 1

    @pytest.mark.asyncio
    async def test_runtime_reuse(self):
        """Test that the same runtime is reused for the same user+task."""
        manager = UnifiedSandboxManager()
        mock_runtime = _make_mock_runtime()

        with (
            patch(
                "app.sandbox.provider.is_provider_available",
                return_value=(True, ""),
            ),
            patch(
                "app.sandbox.provider.create_app_runtime",
                new_callable=AsyncMock,
                return_value=mock_runtime,
            ),
        ):
            runtime1 = await manager.get_or_create_runtime("user1", "task1")
            runtime2 = await manager.get_or_create_runtime("user1", "task1")
            assert runtime1 is runtime2
            assert manager.active_session_count == 1
            assert manager.get_metrics()["total_reused"] == 1

    @pytest.mark.asyncio
    async def test_different_tasks_get_different_runtimes(self):
        """Test that different task IDs get different runtimes."""
        manager = UnifiedSandboxManager()
        mock_runtime_1 = _make_mock_runtime("sandbox-1")
        mock_runtime_2 = _make_mock_runtime("sandbox-2")

        with (
            patch(
                "app.sandbox.provider.is_provider_available",
                return_value=(True, ""),
            ),
            patch(
                "app.sandbox.provider.create_app_runtime",
                new_callable=AsyncMock,
                side_effect=[mock_runtime_1, mock_runtime_2],
            ),
        ):
            runtime1 = await manager.get_or_create_runtime("user1", "task1")
            runtime2 = await manager.get_or_create_runtime("user1", "task2")
            assert runtime1 is not runtime2
            assert manager.active_session_count == 2

    @pytest.mark.asyncio
    async def test_provider_unavailable_raises(self):
        """Test that ValueError is raised when provider is unavailable."""
        manager = UnifiedSandboxManager()

        with patch(
            "app.sandbox.provider.is_provider_available",
            return_value=(False, "E2B API key not configured"),
        ):
            with pytest.raises(ValueError, match="E2B API key not configured"):
                await manager.get_or_create_runtime("user1", "task1")

    @pytest.mark.asyncio
    async def test_cleanup_session(self):
        """Test cleaning up a session."""
        manager = UnifiedSandboxManager()
        mock_runtime = _make_mock_runtime()

        with (
            patch(
                "app.sandbox.provider.is_provider_available",
                return_value=(True, ""),
            ),
            patch(
                "app.sandbox.provider.create_app_runtime",
                new_callable=AsyncMock,
                return_value=mock_runtime,
            ),
        ):
            await manager.get_or_create_runtime("user1", "task1")
            assert manager.active_session_count == 1

            cleaned = await manager.cleanup_session("user1", "task1")
            assert cleaned is True
            assert manager.active_session_count == 0
            mock_runtime.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_session(self):
        """Test cleaning up a session that doesn't exist."""
        manager = UnifiedSandboxManager()
        cleaned = await manager.cleanup_session("user1", "task1")
        assert cleaned is False

    @pytest.mark.asyncio
    async def test_cleanup_all(self):
        """Test cleaning up all sessions."""
        manager = UnifiedSandboxManager()
        mock_runtime_1 = _make_mock_runtime("sandbox-1")
        mock_runtime_2 = _make_mock_runtime("sandbox-2")

        with (
            patch(
                "app.sandbox.provider.is_provider_available",
                return_value=(True, ""),
            ),
            patch(
                "app.sandbox.provider.create_app_runtime",
                new_callable=AsyncMock,
                side_effect=[mock_runtime_1, mock_runtime_2],
            ),
        ):
            await manager.get_or_create_runtime("user1", "task1")
            await manager.get_or_create_runtime("user1", "task2")
            assert manager.active_session_count == 2

            cleaned = await manager.cleanup_all()
            assert cleaned == 2
            assert manager.active_session_count == 0

    @pytest.mark.asyncio
    async def test_get_session_returns_none_when_empty(self):
        """Test get_session returns None when no session exists."""
        manager = UnifiedSandboxManager()
        session = await manager.get_session("user1", "task1")
        assert session is None

    @pytest.mark.asyncio
    async def test_get_session_returns_session(self):
        """Test get_session returns session when it exists."""
        manager = UnifiedSandboxManager()
        mock_runtime = _make_mock_runtime()

        with (
            patch(
                "app.sandbox.provider.is_provider_available",
                return_value=(True, ""),
            ),
            patch(
                "app.sandbox.provider.create_app_runtime",
                new_callable=AsyncMock,
                return_value=mock_runtime,
            ),
        ):
            await manager.get_or_create_runtime("user1", "task1")
            session = await manager.get_session("user1", "task1")
            assert session is not None
            assert session.sandbox_id == "test-sandbox-123"

    @pytest.mark.asyncio
    async def test_unhealthy_sandbox_replaced(self):
        """Test that an unhealthy sandbox is replaced with a new one."""
        manager = UnifiedSandboxManager()
        mock_runtime_old = _make_mock_runtime("sandbox-old")
        mock_runtime_new = _make_mock_runtime("sandbox-new")

        with (
            patch(
                "app.sandbox.provider.is_provider_available",
                return_value=(True, ""),
            ),
            patch(
                "app.sandbox.provider.create_app_runtime",
                new_callable=AsyncMock,
                side_effect=[mock_runtime_old, mock_runtime_new],
            ),
        ):
            # Create first runtime
            runtime1 = await manager.get_or_create_runtime("user1", "task1")
            assert runtime1 is mock_runtime_old

            # Make the old runtime unhealthy and reset health check timestamp
            # so the next call performs an actual health check
            mock_runtime_old.run_command = AsyncMock(
                side_effect=Exception("sandbox dead")
            )
            session_key = manager.make_session_key("user1", "task1")
            manager._sessions[session_key].last_health_check = 0.0

            # Should create a new runtime
            runtime2 = await manager.get_or_create_runtime("user1", "task1")
            assert runtime2 is mock_runtime_new
            assert manager.active_session_count == 1

    def test_get_session_info(self):
        """Test get_session_info returns session details."""
        manager = UnifiedSandboxManager()
        runtime = _make_mock_runtime()
        session = UnifiedSandboxSession(
            runtime=runtime,
            session_key="unified:user1:task1",
        )
        manager._sessions["unified:user1:task1"] = session

        info = manager.get_session_info()
        assert len(info) == 1
        assert info[0]["session_key"] == "unified:user1:task1"
        assert info[0]["sandbox_id"] == "test-sandbox-123"

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test cleaning up expired sessions."""
        manager = UnifiedSandboxManager()
        mock_runtime = _make_mock_runtime()

        # Create a session with 0 timeout (immediately expired)
        session = UnifiedSandboxSession(
            runtime=mock_runtime,
            session_key="unified:user1:task1",
            timeout=timedelta(seconds=0),
        )
        manager._sessions["unified:user1:task1"] = session

        cleaned = await manager.cleanup_expired()
        assert cleaned == 1
        assert manager.active_session_count == 0


class TestUnifiedSandboxManagerGetAppSession:
    """Tests for the get_app_session method."""

    def setup_method(self):
        UnifiedSandboxManager._instance = None
        UnifiedSandboxManager._lock = None

    @pytest.mark.asyncio
    async def test_get_app_session(self):
        """Test getting an app session backed by unified runtime."""
        manager = UnifiedSandboxManager()
        mock_runtime = _make_mock_runtime()

        with (
            patch(
                "app.sandbox.provider.is_provider_available",
                return_value=(True, ""),
            ),
            patch(
                "app.sandbox.provider.create_app_runtime",
                new_callable=AsyncMock,
                return_value=mock_runtime,
            ),
        ):
            app_session = await manager.get_app_session("user1", "task1", template="react")
            assert app_session.sandbox is mock_runtime
            assert app_session.template == "react"
            assert "unified-app" in app_session.session_key


class TestUnifiedSandboxManagerGetCodeExecutor:
    """Tests for the get_code_executor method."""

    def setup_method(self):
        UnifiedSandboxManager._instance = None
        UnifiedSandboxManager._lock = None
        # Also reset the execution sandbox manager singleton
        from app.sandbox.execution_sandbox_manager import ExecutionSandboxManager
        ExecutionSandboxManager._instance = None
        ExecutionSandboxManager._lock = None

    @pytest.mark.asyncio
    async def test_get_code_executor(self):
        """Test getting a code executor backed by unified runtime."""
        manager = UnifiedSandboxManager()
        mock_runtime = _make_mock_runtime()
        mock_executor = MagicMock()
        mock_executor.sandbox_id = "test-sandbox-123"
        mock_executor.set_runtime = MagicMock()

        with (
            patch(
                "app.sandbox.provider.is_provider_available",
                return_value=(True, ""),
            ),
            patch(
                "app.sandbox.provider.create_app_runtime",
                new_callable=AsyncMock,
                return_value=mock_runtime,
            ),
            patch(
                "app.sandbox.provider.create_code_executor",
                return_value=mock_executor,
            ),
            patch(
                "app.sandbox.provider.is_provider_available",
                return_value=(True, ""),
            ),
        ):
            executor = await manager.get_code_executor("user1", "task1")
            assert executor is mock_executor
            mock_executor.set_runtime.assert_called_once_with(mock_runtime)
