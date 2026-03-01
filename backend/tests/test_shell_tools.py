"""Tests for shell / process management tools."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.tools.shell_tools import (
    _background_sessions,
    shell_exec,
    shell_kill,
    shell_view,
    shell_wait,
)
from app.sandbox.runtime import CommandResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_runtime():
    """Create a mock SandboxRuntime."""
    runtime = AsyncMock()
    runtime.sandbox_id = "test-sandbox-123"
    return runtime


@pytest.fixture
def mock_session(mock_runtime):
    """Create a mock ExecutionSandboxSession."""
    executor = MagicMock()
    executor.sandbox_id = "test-sandbox-123"
    executor.get_runtime.return_value = mock_runtime

    session = MagicMock()
    session.executor = executor
    session.sandbox_id = "test-sandbox-123"
    return session


@pytest.fixture
def mock_sandbox_manager(mock_session):
    """Create a mock ExecutionSandboxManager."""
    manager = AsyncMock()
    manager.get_or_create_sandbox.return_value = mock_session
    return manager


@pytest.fixture(autouse=True)
def clean_background_sessions():
    """Clean up background sessions before and after each test."""
    _background_sessions.clear()
    yield
    # Cancel any remaining tasks
    for sid, info in list(_background_sessions.items()):
        task = info.get("task")
        if task and not task.done():
            task.cancel()
    _background_sessions.clear()


def _patch_sandbox(mock_sandbox_manager):
    """Return patch context managers for sandbox availability and manager."""
    return (
        patch(
            "app.agents.tools.shell_tools.get_execution_sandbox_manager",
            return_value=mock_sandbox_manager,
        ),
        patch(
            "app.sandbox.provider.is_provider_available",
            return_value=(True, None),
        ),
    )


# ---------------------------------------------------------------------------
# shell_exec tests
# ---------------------------------------------------------------------------


class TestShellExec:
    """Tests for shell_exec tool."""

    @pytest.mark.asyncio
    async def test_sync_execution_success(self, mock_sandbox_manager, mock_runtime):
        """Successfully execute a command synchronously."""
        mock_runtime.run_command.return_value = CommandResult(
            exit_code=0,
            stdout="hello world\n",
            stderr="",
        )
        p1, p2 = _patch_sandbox(mock_sandbox_manager)

        with p1, p2:
            result = json.loads(
                await shell_exec.ainvoke({"command": "echo hello world"})
            )

        assert result["success"] is True
        assert result["stdout"] == "hello world\n"
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_sync_execution_error(self, mock_sandbox_manager, mock_runtime):
        """Command that fails returns error exit code."""
        mock_runtime.run_command.return_value = CommandResult(
            exit_code=1,
            stdout="",
            stderr="command not found",
        )
        p1, p2 = _patch_sandbox(mock_sandbox_manager)

        with p1, p2:
            result = json.loads(
                await shell_exec.ainvoke({"command": "nonexistent_cmd"})
            )

        assert result["success"] is True  # Tool succeeded, command failed
        assert result["exit_code"] == 1
        assert result["stderr"] == "command not found"

    @pytest.mark.asyncio
    async def test_background_execution(self, mock_sandbox_manager, mock_runtime):
        """Background execution returns session_id immediately."""
        # Make run_command take a while so we can verify it's non-blocking
        async def slow_command(*args, **kwargs):
            await asyncio.sleep(10)
            return CommandResult(exit_code=0, stdout="done", stderr="")

        mock_runtime.run_command.side_effect = slow_command
        p1, p2 = _patch_sandbox(mock_sandbox_manager)

        with p1, p2:
            result = json.loads(
                await shell_exec.ainvoke({
                    "command": "npm start",
                    "background": True,
                })
            )

        assert result["success"] is True
        assert "session_id" in result
        assert result["session_id"] in _background_sessions

    @pytest.mark.asyncio
    async def test_background_execution_custom_session_id(self, mock_sandbox_manager, mock_runtime):
        """Background execution with custom session_id."""
        async def slow_command(*args, **kwargs):
            await asyncio.sleep(10)
            return CommandResult(exit_code=0, stdout="done", stderr="")

        mock_runtime.run_command.side_effect = slow_command
        p1, p2 = _patch_sandbox(mock_sandbox_manager)

        with p1, p2:
            result = json.loads(
                await shell_exec.ainvoke({
                    "command": "npm start",
                    "background": True,
                    "session_id": "my-server",
                })
            )

        assert result["success"] is True
        assert result["session_id"] == "my-server"

    @pytest.mark.asyncio
    async def test_sandbox_unavailable(self, mock_sandbox_manager):
        """Returns error when sandbox is unavailable."""
        mock_sandbox_manager.get_or_create_sandbox.side_effect = RuntimeError(
            "No sandbox configured"
        )
        with patch(
            "app.agents.tools.shell_tools.get_execution_sandbox_manager",
            return_value=mock_sandbox_manager,
        ):
            result = json.loads(
                await shell_exec.ainvoke({"command": "echo hi"})
            )

        assert result["success"] is False
        assert "No sandbox configured" in result["error"]


# ---------------------------------------------------------------------------
# shell_view tests
# ---------------------------------------------------------------------------


class TestShellView:
    """Tests for shell_view tool."""

    @pytest.mark.asyncio
    async def test_view_running_session(self):
        """View output of a running background session."""
        _background_sessions["test-session"] = {
            "command": "npm start",
            "output": "Server starting...\n",
            "exit_code": None,
            "done": False,
            "task": None,
        }

        result = json.loads(
            await shell_view.ainvoke({"session_id": "test-session"})
        )

        assert result["success"] is True
        assert result["done"] is False
        assert result["output"] == "Server starting...\n"

    @pytest.mark.asyncio
    async def test_view_completed_session(self):
        """View output of a completed background session."""
        _background_sessions["done-session"] = {
            "command": "ls -la",
            "output": "file1.txt\nfile2.txt\n",
            "exit_code": 0,
            "done": True,
            "task": None,
        }

        result = json.loads(
            await shell_view.ainvoke({"session_id": "done-session"})
        )

        assert result["success"] is True
        assert result["done"] is True
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_view_not_found(self):
        """View returns error for unknown session_id."""
        result = json.loads(
            await shell_view.ainvoke({"session_id": "nonexistent"})
        )

        assert result["success"] is False
        assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# shell_wait tests
# ---------------------------------------------------------------------------


class TestShellWait:
    """Tests for shell_wait tool."""

    @pytest.mark.asyncio
    async def test_wait_already_completed(self):
        """Waiting on an already-completed session returns immediately."""
        _background_sessions["done-session"] = {
            "command": "echo done",
            "output": "done\n",
            "exit_code": 0,
            "done": True,
            "task": None,
        }

        result = json.loads(
            await shell_wait.ainvoke({"session_id": "done-session"})
        )

        assert result["success"] is True
        assert result["already_done"] is True
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_wait_completes_in_time(self):
        """Waiting on a process that completes within timeout."""
        async def _bg_work():
            await asyncio.sleep(0.1)

        bg_task = asyncio.create_task(_bg_work())
        _background_sessions["fast-session"] = {
            "command": "echo fast",
            "output": "",
            "exit_code": None,
            "done": False,
            "task": bg_task,
        }

        # Simulate the task completing and updating session
        async def update_after_done():
            await bg_task
            _background_sessions["fast-session"]["done"] = True
            _background_sessions["fast-session"]["output"] = "fast\n"
            _background_sessions["fast-session"]["exit_code"] = 0

        asyncio.create_task(update_after_done())

        result = json.loads(
            await shell_wait.ainvoke({
                "session_id": "fast-session",
                "timeout": 5,
            })
        )

        assert result["success"] is True
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_wait_timeout(self):
        """Waiting on a process that exceeds timeout."""
        async def _slow_work():
            await asyncio.sleep(60)

        bg_task = asyncio.create_task(_slow_work())
        _background_sessions["slow-session"] = {
            "command": "sleep 60",
            "output": "",
            "exit_code": None,
            "done": False,
            "task": bg_task,
        }

        result = json.loads(
            await shell_wait.ainvoke({
                "session_id": "slow-session",
                "timeout": 1,
            })
        )

        assert result["success"] is True
        assert result["timed_out"] is True
        assert "still running" in result["message"].lower() or "running" in result["message"].lower()

        # Clean up
        bg_task.cancel()
        try:
            await bg_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_wait_not_found(self):
        """Waiting on unknown session returns error."""
        result = json.loads(
            await shell_wait.ainvoke({"session_id": "nonexistent"})
        )

        assert result["success"] is False
        assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# shell_kill tests
# ---------------------------------------------------------------------------


class TestShellKill:
    """Tests for shell_kill tool."""

    @pytest.mark.asyncio
    async def test_kill_running_process(self):
        """Kill a running background process."""
        async def _slow_work():
            await asyncio.sleep(60)

        bg_task = asyncio.create_task(_slow_work())
        _background_sessions["running-session"] = {
            "command": "npm start",
            "output": "",
            "exit_code": None,
            "done": False,
            "task": bg_task,
        }

        result = json.loads(
            await shell_kill.ainvoke({"session_id": "running-session"})
        )

        assert result["success"] is True
        assert "terminated" in result["message"].lower() or "cleaned up" in result["message"].lower()
        assert "running-session" not in _background_sessions

    @pytest.mark.asyncio
    async def test_kill_completed_process(self):
        """Kill an already-completed process cleans it up."""
        _background_sessions["done-session"] = {
            "command": "echo done",
            "output": "done\n",
            "exit_code": 0,
            "done": True,
            "task": None,
        }

        result = json.loads(
            await shell_kill.ainvoke({"session_id": "done-session"})
        )

        assert result["success"] is True
        assert "done-session" not in _background_sessions

    @pytest.mark.asyncio
    async def test_kill_not_found(self):
        """Kill returns error for unknown session_id."""
        result = json.loads(
            await shell_kill.ainvoke({"session_id": "nonexistent"})
        )

        assert result["success"] is False
        assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# Registry integration tests
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Tests that shell tools are properly registered."""

    def test_shell_category_exists(self):
        """SHELL category should exist in ToolCategory."""
        from app.agents.tools.registry import ToolCategory

        assert hasattr(ToolCategory, "SHELL")
        assert ToolCategory.SHELL.value == "shell"

    def test_shell_tools_in_catalog(self):
        """Shell tools should be in the TOOL_CATALOG."""
        from app.agents.tools.registry import TOOL_CATALOG, ToolCategory

        shell_tools = TOOL_CATALOG.get(ToolCategory.SHELL, [])
        tool_names = {t.name for t in shell_tools}

        assert "shell_exec" in tool_names
        assert "shell_view" in tool_names
        assert "shell_wait" in tool_names
        assert "shell_kill" in tool_names

    def test_shell_in_task_agent_mapping(self):
        """SHELL should be in the TASK agent's tool mapping."""
        from app.agents.state import AgentType
        from app.agents.tools.registry import AGENT_TOOL_MAPPING, ToolCategory

        task_categories = AGENT_TOOL_MAPPING[AgentType.TASK.value]
        assert ToolCategory.SHELL in task_categories
