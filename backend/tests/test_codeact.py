"""Tests for the CodeAct hybrid execution tool."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.tools.codeact import (
    DEFAULT_SCRIPT_TIMEOUT,
    _HYPERAGENT_LIB_INSTALLED,
    _ensure_hyperagent_lib,
    execute_script,
)
from app.agents.tools.registry import (
    TOOL_CATALOG,
    TOOL_CONTRACTS,
    ToolCategory,
    get_tools_for_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(sandbox_id: str = "sbx-codeact-1") -> MagicMock:
    """Create a mock ExecutionSandboxSession."""
    session = MagicMock()
    session.sandbox_id = sandbox_id

    runtime = AsyncMock()
    runtime.sandbox_id = sandbox_id
    runtime.run_command = AsyncMock(
        return_value=SimpleNamespace(exit_code=0, stdout="", stderr="")
    )
    runtime.write_file = AsyncMock()
    runtime.read_file = AsyncMock(return_value=b"")

    session.executor = MagicMock()
    session.executor.get_runtime.return_value = runtime
    return session


def _make_manager(session: MagicMock | None = None) -> MagicMock:
    """Create a mock ExecutionSandboxManager."""
    if session is None:
        session = _make_session()
    manager = MagicMock()
    manager.get_or_create_sandbox = AsyncMock(return_value=session)
    return manager


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestCodeActRegistry:
    """Tests for CodeAct tool registration and contracts."""

    def test_codeact_category_exists(self):
        assert ToolCategory.CODEACT == "codeact"

    def test_execute_script_in_catalog(self):
        tools = TOOL_CATALOG.get(ToolCategory.CODEACT, [])
        tool_names = [t.name for t in tools]
        assert "execute_script" in tool_names

    def test_execute_script_has_contract(self):
        assert "execute_script" in TOOL_CONTRACTS

    def test_codeact_not_in_default_task_tools(self):
        """CodeAct tools should NOT be included by default (auto mode)."""
        tools = get_tools_for_agent("task", include_handoffs=False)
        tool_names = [t.name for t in tools]
        assert "execute_script" not in tool_names

    def test_codeact_included_when_mode_is_codeact(self):
        """CodeAct tools should be included when execution_mode='codeact'."""
        tools = get_tools_for_agent(
            "task", include_handoffs=False, execution_mode="codeact"
        )
        tool_names = [t.name for t in tools]
        assert "execute_script" in tool_names

    def test_codeact_not_included_for_research_agent(self):
        """CodeAct tools should NOT be included for research agent even with mode."""
        tools = get_tools_for_agent(
            "research", include_handoffs=False, execution_mode="codeact"
        )
        tool_names = [t.name for t in tools]
        assert "execute_script" not in tool_names


# ---------------------------------------------------------------------------
# _ensure_hyperagent_lib
# ---------------------------------------------------------------------------


class TestEnsureHyperagentLib:
    """Tests for hyperagent helper library installation."""

    def setup_method(self):
        """Clear installed cache between tests."""
        _HYPERAGENT_LIB_INSTALLED.clear()

    @pytest.mark.asyncio
    async def test_installs_library_on_first_call(self):
        session = _make_session()
        runtime = session.executor.get_runtime()

        await _ensure_hyperagent_lib(session)

        # Should have written __init__.py and setup.py
        assert runtime.write_file.call_count >= 2
        # Should have run pip install
        pip_calls = [
            c for c in runtime.run_command.call_args_list
            if "pip install" in str(c)
        ]
        assert len(pip_calls) >= 1
        # Sandbox should be marked as installed
        assert session.sandbox_id in _HYPERAGENT_LIB_INSTALLED

    @pytest.mark.asyncio
    async def test_skips_on_subsequent_calls(self):
        session = _make_session()
        _HYPERAGENT_LIB_INSTALLED.add(session.sandbox_id)

        await _ensure_hyperagent_lib(session)

        runtime = session.executor.get_runtime()
        runtime.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_install_failure_gracefully(self):
        session = _make_session()
        runtime = session.executor.get_runtime()

        # Make pip install fail
        async def _run_cmd(cmd, timeout=60, cwd=None):
            if "pip install" in cmd:
                return SimpleNamespace(exit_code=1, stdout="", stderr="install error")
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        runtime.run_command = AsyncMock(side_effect=_run_cmd)

        await _ensure_hyperagent_lib(session)

        # Should NOT be marked as installed
        assert session.sandbox_id not in _HYPERAGENT_LIB_INSTALLED


# ---------------------------------------------------------------------------
# execute_script
# ---------------------------------------------------------------------------


class TestExecuteScript:
    """Tests for the execute_script tool."""

    def setup_method(self):
        _HYPERAGENT_LIB_INSTALLED.clear()

    @pytest.mark.asyncio
    async def test_executes_python_script(self):
        session = _make_session()
        runtime = session.executor.get_runtime()

        # Pre-mark as installed to skip lib setup
        _HYPERAGENT_LIB_INSTALLED.add(session.sandbox_id)

        # Mock run_command to return script output
        async def _run_cmd(cmd, timeout=60, cwd=None):
            if "python" in cmd and "current_script" in cmd:
                return SimpleNamespace(exit_code=0, stdout="hello world\n", stderr="")
            if "find" in cmd:
                return SimpleNamespace(exit_code=0, stdout="", stderr="")
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        runtime.run_command = AsyncMock(side_effect=_run_cmd)

        manager = _make_manager(session)

        with patch("app.agents.tools.codeact.get_execution_sandbox_manager", return_value=manager), \
             patch("app.sandbox.is_execution_sandbox_available", return_value=True):
            result_json = await execute_script.ainvoke({
                "code": "print('hello world')",
                "timeout": 30,
            })

        result = json.loads(result_json)
        assert result["success"] is True
        assert "hello world" in result["stdout"]
        assert result["sandbox_id"] == session.sandbox_id

    @pytest.mark.asyncio
    async def test_returns_error_when_sandbox_unavailable(self):
        with patch("app.sandbox.is_execution_sandbox_available", return_value=False):
            result_json = await execute_script.ainvoke({
                "code": "print('test')",
            })

        result = json.loads(result_json)
        assert result["success"] is False
        assert "not available" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_detects_created_files(self):
        session = _make_session()
        runtime = session.executor.get_runtime()
        _HYPERAGENT_LIB_INSTALLED.add(session.sandbox_id)

        call_count = 0

        async def _run_cmd(cmd, timeout=60, cwd=None):
            nonlocal call_count
            if "python" in cmd and "current_script" in cmd:
                return SimpleNamespace(exit_code=0, stdout="done", stderr="")
            if "find" in cmd:
                call_count += 1
                if call_count <= 1:
                    return SimpleNamespace(exit_code=0, stdout="", stderr="")
                else:
                    return SimpleNamespace(
                        exit_code=0,
                        stdout="/home/user/output.csv\n",
                        stderr="",
                    )
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        runtime.run_command = AsyncMock(side_effect=_run_cmd)
        manager = _make_manager(session)

        with patch("app.agents.tools.codeact.get_execution_sandbox_manager", return_value=manager), \
             patch("app.sandbox.is_execution_sandbox_available", return_value=True):
            result_json = await execute_script.ainvoke({
                "code": "open('output.csv', 'w').write('data')",
            })

        result = json.loads(result_json)
        assert result["success"] is True
        assert "/home/user/output.csv" in result["created_files"]

    @pytest.mark.asyncio
    async def test_handles_script_error(self):
        session = _make_session()
        runtime = session.executor.get_runtime()
        _HYPERAGENT_LIB_INSTALLED.add(session.sandbox_id)

        async def _run_cmd(cmd, timeout=60, cwd=None):
            if "python" in cmd and "current_script" in cmd:
                return SimpleNamespace(
                    exit_code=1,
                    stdout="",
                    stderr="NameError: name 'x' is not defined",
                )
            if "find" in cmd:
                return SimpleNamespace(exit_code=0, stdout="", stderr="")
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        runtime.run_command = AsyncMock(side_effect=_run_cmd)
        manager = _make_manager(session)

        with patch("app.agents.tools.codeact.get_execution_sandbox_manager", return_value=manager), \
             patch("app.sandbox.is_execution_sandbox_available", return_value=True):
            result_json = await execute_script.ainvoke({
                "code": "print(x)",
            })

        result = json.loads(result_json)
        assert result["success"] is False
        assert result["exit_code"] == 1
        assert "NameError" in result["stderr"]

    @pytest.mark.asyncio
    async def test_handles_sandbox_exception(self):
        manager = MagicMock()
        manager.get_or_create_sandbox = AsyncMock(side_effect=Exception("sandbox crash"))

        with patch("app.agents.tools.codeact.get_execution_sandbox_manager", return_value=manager), \
             patch("app.sandbox.is_execution_sandbox_available", return_value=True):
            result_json = await execute_script.ainvoke({
                "code": "print('test')",
            })

        result = json.loads(result_json)
        assert result["success"] is False
        assert "sandbox crash" in result["error"]


# ---------------------------------------------------------------------------
# hyperagent helper library
# ---------------------------------------------------------------------------


class TestHyperagentLib:
    """Tests for the hyperagent helper library module."""

    def test_read_file(self, tmp_path):
        from app.sandbox.hyperagent_lib import read_file, write_file

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        content = read_file(str(test_file))
        assert content == "hello world"

    def test_write_file(self, tmp_path):
        from app.sandbox.hyperagent_lib import write_file

        path = str(tmp_path / "subdir" / "output.txt")
        result = write_file(path, "test content")
        assert "Written" in result
        assert "test content" == open(path).read()

    def test_run_command(self):
        from app.sandbox.hyperagent_lib import run_command

        result = run_command("echo 'test'")
        assert result["exit_code"] == 0
        assert "test" in result["stdout"]

    def test_run_command_timeout(self):
        from app.sandbox.hyperagent_lib import run_command

        result = run_command("sleep 10", timeout=1)
        assert result["exit_code"] == -1
        assert "timed out" in result["stderr"].lower()

    def test_list_files(self, tmp_path):
        from app.sandbox.hyperagent_lib import list_files

        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        files = list_files(str(tmp_path))
        assert "a.txt" in files
        assert "b.txt" in files

    def test_web_search_returns_pending(self, tmp_path):
        from app.sandbox.hyperagent_lib import web_search

        # Without cache, should return a pending response
        with patch("app.sandbox.hyperagent_lib.Path") as mock_path:
            cache_dir = MagicMock()
            cache_dir.exists.return_value = False
            mock_path.return_value = cache_dir

            # Need to handle both Path calls differently
            results = web_search("test query")

        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Execution mode schema
# ---------------------------------------------------------------------------


class TestExecutionModeSchema:
    """Test that codeact is a valid execution_mode."""

    def test_codeact_is_valid_execution_mode(self):
        from app.models.schemas import UnifiedQueryRequest

        # Should not raise
        req = UnifiedQueryRequest(
            message="test",
            execution_mode="codeact",
        )
        assert req.execution_mode == "codeact"
