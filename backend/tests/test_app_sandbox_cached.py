"""Tests for pre-cached template scaffold in AppSandboxManager."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.sandbox.app_sandbox_manager import (
    TEMPLATE_CACHE_DIR,
    TEMPLATE_CACHE_MARKER,
    AppSandboxManager,
    AppSandboxSession,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _FakeResult:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class _FakeSandbox:
    """Minimal SandboxRuntime stub for testing."""

    sandbox_id: str = "test-sandbox"
    _commands: dict = field(default_factory=dict)

    async def run_command(self, cmd: str, timeout: int = 30, cwd: str | None = None) -> _FakeResult:
        if cmd in self._commands:
            return self._commands[cmd]
        # Default: success
        return _FakeResult(exit_code=0)


def _make_session(sandbox: _FakeSandbox | None = None) -> AppSandboxSession:
    sb = sandbox or _FakeSandbox()
    return AppSandboxSession(
        sandbox=sb,
        session_key="test-key",
        template="react-ts",
        created_at=_utcnow(),
        last_accessed=_utcnow(),
        last_health_check=time.monotonic(),
    )


@pytest.fixture
def manager():
    return AppSandboxManager()


# ---------- _try_cached_scaffold ----------


@pytest.mark.asyncio
async def test_cached_scaffold_hit(manager):
    """When marker and template dir exist, copies from cache and returns result."""
    sandbox = _FakeSandbox()
    session = _make_session(sandbox)

    result = await manager._try_cached_scaffold(session, "react-ts")

    assert result is not None
    assert result["success"] is True
    assert result["cached"] is True
    assert result["project_dir"] == "/home/user/app"


@pytest.mark.asyncio
async def test_cached_scaffold_miss_no_marker(manager):
    """When marker file is missing, returns None (cache miss)."""
    check_cmd = f"test -f {TEMPLATE_CACHE_MARKER} && test -d {TEMPLATE_CACHE_DIR}/react-ts"
    sandbox = _FakeSandbox(_commands={check_cmd: _FakeResult(exit_code=1)})
    session = _make_session(sandbox)

    result = await manager._try_cached_scaffold(session, "react-ts")

    assert result is None


@pytest.mark.asyncio
async def test_cached_scaffold_copy_failure(manager):
    """When cp -a fails, returns None so caller falls through to live scaffold."""
    copy_cmd = f"cp -a {TEMPLATE_CACHE_DIR}/react-ts /home/user/app"
    sandbox = _FakeSandbox(
        _commands={copy_cmd: _FakeResult(exit_code=1, stderr="cp: error")}
    )
    session = _make_session(sandbox)

    result = await manager._try_cached_scaffold(session, "react-ts")

    assert result is None


@pytest.mark.asyncio
async def test_cached_scaffold_exception(manager):
    """When sandbox.run_command raises, returns None instead of propagating."""
    sandbox = _FakeSandbox()
    sandbox.run_command = AsyncMock(side_effect=RuntimeError("sandbox down"))
    session = _make_session(sandbox)

    result = await manager._try_cached_scaffold(session, "react-ts")

    assert result is None


# ---------- scaffold_project integration ----------


@pytest.mark.asyncio
async def test_scaffold_project_uses_cache(manager):
    """scaffold_project() should use cached path when available."""
    sandbox = _FakeSandbox()
    session = _make_session(sandbox)

    result = await manager.scaffold_project(session, template="react-ts")

    assert result["success"] is True
    assert result.get("cached") is True
    assert session.project_dir == "/home/user/app"
    assert session.template == "react-ts"


@pytest.mark.asyncio
async def test_scaffold_project_falls_back_on_cache_miss(manager):
    """scaffold_project() falls back to live scaffold when cache is missing."""
    # Cache check fails (no marker)
    check_cmd = f"test -f {TEMPLATE_CACHE_MARKER} && test -d {TEMPLATE_CACHE_DIR}/react"
    sandbox = _FakeSandbox(_commands={check_cmd: _FakeResult(exit_code=1)})
    session = _make_session(sandbox)
    session.template = "react"

    result = await manager.scaffold_project(session, template="react")

    assert result["success"] is True
    # No cached key — went through live scaffold
    assert "cached" not in result
    assert session.project_dir == "/home/user/app"
