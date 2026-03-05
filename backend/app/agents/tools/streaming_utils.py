"""Streaming utilities for real-time terminal event dispatch.

Provides helpers to emit terminal events (command, output, complete)
via LangGraph's dispatch_custom_event during tool execution, enabling
line-by-line terminal output streaming to the frontend.
"""

from typing import Any, Callable

from app.agents import events
from app.core.logging import get_logger

logger = get_logger(__name__)

OutputCallback = Callable[[str], None]


def safe_dispatch_event(event: dict[str, Any]) -> None:
    """Dispatch a skill_event via LangGraph, silently dropping if outside context.

    Args:
        event: Event dictionary to dispatch (must have a "type" key).
    """
    try:
        from langchain_core.callbacks import dispatch_custom_event

        dispatch_custom_event("skill_event", data=event)
    except Exception:
        # Outside LangGraph callback context — silently ignore.
        pass


def emit_terminal_command(command: str, cwd: str = "/home/user") -> None:
    """Emit a terminal_command event for a shell command about to run."""
    safe_dispatch_event(events.terminal_command(command=command, cwd=cwd))


def emit_terminal_output(content: str, stream: str = "stdout") -> None:
    """Emit a terminal_output event for a single line/chunk of output."""
    safe_dispatch_event(events.terminal_output(content=content, stream=stream))


def emit_terminal_error(content: str, exit_code: int | None = None) -> None:
    """Emit a terminal_error event."""
    safe_dispatch_event(events.terminal_error(content=content, exit_code=exit_code))


def emit_terminal_complete(exit_code: int) -> None:
    """Emit a terminal_complete event after command finishes."""
    safe_dispatch_event(events.terminal_complete(exit_code=exit_code))


def create_stdout_callback() -> OutputCallback:
    """Return a callback that emits terminal_output(stream='stdout') per line."""

    def _on_stdout(line: str) -> None:
        if line:
            emit_terminal_output(content=line, stream="stdout")

    return _on_stdout


def create_stderr_callback() -> OutputCallback:
    """Return a callback that emits terminal_output(stream='stderr') per line."""

    def _on_stderr(line: str) -> None:
        if line:
            emit_terminal_output(content=line, stream="stderr")

    return _on_stderr
