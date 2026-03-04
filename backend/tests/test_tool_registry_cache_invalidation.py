"""Tests for dynamic tool registry cache invalidation."""

from types import SimpleNamespace
from unittest.mock import patch

from app.agents.tools.registry import ToolCategory, register_tool, unregister_tool


def test_register_tool_invalidates_cached_agent_tools():
    """Registering a new tool should invalidate cached tool lists."""
    tool_name = "test_cache_invalidation_register_tool"
    tool = SimpleNamespace(name=tool_name)

    with patch("app.agents.tools.registry._invalidate_cached_agent_tools") as invalidate:
        register_tool(ToolCategory.MCP, tool)
        assert invalidate.called

    # Cleanup test tool from global registry
    unregister_tool(ToolCategory.MCP, tool_name)


def test_unregister_tool_invalidates_cached_agent_tools():
    """Unregistering a tool should invalidate cached tool lists."""
    tool_name = "test_cache_invalidation_unregister_tool"
    tool = SimpleNamespace(name=tool_name)
    register_tool(ToolCategory.MCP, tool)

    with patch("app.agents.tools.registry._invalidate_cached_agent_tools") as invalidate:
        removed = unregister_tool(ToolCategory.MCP, tool_name)
        assert removed is True
        assert invalidate.called
