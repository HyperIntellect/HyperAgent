"""Tests for browser enhancement tools (console_exec, select_option, wait_for_element)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.tools.browser_use import (
    browser_console_exec,
    browser_dom_query,
    browser_select_option,
    browser_wait_for_element,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_executor():
    """Create a mock desktop executor."""
    executor = AsyncMock()
    executor.sandbox_id = "test-desktop-123"
    executor.browser_launched = True
    executor.run_command = AsyncMock()
    return executor


@pytest.fixture
def mock_session(mock_executor):
    """Create a mock DesktopSandboxSession."""
    session = MagicMock()
    session.executor = mock_executor
    session.sandbox_id = "test-desktop-123"
    session.browser_launched = True
    return session


@pytest.fixture
def mock_desktop_manager(mock_session):
    """Create a mock DesktopSandboxManager."""
    manager = AsyncMock()
    manager.get_session.return_value = mock_session
    return manager


def _patch_browser(mock_desktop_manager):
    """Return patch context managers for browser tools."""
    return (
        patch(
            "app.sandbox.get_desktop_sandbox_manager",
            return_value=mock_desktop_manager,
        ),
        patch(
            "app.agents.tools.browser_use.is_desktop_sandbox_available",
            return_value=True,
        ),
    )


# ---------------------------------------------------------------------------
# browser_console_exec tests
# ---------------------------------------------------------------------------


class TestBrowserConsoleExec:
    """Tests for browser_console_exec tool."""

    @pytest.mark.asyncio
    async def test_exec_js_success(self, mock_desktop_manager, mock_executor):
        """Successfully execute JavaScript and return result."""
        mock_executor.run_command.return_value = (
            json.dumps({"value": "Hello World", "type": "string"}),
            "",
            0,
        )
        p1, p2 = _patch_browser(mock_desktop_manager)

        with p1, p2:
            result = json.loads(
                await browser_console_exec.ainvoke({"javascript": "document.title"})
            )

        assert result["success"] is True
        assert result["result"] == "Hello World"
        assert result["result_type"] == "string"
        assert result["sandbox_id"] == "test-desktop-123"

    @pytest.mark.asyncio
    async def test_exec_js_error(self, mock_desktop_manager, mock_executor):
        """JavaScript execution error is reported."""
        mock_executor.run_command.return_value = (
            json.dumps({"error": "ReferenceError: foo is not defined"}),
            "",
            0,
        )
        p1, p2 = _patch_browser(mock_desktop_manager)

        with p1, p2:
            result = json.loads(
                await browser_console_exec.ainvoke({"javascript": "foo.bar"})
            )

        assert result["success"] is False
        assert "ReferenceError" in result["error"]

    @pytest.mark.asyncio
    async def test_exec_js_command_failure(self, mock_desktop_manager, mock_executor):
        """Command execution failure returns error."""
        mock_executor.run_command.return_value = ("", "command not found", 127)
        p1, p2 = _patch_browser(mock_desktop_manager)

        with p1, p2:
            result = json.loads(
                await browser_console_exec.ainvoke({"javascript": "1+1"})
            )

        assert result["success"] is False
        assert "failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_exec_js_no_session(self, mock_desktop_manager):
        """Returns error when no browser session exists."""
        mock_desktop_manager.get_session.return_value = None
        p1, p2 = _patch_browser(mock_desktop_manager)

        with p1, p2:
            result = json.loads(
                await browser_console_exec.ainvoke({"javascript": "1+1"})
            )

        assert result["success"] is False
        assert "No active browser sandbox session" in result["error"]

    @pytest.mark.asyncio
    async def test_exec_js_sandbox_unavailable(self):
        """Returns error when desktop sandbox is not available."""
        with patch(
            "app.agents.tools.browser_use.is_desktop_sandbox_available",
            return_value=False,
        ):
            result = json.loads(
                await browser_console_exec.ainvoke({"javascript": "1+1"})
            )

        assert result["success"] is False
        assert "not available" in result["error"].lower()


# ---------------------------------------------------------------------------
# browser_select_option tests
# ---------------------------------------------------------------------------


class TestBrowserSelectOption:
    """Tests for browser_select_option tool."""

    @pytest.mark.asyncio
    async def test_select_option_success(self, mock_desktop_manager, mock_executor):
        """Successfully select an option from a dropdown."""
        mock_executor.run_command.return_value = (
            json.dumps({
                "value": json.dumps({"selected": "us", "text": "United States"}),
                "type": "string",
            }),
            "",
            0,
        )
        p1, p2 = _patch_browser(mock_desktop_manager)

        with p1, p2:
            result = json.loads(
                await browser_select_option.ainvoke({
                    "selector": "#country",
                    "value": "us",
                })
            )

        assert result["success"] is True
        assert result["value"] == "us"
        assert result["text"] == "United States"

    @pytest.mark.asyncio
    async def test_select_option_element_not_found(self, mock_desktop_manager, mock_executor):
        """Returns error when element is not found."""
        mock_executor.run_command.return_value = (
            json.dumps({
                "value": json.dumps({"error": "Element not found: #missing"}),
                "type": "string",
            }),
            "",
            0,
        )
        p1, p2 = _patch_browser(mock_desktop_manager)

        with p1, p2:
            result = json.loads(
                await browser_select_option.ainvoke({
                    "selector": "#missing",
                    "value": "test",
                })
            )

        assert result["success"] is False
        assert "Element not found" in result["error"]

    @pytest.mark.asyncio
    async def test_select_option_no_session(self, mock_desktop_manager):
        """Returns error when no browser session exists."""
        mock_desktop_manager.get_session.return_value = None
        p1, p2 = _patch_browser(mock_desktop_manager)

        with p1, p2:
            result = json.loads(
                await browser_select_option.ainvoke({
                    "selector": "#country",
                    "value": "us",
                })
            )

        assert result["success"] is False
        assert "No active browser sandbox session" in result["error"]

    @pytest.mark.asyncio
    async def test_select_option_sandbox_unavailable(self):
        """Returns error when desktop sandbox is not available."""
        with patch(
            "app.agents.tools.browser_use.is_desktop_sandbox_available",
            return_value=False,
        ):
            result = json.loads(
                await browser_select_option.ainvoke({
                    "selector": "#country",
                    "value": "us",
                })
            )

        assert result["success"] is False
        assert "not available" in result["error"].lower()


# ---------------------------------------------------------------------------
# browser_wait_for_element tests
# ---------------------------------------------------------------------------


class TestBrowserWaitForElement:
    """Tests for browser_wait_for_element tool."""

    @pytest.mark.asyncio
    async def test_wait_element_found_immediately(self, mock_desktop_manager, mock_executor):
        """Element is found on the first poll."""
        mock_executor.run_command.return_value = (
            json.dumps({"value": True, "type": "boolean"}),
            "",
            0,
        )
        p1, p2 = _patch_browser(mock_desktop_manager)

        with p1, p2:
            result = json.loads(
                await browser_wait_for_element.ainvoke({
                    "selector": "#result",
                    "timeout": 5,
                })
            )

        assert result["success"] is True
        assert result["found"] is True
        assert result["elapsed_seconds"] == 0.0

    @pytest.mark.asyncio
    async def test_wait_element_found_after_polls(self, mock_desktop_manager, mock_executor):
        """Element is found after a few polling cycles."""
        # First two polls return false, third returns true
        mock_executor.run_command.side_effect = [
            (json.dumps({"value": False, "type": "boolean"}), "", 0),
            (json.dumps({"value": False, "type": "boolean"}), "", 0),
            (json.dumps({"value": True, "type": "boolean"}), "", 0),
        ]
        p1, p2 = _patch_browser(mock_desktop_manager)

        with p1, p2:
            with patch("app.agents.tools.browser_use.asyncio.sleep", new_callable=AsyncMock):
                result = json.loads(
                    await browser_wait_for_element.ainvoke({
                        "selector": ".loaded",
                        "timeout": 10,
                    })
                )

        assert result["success"] is True
        assert result["found"] is True

    @pytest.mark.asyncio
    async def test_wait_element_timeout(self, mock_desktop_manager, mock_executor):
        """Returns timeout error when element is not found within timeout."""
        mock_executor.run_command.return_value = (
            json.dumps({"value": False, "type": "boolean"}),
            "",
            0,
        )
        p1, p2 = _patch_browser(mock_desktop_manager)

        with p1, p2:
            with patch("app.agents.tools.browser_use.asyncio.sleep", new_callable=AsyncMock):
                result = json.loads(
                    await browser_wait_for_element.ainvoke({
                        "selector": "#never-appears",
                        "timeout": 1,
                    })
                )

        assert result["success"] is False
        assert "Timeout" in result["error"]
        assert "#never-appears" in result["error"]

    @pytest.mark.asyncio
    async def test_wait_element_no_session(self, mock_desktop_manager):
        """Returns error when no browser session exists."""
        mock_desktop_manager.get_session.return_value = None
        p1, p2 = _patch_browser(mock_desktop_manager)

        with p1, p2:
            result = json.loads(
                await browser_wait_for_element.ainvoke({
                    "selector": "#result",
                })
            )

        assert result["success"] is False
        assert "No active browser sandbox session" in result["error"]

    @pytest.mark.asyncio
    async def test_wait_element_sandbox_unavailable(self):
        """Returns error when desktop sandbox is not available."""
        with patch(
            "app.agents.tools.browser_use.is_desktop_sandbox_available",
            return_value=False,
        ):
            result = json.loads(
                await browser_wait_for_element.ainvoke({
                    "selector": "#result",
                })
            )

        assert result["success"] is False
        assert "not available" in result["error"].lower()


# ---------------------------------------------------------------------------
# browser_dom_query tests
# ---------------------------------------------------------------------------


class TestBrowserDomQuery:
    """Tests for browser_dom_query tool."""

    @pytest.mark.asyncio
    async def test_dom_query_success(self, mock_desktop_manager, mock_executor):
        """Successfully query matching elements."""
        mock_executor.run_command.return_value = (
            json.dumps({
                "value": json.dumps({
                    "count": 2,
                    "selector": ".item",
                    "items": [{"tag": "a", "text": "First"}, {"tag": "a", "text": "Second"}],
                }),
                "type": "string",
            }),
            "",
            0,
        )
        p1, p2 = _patch_browser(mock_desktop_manager)
        with p1, p2:
            result = json.loads(await browser_dom_query.ainvoke({"selector": ".item"}))

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["items"]) == 2

    @pytest.mark.asyncio
    async def test_dom_query_no_session(self, mock_desktop_manager):
        """Returns error when no browser session exists."""
        mock_desktop_manager.get_session.return_value = None
        p1, p2 = _patch_browser(mock_desktop_manager)
        with p1, p2:
            result = json.loads(await browser_dom_query.ainvoke({"selector": ".item"}))

        assert result["success"] is False
        assert "No active browser sandbox session" in result["error"]

    @pytest.mark.asyncio
    async def test_dom_query_sandbox_unavailable(self):
        """Returns error when desktop sandbox is unavailable."""
        with patch(
            "app.agents.tools.browser_use.is_desktop_sandbox_available",
            return_value=False,
        ):
            result = json.loads(await browser_dom_query.ainvoke({"selector": ".item"}))

        assert result["success"] is False
        assert "not available" in result["error"].lower()


# ---------------------------------------------------------------------------
# Registry integration tests
# ---------------------------------------------------------------------------


class TestBrowserEnhancementsRegistry:
    """Tests that new browser tools are properly registered."""

    def test_browser_category_has_new_tools(self):
        """New browser tools should be in the TOOL_CATALOG."""
        from app.agents.tools.registry import TOOL_CATALOG, ToolCategory

        browser_tools = TOOL_CATALOG.get(ToolCategory.BROWSER, [])
        tool_names = {t.name for t in browser_tools}

        assert "browser_console_exec" in tool_names
        assert "browser_dom_query" in tool_names
        assert "browser_select_option" in tool_names
        assert "browser_wait_for_element" in tool_names

    def test_new_tools_available_to_task_agent(self):
        """New browser tools should be accessible to task agent."""
        from app.agents.tools.registry import get_tool_names_for_agent

        tool_names = get_tool_names_for_agent("task")
        assert "browser_console_exec" in tool_names
        assert "browser_dom_query" in tool_names
        assert "browser_select_option" in tool_names
        assert "browser_wait_for_element" in tool_names

    def test_new_tools_available_to_research_agent(self):
        """New browser tools should be accessible to research agent."""
        from app.agents.tools.registry import get_tool_names_for_agent

        tool_names = get_tool_names_for_agent("research")
        assert "browser_console_exec" in tool_names
        assert "browser_dom_query" in tool_names
        assert "browser_select_option" in tool_names
        assert "browser_wait_for_element" in tool_names
