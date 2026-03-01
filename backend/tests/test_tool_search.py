"""Tests for tool search / lazy loading system."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.agents.tools.tool_search import (
    SearchToolsInput,
    ToolMetadata,
    ToolMetadataRegistry,
    get_tool_metadata_registry,
    populate_registry_from_catalog,
    search_tools,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    """Create a fresh ToolMetadataRegistry for testing."""
    return ToolMetadataRegistry()


@pytest.fixture
def populated_registry(registry):
    """Create a registry populated with sample tools."""
    registry.register("file_read", "Read file contents from the sandbox filesystem", "file_ops")
    registry.register("file_write", "Write content to a file in the sandbox", "file_ops")
    registry.register("file_str_replace", "Replace a string in a file with a new string", "file_ops")
    registry.register("file_find_by_name", "Find files by name or glob pattern", "file_ops")
    registry.register("file_find_in_content", "Search for text patterns across file contents", "file_ops")
    registry.register("browser_navigate", "Navigate to a URL in the browser sandbox", "browser")
    registry.register("browser_click", "Click at specific coordinates on the page", "browser")
    registry.register("browser_screenshot", "Capture a screenshot of the current page", "browser")
    registry.register("browser_type", "Type text at the current cursor position", "browser")
    registry.register("shell_exec", "Execute a shell command in the sandbox", "shell")
    registry.register("shell_view", "View recent output from a background shell session", "shell")
    registry.register("shell_wait", "Wait for a background process to complete", "shell")
    registry.register("shell_kill", "Terminate a running background process", "shell")
    registry.register("web_search", "Search the web for information on a topic", "search")
    registry.register("execute_code", "Execute code in an isolated sandbox environment", "code_exec")
    registry.register("generate_image", "Generate an image from a text description", "image")
    registry.register("analyze_image", "Analyze and extract information from an image", "image")
    return registry


# ---------------------------------------------------------------------------
# ToolMetadataRegistry.register tests
# ---------------------------------------------------------------------------


class TestToolMetadataRegistration:
    """Tests for registering tools in the metadata registry."""

    def test_register_single_tool(self, registry):
        """Registering a tool stores its metadata."""
        registry.register("my_tool", "A description of my tool", "general")

        assert "my_tool" in registry._tools
        meta = registry._tools["my_tool"]
        assert meta.name == "my_tool"
        assert meta.description == "A description of my tool"
        assert meta.category == "general"

    def test_register_truncates_long_description(self, registry):
        """Descriptions longer than 200 chars are truncated."""
        long_desc = "x" * 300
        registry.register("long_tool", long_desc, "general")

        meta = registry._tools["long_tool"]
        assert len(meta.description) == 200

    def test_register_overwrites_existing(self, registry):
        """Registering a tool with the same name overwrites the old entry."""
        registry.register("my_tool", "Version 1", "cat_a")
        registry.register("my_tool", "Version 2", "cat_b")

        assert registry._tools["my_tool"].description == "Version 2"
        assert registry._tools["my_tool"].category == "cat_b"

    def test_get_all_returns_all_registered(self, registry):
        """get_all returns every registered tool."""
        registry.register("tool_a", "Desc A", "cat")
        registry.register("tool_b", "Desc B", "cat")
        registry.register("tool_c", "Desc C", "cat")

        all_tools = registry.get_all()
        assert len(all_tools) == 3
        names = {t.name for t in all_tools}
        assert names == {"tool_a", "tool_b", "tool_c"}


# ---------------------------------------------------------------------------
# ToolMetadataRegistry.search tests
# ---------------------------------------------------------------------------


class TestToolMetadataSearch:
    """Tests for searching the metadata registry."""

    def test_search_returns_relevant_file_tools(self, populated_registry):
        """Searching for 'file' returns file-related tools."""
        results = populated_registry.search("file")
        names = {r.name for r in results}

        assert "file_read" in names
        assert "file_write" in names
        assert "file_str_replace" in names

    def test_search_returns_relevant_browser_tools(self, populated_registry):
        """Searching for 'browser' returns browser-related tools."""
        results = populated_registry.search("browser")
        names = {r.name for r in results}

        assert "browser_navigate" in names
        assert "browser_click" in names
        assert "browser_screenshot" in names

    def test_search_returns_relevant_shell_tools(self, populated_registry):
        """Searching for 'shell' returns shell-related tools."""
        results = populated_registry.search("shell")
        names = {r.name for r in results}

        assert "shell_exec" in names
        assert "shell_view" in names
        assert "shell_kill" in names

    def test_search_empty_query(self, populated_registry):
        """Searching with an empty string returns results based on similarity."""
        results = populated_registry.search("")
        # Empty query may still match some tools via sequence similarity
        # but the important thing is it doesn't crash
        assert isinstance(results, list)

    def test_search_no_matches(self, registry):
        """Searching an empty registry returns no results."""
        results = registry.search("anything")
        assert results == []

    def test_search_nonsense_query(self, populated_registry):
        """Searching for a completely unrelated term returns few or no results."""
        results = populated_registry.search("xyzzy_nonsense_12345")
        # May return some low-scoring results, but nothing relevant
        for r in results:
            assert isinstance(r, ToolMetadata)

    def test_search_respects_limit(self, populated_registry):
        """Search results are limited by the limit parameter."""
        results = populated_registry.search("file", limit=2)
        assert len(results) <= 2

    def test_exact_name_match_scores_highest(self, populated_registry):
        """An exact name match should be the top result."""
        results = populated_registry.search("file_read")
        assert len(results) > 0
        assert results[0].name == "file_read"

    def test_search_by_description_keyword(self, populated_registry):
        """Searching by a description keyword finds relevant tools."""
        results = populated_registry.search("screenshot")
        names = {r.name for r in results}
        assert "browser_screenshot" in names

    def test_search_by_category(self, populated_registry):
        """Searching by category name finds tools in that category."""
        results = populated_registry.search("image")
        names = {r.name for r in results}
        assert "generate_image" in names or "analyze_image" in names

    def test_search_returns_metadata_objects(self, populated_registry):
        """Search results are ToolMetadata instances with all fields."""
        results = populated_registry.search("shell")
        assert len(results) > 0

        for r in results:
            assert isinstance(r, ToolMetadata)
            assert isinstance(r.name, str)
            assert isinstance(r.description, str)
            assert isinstance(r.category, str)
            assert len(r.name) > 0


# ---------------------------------------------------------------------------
# search_tools tool tests
# ---------------------------------------------------------------------------


class TestSearchToolsTool:
    """Tests for the search_tools LangChain tool."""

    def test_search_tools_returns_valid_json(self, populated_registry):
        """search_tools returns valid JSON output."""
        with patch(
            "app.agents.tools.tool_search._metadata_registry",
            populated_registry,
        ):
            result = search_tools.invoke({"query": "file"})

        parsed = json.loads(result)
        assert "matches" in parsed
        assert "message" in parsed
        assert isinstance(parsed["matches"], list)

    def test_search_tools_matches_have_expected_fields(self, populated_registry):
        """Each match in the result has name, description, and category."""
        with patch(
            "app.agents.tools.tool_search._metadata_registry",
            populated_registry,
        ):
            result = search_tools.invoke({"query": "browser"})

        parsed = json.loads(result)
        for match in parsed["matches"]:
            assert "name" in match
            assert "description" in match
            assert "category" in match

    def test_search_tools_no_results(self):
        """search_tools with empty registry returns empty matches."""
        empty_registry = ToolMetadataRegistry()
        with patch(
            "app.agents.tools.tool_search._metadata_registry",
            empty_registry,
        ):
            result = search_tools.invoke({"query": "anything"})

        parsed = json.loads(result)
        assert parsed["matches"] == []
        assert "No tools found" in parsed["message"]

    def test_search_tools_limit(self, populated_registry):
        """search_tools returns at most 8 results."""
        with patch(
            "app.agents.tools.tool_search._metadata_registry",
            populated_registry,
        ):
            result = search_tools.invoke({"query": "file"})

        parsed = json.loads(result)
        assert len(parsed["matches"]) <= 8

    def test_search_tools_message_includes_count(self, populated_registry):
        """The message in the result includes the count of matches found."""
        with patch(
            "app.agents.tools.tool_search._metadata_registry",
            populated_registry,
        ):
            result = search_tools.invoke({"query": "shell"})

        parsed = json.loads(result)
        if parsed["matches"]:
            count = len(parsed["matches"])
            assert f"Found {count}" in parsed["message"]


# ---------------------------------------------------------------------------
# populate_registry_from_catalog tests
# ---------------------------------------------------------------------------


class TestPopulateRegistryFromCatalog:
    """Tests for populating the registry from the tool catalog."""

    def test_populate_from_catalog(self):
        """populate_registry_from_catalog loads tools from TOOL_CATALOG."""
        # Create mock tools
        mock_tool_a = MagicMock()
        mock_tool_a.name = "tool_a"
        mock_tool_a.description = "Tool A description"

        mock_tool_b = MagicMock()
        mock_tool_b.name = "tool_b"
        mock_tool_b.description = "Tool B description"

        # Create a mock ToolCategory enum value
        mock_cat = MagicMock()
        mock_cat.value = "test_cat"

        mock_catalog = {mock_cat: [mock_tool_a, mock_tool_b]}

        fresh_registry = ToolMetadataRegistry()

        with (
            patch("app.agents.tools.registry.TOOL_CATALOG", mock_catalog),
            patch("app.agents.tools.tool_search._metadata_registry", fresh_registry),
        ):
            populate_registry_from_catalog()

        assert "tool_a" in fresh_registry._tools
        assert "tool_b" in fresh_registry._tools
        assert fresh_registry._tools["tool_a"].category == "test_cat"
        assert fresh_registry._tools["tool_b"].description == "Tool B description"

    def test_populate_handles_none_description(self):
        """populate_registry_from_catalog handles tools with None description."""
        mock_tool = MagicMock()
        mock_tool.name = "no_desc_tool"
        mock_tool.description = None

        mock_cat = MagicMock()
        mock_cat.value = "test_cat"

        mock_catalog = {mock_cat: [mock_tool]}

        fresh_registry = ToolMetadataRegistry()

        with (
            patch("app.agents.tools.registry.TOOL_CATALOG", mock_catalog),
            patch("app.agents.tools.tool_search._metadata_registry", fresh_registry),
        ):
            populate_registry_from_catalog()

        assert "no_desc_tool" in fresh_registry._tools
        assert fresh_registry._tools["no_desc_tool"].description == ""


# ---------------------------------------------------------------------------
# Registry integration tests
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """Tests that tool_search is properly registered in the tool registry."""

    def test_tool_search_category_exists(self):
        """TOOL_SEARCH category should exist in ToolCategory."""
        from app.agents.tools.registry import ToolCategory

        assert hasattr(ToolCategory, "TOOL_SEARCH")
        assert ToolCategory.TOOL_SEARCH.value == "tool_search"

    def test_search_tools_in_catalog(self):
        """search_tools should be in the TOOL_CATALOG."""
        from app.agents.tools.registry import TOOL_CATALOG, ToolCategory

        tool_search_tools = TOOL_CATALOG.get(ToolCategory.TOOL_SEARCH, [])
        tool_names = {t.name for t in tool_search_tools}
        assert "search_tools" in tool_names

    def test_tool_search_in_task_agent_mapping(self):
        """TOOL_SEARCH should be in the TASK agent's tool mapping."""
        from app.agents.state import AgentType
        from app.agents.tools.registry import AGENT_TOOL_MAPPING, ToolCategory

        task_categories = AGENT_TOOL_MAPPING[AgentType.TASK.value]
        assert ToolCategory.TOOL_SEARCH in task_categories
