"""Tests for web_extract_structured tool and registry integration."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.tools.registry import ToolCategory, get_tools_by_category
from app.agents.tools.web_search import web_extract_structured
from app.services.search import SearchResult


class TestWebExtractStructured:
    """Validation tests for structured web extraction."""

    @pytest.mark.asyncio
    async def test_returns_structured_items(self):
        mock_results = [
            SearchResult(
                title="OpenAI Agents Guide",
                url="https://platform.openai.com/docs/guides/agents",
                snippet="Published 2026-01-10 by OpenAI",
                content="This guide explains agent workflows and tool usage.",
                relevance_score=0.98,
            )
        ]

        with patch(
            "app.agents.tools.web_search.search_service.search_raw",
            new=AsyncMock(return_value=mock_results),
        ):
            output = await web_extract_structured.ainvoke({
                "query": "OpenAI agent guides",
                "fields": ["title", "url", "published_date", "author", "summary"],
                "max_results": 1,
            })

        data = json.loads(output)
        assert data["success"] is True
        assert data["count"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["title"] == "OpenAI Agents Guide"
        assert data["items"][0]["url"] == "https://platform.openai.com/docs/guides/agents"
        assert data["items"][0]["published_date"] == "2026-01-10"
        assert data["items"][0]["author"] == "OpenAI"
        assert data["items"][0]["summary"]

    @pytest.mark.asyncio
    async def test_handles_empty_results(self):
        with patch(
            "app.agents.tools.web_search.search_service.search_raw",
            new=AsyncMock(return_value=[]),
        ):
            output = await web_extract_structured.ainvoke({
                "query": "no results query",
                "fields": ["title", "url"],
            })

        data = json.loads(output)
        assert data["success"] is True
        assert data["count"] == 0
        assert data["items"] == []

    @pytest.mark.asyncio
    async def test_supports_url_first_mode(self):
        page = SearchResult(
            title="Direct Page",
            url="https://example.com/post",
            snippet="Direct fetch snippet",
            content="By Example Author Published 2026-02-20. Full page body.",
            relevance_score=None,
        )
        with patch(
            "app.agents.tools.web_search.asyncio.to_thread",
            new=AsyncMock(return_value=page),
        ):
            output = await web_extract_structured.ainvoke({
                "url": "https://example.com/post",
                "fields": ["title", "url", "author", "published_date"],
            })
        data = json.loads(output)
        assert data["success"] is True
        assert data["url"] == "https://example.com/post"
        assert data["count"] == 1
        assert data["items"][0]["title"] == "Direct Page"
        assert data["items"][0]["author"] == "Example Author"
        assert data["items"][0]["published_date"] == "2026-02-20"

    @pytest.mark.asyncio
    async def test_requires_query_or_url(self):
        output = await web_extract_structured.ainvoke({
            "fields": ["title", "url"],
        })
        data = json.loads(output)
        assert data["success"] is False
        assert "either query or url" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_surfaces_search_error(self):
        with patch(
            "app.agents.tools.web_search.search_service.search_raw",
            new=AsyncMock(side_effect=ValueError("Search API key missing")),
        ):
            output = await web_extract_structured.ainvoke({
                "query": "error case",
                "fields": ["title"],
            })

        data = json.loads(output)
        assert data["success"] is False
        assert "missing" in data["error"].lower()


def test_registered_in_search_catalog():
    """Tool should be available to SEARCH category agents."""
    search_tools = get_tools_by_category(ToolCategory.SEARCH)
    search_tool_names = {tool.name for tool in search_tools}
    assert "web_extract_structured" in search_tool_names
