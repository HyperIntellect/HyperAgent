"""Web search tool for LangGraph agents."""

import json
import re
from html.parser import HTMLParser
from typing import Literal
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import asyncio

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.services.search import search_service, SearchResult

logger = get_logger(__name__)


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""

    query: str = Field(description="The search query to execute")
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return (1-20)",
    )
    search_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="Search depth: 'basic' for quick results, 'advanced' for comprehensive search",
    )


class WebExtractStructuredInput(BaseModel):
    """Input schema for structured web extraction."""

    query: str | None = Field(
        default=None,
        description="Search query for discovering relevant pages",
    )
    url: str | None = Field(
        default=None,
        description="Optional direct URL for URL-first extraction mode",
    )
    fields: list[str] = Field(
        default_factory=lambda: ["title", "url", "snippet", "summary"],
        description="Structured fields to extract from each result",
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of structured results to return (1-10)",
    )
    search_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="Search depth: 'basic' for quick results, 'advanced' for comprehensive search",
    )


class _TextExtractor(HTMLParser):
    """Minimal HTML to text extractor for URL-first structured extraction."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = (data or "").strip()
        if not text:
            return
        if self._in_title and not self.title:
            self.title = text
        self._chunks.append(text)

    def get_text(self) -> str:
        return " ".join(self._chunks)


def _fetch_page_for_structured_extraction(url: str) -> SearchResult:
    """Fetch and convert a URL into SearchResult-like content."""
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; HyperAgent/1.0; +https://hyperagent.local)"
            )
        },
    )
    with urlopen(req, timeout=20) as resp:  # nosec B310 - URL comes from user tool args by design
        raw = resp.read(1_000_000)
        html = raw.decode("utf-8", errors="ignore")

    parser = _TextExtractor()
    parser.feed(html)
    text = re.sub(r"\s+", " ", parser.get_text()).strip()
    title = parser.title or url
    snippet = text[:280]
    return SearchResult(
        title=title,
        url=url,
        snippet=snippet,
        content=text[:8000],
        relevance_score=None,
    )


@tool(args_schema=WebSearchInput)
async def web_search(
    query: str,
    max_results: int = 5,
    search_depth: Literal["basic", "advanced"] = "basic",
) -> str:
    """Search the web for current information on any topic.

    Use this tool when you need to find up-to-date information, verify facts,
    or gather data from the internet. Returns relevant web pages with titles,
    URLs, and content snippets.

    Args:
        query: The search query to execute
        max_results: Maximum number of results (1-20)
        search_depth: 'basic' for quick results, 'advanced' for comprehensive search

    Returns:
        JSON string with formatted results and structured data
    """
    logger.info(
        "web_search_tool_invoked",
        query=query,
        max_results=max_results,
        search_depth=search_depth,
    )

    try:
        results = await search_service.search_raw(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
        )

        if not results:
            return json.dumps({
                "formatted": f"No results found for: {query}",
                "results": [],
                "query": query,
            })

        return json.dumps({
            "formatted": _format_results(results),
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "content": r.content,
                    "relevance_score": r.relevance_score,
                }
                for r in results
            ],
            "query": query,
        })

    except ValueError as e:
        logger.warning("web_search_api_not_configured", error=str(e))
        return json.dumps({
            "formatted": f"Search service unavailable: {e}",
            "results": [],
            "query": query,
            "error": str(e),
        })
    except Exception as e:
        logger.error("web_search_failed", query=query, error=str(e))
        return json.dumps({
            "formatted": f"Search failed: {e}",
            "results": [],
            "query": query,
            "error": str(e),
        })


def _extract_structured_field(result: SearchResult, field_name: str) -> str | None:
    """Extract common structured fields from a search result payload."""
    name = field_name.strip().lower()
    content = result.content or ""
    combined = f"{result.title}\n{result.snippet}\n{content}".strip()

    if name == "title":
        return result.title
    if name == "url":
        return result.url
    if name == "snippet":
        return result.snippet
    if name == "summary":
        base = result.snippet or content or result.title
        return base[:400].strip()
    if name == "domain":
        parsed = urlparse(result.url)
        return parsed.netloc or None
    if name in {"published_at", "date", "published_date"}:
        match = re.search(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b", combined)
        return match.group(1) if match else None
    if name == "author":
        match = re.search(r"\bby\s+([A-Za-z.\- ]{2,80})", combined, flags=re.IGNORECASE)
        if match:
            author = match.group(1).strip()
            author = re.split(
                r"\s+(?:published|on|at)\b|\s+\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
                author,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            return author or None
        return None

    # Generic fallback: look for "field: value" pattern in snippet/content.
    escaped = re.escape(field_name)
    generic_match = re.search(rf"{escaped}\s*:\s*([^\n|]{{1,200}})", combined, flags=re.IGNORECASE)
    if generic_match:
        return generic_match.group(1).strip()
    return None


@tool(args_schema=WebExtractStructuredInput)
async def web_extract_structured(
    query: str | None = None,
    url: str | None = None,
    fields: list[str] | None = None,
    max_results: int = 3,
    search_depth: Literal["basic", "advanced"] = "basic",
) -> str:
    """Search the web and return schema-friendly structured extraction.

    This tool is optimized for downstream deterministic processing by returning
    normalized fields for each result.
    """
    selected_fields = [f for f in (fields or ["title", "url", "snippet", "summary"]) if f]
    logger.info(
        "web_extract_structured_tool_invoked",
        query=query,
        url=url,
        fields=selected_fields,
        max_results=max_results,
        search_depth=search_depth,
    )

    try:
        if url:
            fetched = await asyncio.to_thread(_fetch_page_for_structured_extraction, url)
            results = [fetched]
        else:
            if not query:
                return json.dumps({
                    "success": False,
                    "query": query or "",
                    "url": url,
                    "fields": selected_fields,
                    "count": 0,
                    "items": [],
                    "error": "Either query or url must be provided.",
                })
            results = await search_service.search_raw(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
            )
        if not results:
            return json.dumps({
                "success": True,
                "query": query or "",
                "url": url,
                "fields": selected_fields,
                "count": 0,
                "items": [],
            })

        items: list[dict[str, str | None]] = []
        for result in results:
            item: dict[str, str | None] = {}
            for field_name in selected_fields:
                item[field_name] = _extract_structured_field(result, field_name)
            items.append(item)

        return json.dumps({
            "success": True,
            "query": query or "",
            "url": url,
            "fields": selected_fields,
            "count": len(items),
            "items": items,
        })
    except ValueError as e:
        logger.warning("web_extract_structured_unavailable", error=str(e))
        return json.dumps({
            "success": False,
            "query": query or "",
            "url": url,
            "fields": selected_fields,
            "count": 0,
            "items": [],
            "error": str(e),
        })
    except Exception as e:
        logger.error("web_extract_structured_failed", query=query, url=url, error=str(e))
        return json.dumps({
            "success": False,
            "query": query or "",
            "url": url,
            "fields": selected_fields,
            "count": 0,
            "items": [],
            "error": str(e),
        })


def _format_results(results: list[SearchResult]) -> str:
    """Format search results for LLM consumption."""
    formatted = []
    for i, result in enumerate(results, 1):
        parts = [f"[{i}] {result.title}"]
        parts.append(f"    URL: {result.url}")
        if result.snippet:
            parts.append(f"    {result.snippet}")
        if result.relevance_score:
            parts.append(f"    Relevance: {result.relevance_score:.2f}")
        formatted.append("\n".join(parts))

    return "\n\n".join(formatted)


def parse_search_results(tool_output: str) -> list[SearchResult]:
    """Parse tool output JSON back into SearchResult objects.

    Args:
        tool_output: JSON string from web_search tool

    Returns:
        List of SearchResult objects
    """
    try:
        data = json.loads(tool_output)
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("snippet", ""),
                content=r.get("content"),
                relevance_score=r.get("relevance_score"),
            )
            for r in data.get("results", [])
        ]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("parse_search_results_failed", error=str(e))
        return []
