# Agentic Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace web_research and deep_research skills with a single agentic search system that decomposes queries, searches in parallel across multiple providers, evaluates coverage, and returns structured findings.

**Architecture:** Layered LangGraph state machine with classify → plan → parallel search → evaluate → refine loop → synthesize. Tiered: trivial queries fast-path through single-shot search; complex queries get full agentic pipeline. Multi-provider search via abstract SearchProvider interface.

**Tech Stack:** LangGraph, LangChain, Pydantic, asyncio, Tavily API, Serper API, Jina Reader API

**Design doc:** `docs/plans/2026-03-03-agentic-search-design.md`

---

### Task 1: Search Provider Base Classes

**Files:**
- Create: `backend/app/services/search_providers/__init__.py`
- Create: `backend/app/services/search_providers/base.py`
- Test: `backend/tests/test_search_providers.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_search_providers.py
"""Tests for search provider base classes and registry."""

import pytest
from app.services.search_providers.base import (
    SearchProvider,
    SearchProviderRegistry,
)
from app.services.search import SearchResult


class FakeProvider(SearchProvider):
    """Test provider."""

    @property
    def name(self) -> str:
        return "fake"

    @property
    def supported_types(self) -> list[str]:
        return ["factual", "news"]

    async def search(self, query: str, max_results: int = 5, **kwargs) -> list[SearchResult]:
        return [SearchResult(title="Fake", url="https://fake.com", snippet=query)]


class FakeProvider2(SearchProvider):
    @property
    def name(self) -> str:
        return "fake2"

    @property
    def supported_types(self) -> list[str]:
        return ["academic"]

    async def search(self, query: str, max_results: int = 5, **kwargs) -> list[SearchResult]:
        return []


def test_registry_register_and_list():
    reg = SearchProviderRegistry()
    p = FakeProvider()
    reg.register(p)
    assert "fake" in reg.list_providers()


def test_registry_get_provider_for_type():
    reg = SearchProviderRegistry()
    reg.register(FakeProvider())
    reg.register(FakeProvider2())
    assert reg.get_provider_for_type("factual").name == "fake"
    assert reg.get_provider_for_type("academic").name == "fake2"


def test_registry_fallback_to_default():
    reg = SearchProviderRegistry()
    p = FakeProvider()
    reg.register(p)
    reg.set_default("fake")
    # Unknown type falls back to default
    assert reg.get_provider_for_type("unknown_type").name == "fake"


def test_registry_no_provider_raises():
    reg = SearchProviderRegistry()
    with pytest.raises(ValueError, match="No search provider"):
        reg.get_provider_for_type("factual")


@pytest.mark.asyncio
async def test_registry_search_routes_to_provider():
    reg = SearchProviderRegistry()
    reg.register(FakeProvider())
    reg.set_default("fake")
    results = await reg.search("test query", query_type="factual")
    assert len(results) == 1
    assert results[0].title == "Fake"
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_search_providers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.search_providers'`

**Step 3: Write the implementation**

```python
# backend/app/services/search_providers/__init__.py
"""Search provider registry and auto-registration."""

from app.services.search_providers.base import SearchProvider, SearchProviderRegistry

__all__ = ["SearchProvider", "SearchProviderRegistry", "get_search_registry"]

# Global singleton
_registry: SearchProviderRegistry | None = None


def get_search_registry() -> SearchProviderRegistry:
    """Get the global search provider registry, initializing if needed."""
    global _registry
    if _registry is None:
        _registry = SearchProviderRegistry()
        _auto_register_providers(_registry)
    return _registry


def _auto_register_providers(registry: SearchProviderRegistry) -> None:
    """Auto-register providers based on available API keys."""
    from app.config import settings
    from app.core.logging import get_logger

    logger = get_logger(__name__)

    # Always register Tavily if key is available (it's the default)
    if settings.tavily_api_key:
        from app.services.search_providers.tavily_provider import TavilySearchProvider

        registry.register(TavilySearchProvider())
        registry.set_default("tavily")
        logger.info("search_provider_registered", provider="tavily")

    # Register Serper if key is available
    serper_key = getattr(settings, "serper_api_key", "")
    if serper_key:
        from app.services.search_providers.serper_provider import SerperSearchProvider

        registry.register(SerperSearchProvider())
        logger.info("search_provider_registered", provider="serper")

    # Register Jina if key is available
    jina_key = getattr(settings, "jina_api_key", "")
    if jina_key:
        from app.services.search_providers.jina_provider import JinaReaderProvider

        registry.register(JinaReaderProvider())
        logger.info("search_provider_registered", provider="jina")
```

```python
# backend/app/services/search_providers/base.py
"""Abstract search provider interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.services.search import SearchResult


class SearchProvider(ABC):
    """Abstract base class for search providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider name (e.g., 'tavily', 'serper')."""
        ...

    @property
    @abstractmethod
    def supported_types(self) -> list[str]:
        """Query types this provider handles well.

        Types: factual, comparative, temporal, statistical, exploratory,
               news, academic, technical, recent, deep_read
        """
        ...

    @abstractmethod
    async def search(
        self, query: str, max_results: int = 5, **kwargs
    ) -> list[SearchResult]:
        """Execute a search query.

        Args:
            query: Search query string.
            max_results: Maximum results to return.
            **kwargs: Provider-specific options.

        Returns:
            List of SearchResult objects.
        """
        ...


class SearchProviderRegistry:
    """Registry that routes searches to the best available provider."""

    def __init__(self) -> None:
        self._providers: dict[str, SearchProvider] = {}
        self._type_map: dict[str, str] = {}  # query_type -> provider_name
        self._default: str | None = None

    def register(self, provider: SearchProvider) -> None:
        """Register a search provider."""
        self._providers[provider.name] = provider
        for qtype in provider.supported_types:
            # First provider to claim a type wins (unless overridden)
            if qtype not in self._type_map:
                self._type_map[qtype] = provider.name

    def set_default(self, provider_name: str) -> None:
        """Set the default fallback provider."""
        if provider_name not in self._providers:
            raise ValueError(f"Provider not registered: {provider_name}")
        self._default = provider_name

    def list_providers(self) -> list[str]:
        """List registered provider names."""
        return list(self._providers.keys())

    def get_provider_for_type(self, query_type: str) -> SearchProvider:
        """Get the best provider for a query type.

        Falls back to the default provider if no specific mapping exists.

        Raises:
            ValueError: If no provider is available.
        """
        name = self._type_map.get(query_type)
        if name and name in self._providers:
            return self._providers[name]
        if self._default and self._default in self._providers:
            return self._providers[self._default]
        raise ValueError(
            f"No search provider available for type '{query_type}'. "
            f"Registered: {list(self._providers.keys())}"
        )

    async def search(
        self, query: str, query_type: str = "factual", max_results: int = 5, **kwargs
    ) -> list[SearchResult]:
        """Route a search to the appropriate provider.

        Args:
            query: Search query.
            query_type: Type of query for provider selection.
            max_results: Maximum results.

        Returns:
            List of SearchResult objects.
        """
        provider = self.get_provider_for_type(query_type)
        return await provider.search(query, max_results=max_results, **kwargs)
```

**Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_search_providers.py -v`
Expected: All 5 tests PASS

---

### Task 2: Tavily Search Provider

**Files:**
- Create: `backend/app/services/search_providers/tavily_provider.py`
- Test: `backend/tests/test_search_providers.py` (append)

**Step 1: Write the failing test**

Append to `backend/tests/test_search_providers.py`:

```python
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_tavily_provider_search():
    """Test TavilySearchProvider wraps search_service correctly."""
    from app.services.search_providers.tavily_provider import TavilySearchProvider

    provider = TavilySearchProvider()
    assert provider.name == "tavily"
    assert "factual" in provider.supported_types

    mock_results = [SearchResult(title="Test", url="https://test.com", snippet="snippet")]
    with patch("app.services.search_providers.tavily_provider.search_service") as mock_svc:
        mock_svc.search_raw = AsyncMock(return_value=mock_results)
        results = await provider.search("test query", max_results=3)
        assert len(results) == 1
        assert results[0].title == "Test"
        mock_svc.search_raw.assert_called_once_with(
            query="test query", max_results=3, search_depth="basic"
        )


@pytest.mark.asyncio
async def test_tavily_provider_advanced_depth():
    from app.services.search_providers.tavily_provider import TavilySearchProvider

    provider = TavilySearchProvider()
    with patch("app.services.search_providers.tavily_provider.search_service") as mock_svc:
        mock_svc.search_raw = AsyncMock(return_value=[])
        await provider.search("test", search_depth="advanced")
        mock_svc.search_raw.assert_called_once_with(
            query="test", max_results=5, search_depth="advanced"
        )
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_search_providers.py::test_tavily_provider_search -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# backend/app/services/search_providers/tavily_provider.py
"""Tavily search provider — wraps the existing SearchService."""

from app.services.search import SearchResult, search_service
from app.services.search_providers.base import SearchProvider


class TavilySearchProvider(SearchProvider):
    """Search provider using Tavily API (general-purpose, default)."""

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def supported_types(self) -> list[str]:
        return ["factual", "comparative", "exploratory", "technical", "statistical"]

    async def search(
        self, query: str, max_results: int = 5, **kwargs
    ) -> list[SearchResult]:
        search_depth = kwargs.get("search_depth", "basic")
        return await search_service.search_raw(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
        )
```

**Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_search_providers.py -v`
Expected: All tests PASS

---

### Task 3: Serper Search Provider

**Files:**
- Create: `backend/app/services/search_providers/serper_provider.py`
- Test: `backend/tests/test_search_providers.py` (append)

**Step 1: Write the failing test**

Append to `backend/tests/test_search_providers.py`:

```python
@pytest.mark.asyncio
async def test_serper_provider_search():
    from app.services.search_providers.serper_provider import SerperSearchProvider

    provider = SerperSearchProvider()
    assert provider.name == "serper"
    assert "news" in provider.supported_types
    assert "temporal" in provider.supported_types

    mock_response = {
        "organic": [
            {
                "title": "News Article",
                "link": "https://news.com/article",
                "snippet": "Breaking news snippet",
                "position": 1,
            }
        ]
    }

    with patch("app.services.search_providers.serper_provider.aiohttp") as mock_aiohttp:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_aiohttp.ClientSession = AsyncMock(return_value=mock_session)

        with patch("app.services.search_providers.serper_provider.settings") as mock_settings:
            mock_settings.serper_api_key = "test-key"
            results = await provider.search("breaking news", max_results=3)

    assert len(results) == 1
    assert results[0].title == "News Article"
    assert results[0].url == "https://news.com/article"
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_search_providers.py::test_serper_provider_search -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# backend/app/services/search_providers/serper_provider.py
"""Serper search provider — Google Search via Serper API."""

import aiohttp

from app.config import settings
from app.core.logging import get_logger
from app.services.search import SearchResult
from app.services.search_providers.base import SearchProvider

logger = get_logger(__name__)

SERPER_API_URL = "https://google.serper.dev/search"


class SerperSearchProvider(SearchProvider):
    """Search provider using Google Search via Serper API.

    Best for: news, temporal queries, recent events.
    """

    @property
    def name(self) -> str:
        return "serper"

    @property
    def supported_types(self) -> list[str]:
        return ["news", "temporal", "recent"]

    async def search(
        self, query: str, max_results: int = 5, **kwargs
    ) -> list[SearchResult]:
        api_key = getattr(settings, "serper_api_key", "")
        if not api_key:
            raise ValueError("SERPER_API_KEY not configured")

        payload = {"q": query, "num": max_results}
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

        logger.info("serper_search_started", query=query[:80], max_results=max_results)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                SERPER_API_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=20)
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("serper_search_failed", status=resp.status, body=body[:200])
                    raise RuntimeError(f"Serper API error: {resp.status}")
                data = await resp.json()

        results: list[SearchResult] = []
        for item in data.get("organic", [])[:max_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    content=None,
                    relevance_score=1.0 / (item.get("position", 99)),
                )
            )

        logger.info("serper_search_completed", query=query[:80], results=len(results))
        return results
```

**Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_search_providers.py::test_serper_provider_search -v`
Expected: PASS

---

### Task 4: Jina Reader Provider

**Files:**
- Create: `backend/app/services/search_providers/jina_provider.py`
- Test: `backend/tests/test_search_providers.py` (append)

**Step 1: Write the failing test**

Append to `backend/tests/test_search_providers.py`:

```python
@pytest.mark.asyncio
async def test_jina_provider_search():
    from app.services.search_providers.jina_provider import JinaReaderProvider

    provider = JinaReaderProvider()
    assert provider.name == "jina"
    assert "deep_read" in provider.supported_types

    mock_text = "# Page Title\n\nThis is the full page content extracted by Jina."

    with patch("app.services.search_providers.jina_provider.aiohttp") as mock_aiohttp:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.text = AsyncMock(return_value=mock_text)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_aiohttp.ClientSession = AsyncMock(return_value=mock_session)

        with patch("app.services.search_providers.jina_provider.settings") as mock_settings:
            mock_settings.jina_api_key = "test-key"
            results = await provider.search(
                "https://example.com/article", max_results=1
            )

    assert len(results) == 1
    assert results[0].url == "https://example.com/article"
    assert "full page content" in results[0].content
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_search_providers.py::test_jina_provider_search -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# backend/app/services/search_providers/jina_provider.py
"""Jina Reader provider — deep page content extraction."""

import aiohttp

from app.config import settings
from app.core.logging import get_logger
from app.services.search import SearchResult
from app.services.search_providers.base import SearchProvider

logger = get_logger(__name__)

JINA_READER_URL = "https://r.jina.ai/"


class JinaReaderProvider(SearchProvider):
    """Deep content extraction using Jina Reader.

    Not a search engine — fetches and extracts full markdown content
    from a given URL. Use for follow-up deep reading of specific pages.

    The `query` parameter should be a URL for this provider.
    """

    @property
    def name(self) -> str:
        return "jina"

    @property
    def supported_types(self) -> list[str]:
        return ["deep_read"]

    async def search(
        self, query: str, max_results: int = 1, **kwargs
    ) -> list[SearchResult]:
        """Extract content from a URL using Jina Reader.

        Args:
            query: The URL to extract content from.
            max_results: Ignored (always returns 1 result).
        """
        api_key = getattr(settings, "jina_api_key", "")
        if not api_key:
            raise ValueError("JINA_API_KEY not configured")

        url = query  # For Jina, the query IS the URL
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/markdown",
        }

        logger.info("jina_reader_started", url=url[:100])

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{JINA_READER_URL}{url}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("jina_reader_failed", status=resp.status, body=body[:200])
                    raise RuntimeError(f"Jina Reader error: {resp.status}")
                content = await resp.text()

        # Extract title from first markdown heading
        title = url
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break

        snippet = content[:300].strip()

        logger.info("jina_reader_completed", url=url[:100], content_len=len(content))
        return [
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                content=content[:15000],  # Cap at 15k chars
                relevance_score=None,
            )
        ]
```

**Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_search_providers.py -v`
Expected: All tests PASS

---

### Task 5: Configuration — Add New Env Vars

**Files:**
- Modify: `backend/app/config.py:146-148` (after `tavily_api_key`)
- Modify: `backend/.env.example`

**Step 1: Add new settings to config.py**

In `backend/app/config.py`, after line 147 (`tavily_api_key: str = ""`), add:

```python
    # Additional search providers
    serper_api_key: str = ""  # Google Search via Serper API
    jina_api_key: str = ""    # Jina Reader for deep content extraction

    # Agentic search tuning
    agentic_search_max_refinements: int = 3
    agentic_search_confidence_threshold: float = 0.7
    agentic_search_max_sub_queries: int = 6
```

**Step 2: Update .env.example**

Add to `.env.example` in the Search section:

```env
# Search Providers
TAVILY_API_KEY=
SERPER_API_KEY=           # Optional — enables Google Search via Serper
JINA_API_KEY=             # Optional — enables Jina Reader for deep content

# Agentic Search
AGENTIC_SEARCH_MAX_REFINEMENTS=3
AGENTIC_SEARCH_CONFIDENCE_THRESHOLD=0.7
AGENTIC_SEARCH_MAX_SUB_QUERIES=6
```

**Step 3: Verify config loads**

Run: `cd backend && python -c "from app.config import settings; print('serper_api_key' in dir(settings))"`
Expected: `True`

---

### Task 6: New Event Types for Agentic Search

**Files:**
- Modify: `backend/app/agents/events.py`
- Test: `backend/tests/test_agentic_search.py` (new)

**Step 1: Write the failing test**

```python
# backend/tests/test_agentic_search.py
"""Tests for agentic search skill."""

from app.agents.events import (
    search_plan,
    sub_query_status,
    knowledge_update,
    confidence_update,
    refinement_start,
)


def test_search_plan_event():
    event = search_plan(sub_queries=[
        {"id": "sq1", "query": "test", "type": "factual", "provider": "tavily"}
    ])
    assert event["type"] == "search_plan"
    assert len(event["sub_queries"]) == 1


def test_sub_query_status_event():
    event = sub_query_status(id="sq1", status="searching")
    assert event["type"] == "sub_query_status"
    assert event["id"] == "sq1"


def test_knowledge_update_event():
    event = knowledge_update(facts_count=5, sources_count=3)
    assert event["type"] == "knowledge_update"
    assert event["facts_count"] == 5


def test_confidence_update_event():
    event = confidence_update(confidence=0.75, coverage_summary={"sq1": 0.8})
    assert event["type"] == "confidence_update"
    assert event["confidence"] == 0.75


def test_refinement_start_event():
    event = refinement_start(round=1, follow_up_queries=["query1", "query2"])
    assert event["type"] == "refinement_start"
    assert event["round"] == 1
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_agentic_search.py::test_search_plan_event -v`
Expected: FAIL — `ImportError: cannot import name 'search_plan'`

**Step 3: Add event types and factory functions to events.py**

Add to the `EventType` enum in `backend/app/agents/events.py`:

```python
    # Agentic search events
    SEARCH_PLAN = "search_plan"
    SUB_QUERY_STATUS = "sub_query_status"
    KNOWLEDGE_UPDATE = "knowledge_update"
    CONFIDENCE_UPDATE = "confidence_update"
    REFINEMENT_START = "refinement_start"
```

Add Pydantic models:

```python
class SearchPlanEvent(BaseModel):
    type: Literal["search_plan"] = "search_plan"
    sub_queries: list[dict[str, Any]] = Field(..., description="Planned sub-queries")
    timestamp: int = Field(default_factory=_timestamp)


class SubQueryStatusEvent(BaseModel):
    type: Literal["sub_query_status"] = "sub_query_status"
    id: str = Field(..., description="Sub-query ID")
    status: str = Field(..., description="pending|searching|done|gap")
    coverage: float | None = Field(default=None, description="Coverage score 0-1")
    timestamp: int = Field(default_factory=_timestamp)


class KnowledgeUpdateEvent(BaseModel):
    type: Literal["knowledge_update"] = "knowledge_update"
    facts_count: int = Field(..., description="Total extracted facts")
    sources_count: int = Field(..., description="Total unique sources")
    timestamp: int = Field(default_factory=_timestamp)


class ConfidenceUpdateEvent(BaseModel):
    type: Literal["confidence_update"] = "confidence_update"
    confidence: float = Field(..., description="Overall confidence 0-1")
    coverage_summary: dict[str, float] = Field(
        default_factory=dict, description="Per sub-query coverage"
    )
    timestamp: int = Field(default_factory=_timestamp)


class RefinementStartEvent(BaseModel):
    type: Literal["refinement_start"] = "refinement_start"
    round: int = Field(..., description="Refinement round number")
    follow_up_queries: list[str] = Field(default_factory=list)
    timestamp: int = Field(default_factory=_timestamp)
```

Add factory functions:

```python
def search_plan(sub_queries: list[dict[str, Any]]) -> dict[str, Any]:
    return SearchPlanEvent(sub_queries=sub_queries).model_dump()


def sub_query_status(
    id: str, status: str, coverage: float | None = None
) -> dict[str, Any]:
    return SubQueryStatusEvent(id=id, status=status, coverage=coverage).model_dump()


def knowledge_update(facts_count: int, sources_count: int) -> dict[str, Any]:
    return KnowledgeUpdateEvent(
        facts_count=facts_count, sources_count=sources_count
    ).model_dump()


def confidence_update(
    confidence: float, coverage_summary: dict[str, float] | None = None
) -> dict[str, Any]:
    return ConfidenceUpdateEvent(
        confidence=confidence, coverage_summary=coverage_summary or {}
    ).model_dump()


def refinement_start(
    round: int, follow_up_queries: list[str] | None = None
) -> dict[str, Any]:
    return RefinementStartEvent(
        round=round, follow_up_queries=follow_up_queries or []
    ).model_dump()
```

**Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_agentic_search.py -v`
Expected: All 5 tests PASS

---

### Task 7: Domain Credibility Scoring

**Files:**
- Create: `backend/app/services/search_providers/credibility.py`
- Test: `backend/tests/test_search_providers.py` (append)

**Step 1: Write the failing test**

Append to `backend/tests/test_search_providers.py`:

```python
def test_domain_credibility_scoring():
    from app.services.search_providers.credibility import score_domain

    assert score_domain("https://www.nasa.gov/article") == 0.9
    assert score_domain("https://mit.edu/research") == 0.9
    assert score_domain("https://www.nytimes.com/2024/article") == 0.85
    assert score_domain("https://en.wikipedia.org/wiki/Test") == 0.7
    assert score_domain("https://docs.python.org/3/library") == 0.8
    assert score_domain("https://stackoverflow.com/questions/123") == 0.7
    assert score_domain("https://random-blog.xyz/post") == 0.5
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_search_providers.py::test_domain_credibility_scoring -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# backend/app/services/search_providers/credibility.py
"""Domain credibility scoring for source evaluation."""

from urllib.parse import urlparse

# TLD-based scoring
_TLD_SCORES: dict[str, float] = {
    ".gov": 0.9,
    ".edu": 0.9,
    ".org": 0.65,
}

# Domain pattern scoring (checked in order, first match wins)
_DOMAIN_SCORES: list[tuple[list[str], float]] = [
    # Major news outlets
    (
        [
            "nytimes.com", "bbc.com", "bbc.co.uk", "reuters.com",
            "apnews.com", "washingtonpost.com", "theguardian.com",
            "bloomberg.com", "ft.com", "economist.com", "wsj.com",
        ],
        0.85,
    ),
    # Wikipedia
    (["wikipedia.org"], 0.7),
    # Official documentation
    (["docs.python.org", "docs.microsoft.com", "developer.mozilla.org",
      "developer.apple.com", "cloud.google.com", "aws.amazon.com"], 0.8),
    # Tech community
    (["stackoverflow.com", "stackexchange.com", "github.com", "gitlab.com"], 0.7),
    # Academic
    (["arxiv.org", "scholar.google.com", "pubmed.ncbi.nlm.nih.gov",
      "semanticscholar.org", "nature.com", "science.org"], 0.85),
]


def score_domain(url: str) -> float:
    """Score a URL's domain credibility (0.0-1.0).

    Args:
        url: Full URL string.

    Returns:
        Credibility score between 0.0 and 1.0.
    """
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
    except Exception:
        return 0.5

    # Check TLD
    for tld, score in _TLD_SCORES.items():
        if hostname.endswith(tld):
            return score

    # Check specific domains
    for domains, score in _DOMAIN_SCORES:
        for domain in domains:
            if hostname == domain or hostname.endswith(f".{domain}"):
                return score

    # Check docs.* pattern
    if hostname.startswith("docs.") or hostname.startswith("developer."):
        return 0.8

    return 0.5  # Unknown domain default
```

**Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_search_providers.py::test_domain_credibility_scoring -v`
Expected: PASS

---

### Task 8: Agentic Search Skill — State Model & Classify Node

**Files:**
- Create: `backend/app/agents/skills/builtin/agentic_search_skill.py`
- Test: `backend/tests/test_agentic_search.py` (append)

**Step 1: Write the failing test**

Append to `backend/tests/test_agentic_search.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_classify_trivial_query():
    """Test that simple factual queries are classified as trivial."""
    from app.agents.skills.builtin.agentic_search_skill import AgenticSearchSkill

    skill = AgenticSearchSkill()
    assert skill.metadata.id == "agentic_search"

    # Mock the LLM to return a trivial classification
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"complexity": "trivial", "intent_signals": ["factual"]}'
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("app.agents.skills.builtin.agentic_search_skill.llm_service") as mock_svc:
        mock_svc.get_llm_for_tier = MagicMock(return_value=mock_llm)
        graph = skill.create_graph()
        # Just verify the graph compiles without error
        assert graph is not None
```

**Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_agentic_search.py::test_classify_trivial_query -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the skill skeleton with state model and classify node**

```python
# backend/app/agents/skills/builtin/agentic_search_skill.py
"""Agentic Search Skill — multi-step search with query decomposition and evaluation.

Replaces web_research and deep_research skills with a unified agentic search
that decomposes queries, searches in parallel, evaluates coverage, and
returns structured findings.
"""

import asyncio
import json
import operator
import uuid
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.agents import events as agent_events
from app.agents.skills.skill_base import Skill, SkillMetadata, SkillParameter, SkillState
from app.agents.state import _override_reducer
from app.ai.llm import extract_text_from_content, llm_service
from app.ai.model_tiers import ModelTier
from app.core.logging import get_logger
from app.services.search import SearchResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# State types
# ---------------------------------------------------------------------------


class SubQuery(TypedDict):
    id: str
    query: str
    type: str       # factual|comparative|temporal|statistical|exploratory
    provider: str   # preferred search provider name
    priority: int   # 1-3 (3 = highest)
    status: str     # pending|searched|sufficient|gap


class ExtractedFact(TypedDict):
    claim: str
    source_urls: list[str]
    sub_query_id: str
    confidence: float


class SourceEntry(TypedDict):
    title: str
    url: str
    snippet: str
    content: str | None
    domain: str
    credibility: float
    used_for: list[str]


class Contradiction(TypedDict):
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    sub_query_id: str


class SearchKnowledge(TypedDict, total=False):
    sub_queries: list[SubQuery]
    facts: list[ExtractedFact]
    sources: dict[str, SourceEntry]       # url -> entry
    contradictions: list[Contradiction]
    coverage_scores: dict[str, float]     # sq_id -> 0.0-1.0
    overall_confidence: float
    refinement_round: int


class AgenticSearchState(SkillState, total=False):
    # Input
    query: str
    complexity: Annotated[str, _override_reducer]      # trivial | complex
    intent_signals: list[str]

    # Knowledge
    knowledge: Annotated[dict, _override_reducer]       # SearchKnowledge as dict

    # LLM context for evaluation/refinement
    lc_messages: Annotated[list[BaseMessage], operator.add]

    # Loop control
    max_refinements: Annotated[int, _override_reducer]
    current_refinement: Annotated[int, _override_reducer]

    # Events
    pending_events: Annotated[list[dict[str, Any]], operator.add]

    # Context passthrough
    locale: str
    provider: str | None
    model: str | None
    tier: Any | None


# ---------------------------------------------------------------------------
# Classify prompt
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = """You are a search query classifier. Given a user query, determine:

1. **complexity**: Is this "trivial" (can be answered with a single web search) or "complex" (needs multiple searches, comparison, analysis)?

Trivial examples: "what is the capital of France", "python list sort syntax", "current weather in Tokyo"
Complex examples: "compare React vs Vue for enterprise apps in 2025", "what are the economic impacts of AI on healthcare", "latest developments in quantum computing and their commercial applications"

2. **intent_signals**: What types of information are needed? Pick 1-3 from:
   - factual: simple facts, definitions, how-to
   - comparative: comparing multiple things
   - temporal: time-sensitive, recent events, trends
   - statistical: numbers, data, metrics
   - exploratory: broad topic exploration

Respond in JSON only:
{"complexity": "trivial"|"complex", "intent_signals": ["factual", ...]}"""


# ---------------------------------------------------------------------------
# Plan prompt
# ---------------------------------------------------------------------------

PLAN_PROMPT = """You are a search strategist. Given a complex query, decompose it into 2-{max_sub_queries} focused sub-queries.

For each sub-query, specify:
- "query": the search query string (specific and focused)
- "type": one of [factual, comparative, temporal, statistical, exploratory]
- "provider": preferred search provider — "tavily" (general), "serper" (news/recent), or "tavily" as default
- "priority": 1-3 (3 = most important to answer the main query)

Respond in JSON only:
{{"sub_queries": [{{"query": "...", "type": "...", "provider": "...", "priority": N}}, ...]}}"""


# ---------------------------------------------------------------------------
# Evaluate prompt
# ---------------------------------------------------------------------------

EVALUATE_PROMPT = """You are a research evaluator. Given sub-queries and the search results gathered so far, assess:

1. For each sub-query, rate coverage from 0.0 (no useful results) to 1.0 (fully answered).
2. Identify any contradictions between sources.
3. Decide: is the information sufficient to answer the original query?

Original query: {query}

Sub-queries and their results:
{sub_query_summary}

Respond in JSON only:
{{
  "coverage": {{"sq_id": score, ...}},
  "contradictions": [{{"claim_a": "...", "source_a": "url", "claim_b": "...", "source_b": "url", "sub_query_id": "..."}}],
  "sufficient": true|false,
  "gap_analysis": "what information is still missing (empty string if sufficient)"
}}"""


# ---------------------------------------------------------------------------
# Refine prompt
# ---------------------------------------------------------------------------

REFINE_PROMPT = """You are a search strategist refining a research plan.

Original query: {query}
Gap analysis: {gap_analysis}

Current sub-queries and their coverage:
{coverage_summary}

Generate 1-3 follow-up sub-queries to fill the gaps. Use different search terms
than the original queries. Be specific.

Respond in JSON only:
{{"follow_up_queries": [{{"query": "...", "type": "...", "provider": "...", "priority": N}}, ...]}}"""


# ---------------------------------------------------------------------------
# Synthesize prompt
# ---------------------------------------------------------------------------

SYNTHESIZE_PROMPT = """You are a research synthesizer. Given the accumulated facts and sources,
produce a structured summary answering the original query.

Original query: {query}

Accumulated facts:
{facts_summary}

Sources:
{sources_summary}

Contradictions:
{contradictions_summary}

Respond in JSON only:
{{
  "facts": [
    {{"claim": "...", "sources": ["url1", "url2"], "confidence": 0.0-1.0, "sub_query": "sq_id"}}
  ],
  "summary": "2-3 paragraph synthesis answering the original query with inline citations [1], [2]",
  "unanswered": ["aspects of the query that couldn't be fully answered"],
  "confidence": 0.0-1.0
}}"""


# ---------------------------------------------------------------------------
# AgenticSearchSkill
# ---------------------------------------------------------------------------


class AgenticSearchSkill(Skill):
    """Agentic search with query decomposition, parallel search, and evaluation."""

    metadata = SkillMetadata(
        id="agentic_search",
        name="Agentic Search",
        version="1.0.0",
        description=(
            "Multi-step agentic search that decomposes queries, searches in parallel "
            "across multiple providers, evaluates coverage, and returns structured findings."
        ),
        category="research",
        parameters=[
            SkillParameter(
                name="query",
                type="string",
                description="The search query or research question",
                required=True,
            ),
            SkillParameter(
                name="max_sources",
                type="number",
                description="Maximum sources per sub-query (1-10)",
                required=False,
                default=5,
            ),
        ],
        output_schema={
            "type": "object",
            "properties": {
                "facts": {"type": "array"},
                "sources": {"type": "array"},
                "overall_confidence": {"type": "number"},
                "coverage": {"type": "object"},
                "unanswered": {"type": "array"},
                "contradictions": {"type": "array"},
                "complexity": {"type": "string"},
                "summary": {"type": "string"},
            },
        },
        required_tools=["web_search"],
        risk_level="medium",
        side_effect_level="low",
        data_sensitivity="public",
        network_scope="external",
        max_execution_time_seconds=300,
        max_iterations=20,
        tags=["search", "research", "agentic"],
    )

    def create_graph(self):
        """Build the agentic search LangGraph."""
        graph = StateGraph(AgenticSearchState)

        # ---- Node: classify ----
        async def classify(state: AgenticSearchState) -> dict:
            params = state.get("input_params", {})
            query = params.get("query", "")
            locale = params.get("locale", state.get("locale", "en"))
            provider = params.get("provider", state.get("provider"))
            model = params.get("model", state.get("model"))
            tier = params.get("tier", state.get("tier"))
            max_sources = int(params.get("max_sources", 5))

            from app.config import settings

            pending_events: list[dict] = [
                agent_events.stage("classifying", "Analyzing query complexity...", "running"),
            ]

            try:
                llm = llm_service.get_llm_for_tier(ModelTier.LITE, provider=provider)
                resp = await llm.ainvoke([
                    SystemMessage(content=CLASSIFY_PROMPT),
                    HumanMessage(content=f"Query: {query}"),
                ])
                text = extract_text_from_content(resp.content) or ""
                # Parse JSON from response
                result = json.loads(text)
                complexity = result.get("complexity", "complex")
                intent_signals = result.get("intent_signals", ["factual"])
            except Exception as e:
                logger.warning("classify_failed_defaulting_complex", error=str(e))
                complexity = "complex"
                intent_signals = ["exploratory"]

            pending_events.append(
                agent_events.stage("classifying", f"Query classified as {complexity}", "completed")
            )

            # Initialize knowledge
            knowledge: SearchKnowledge = {
                "sub_queries": [],
                "facts": [],
                "sources": {},
                "contradictions": [],
                "coverage_scores": {},
                "overall_confidence": 0.0,
                "refinement_round": 0,
            }

            return {
                "query": query,
                "complexity": complexity,
                "intent_signals": intent_signals,
                "knowledge": knowledge,
                "max_refinements": settings.agentic_search_max_refinements,
                "current_refinement": 0,
                "locale": locale,
                "provider": provider,
                "model": model,
                "tier": tier,
                "pending_events": pending_events,
            }

        # ---- Node: quick_search ----
        async def quick_search(state: AgenticSearchState) -> dict:
            """Fast path for trivial queries — single search, no LLM planning."""
            query = state.get("query", "")
            intent_signals = state.get("intent_signals", ["factual"])
            pending_events: list[dict] = [
                agent_events.stage("searching", "Quick search...", "running"),
            ]

            from app.services.search_providers import get_search_registry

            registry = get_search_registry()
            query_type = intent_signals[0] if intent_signals else "factual"

            try:
                results = await registry.search(query, query_type=query_type, max_results=5)
            except Exception as e:
                logger.error("quick_search_failed", error=str(e))
                results = []

            # Build knowledge from results
            knowledge = dict(state.get("knowledge", {}))
            from app.services.search_providers.credibility import score_domain

            for r in results:
                url = r.url
                if url not in knowledge.get("sources", {}):
                    knowledge.setdefault("sources", {})[url] = {
                        "title": r.title,
                        "url": url,
                        "snippet": r.snippet,
                        "content": r.content,
                        "domain": url.split("/")[2] if "/" in url else "",
                        "credibility": score_domain(url),
                        "used_for": ["quick"],
                    }
                pending_events.append(
                    agent_events.source(
                        title=r.title, url=url, snippet=r.snippet,
                        relevance_score=r.relevance_score,
                    )
                )

            knowledge["overall_confidence"] = 0.7 if results else 0.0
            pending_events.append(
                agent_events.stage("searching", "Quick search complete", "completed"),
            )

            return {
                "knowledge": knowledge,
                "pending_events": pending_events,
            }

        # ---- Node: plan_search ----
        async def plan_search(state: AgenticSearchState) -> dict:
            """Decompose complex query into sub-queries."""
            query = state.get("query", "")
            provider = state.get("provider")
            pending_events: list[dict] = [
                agent_events.stage("planning", "Decomposing query into sub-queries...", "running"),
            ]

            from app.config import settings

            max_sq = settings.agentic_search_max_sub_queries

            try:
                llm = llm_service.get_llm_for_tier(ModelTier.PRO, provider=provider)
                prompt = PLAN_PROMPT.replace("{max_sub_queries}", str(max_sq))
                resp = await llm.ainvoke([
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"Query: {query}"),
                ])
                text = extract_text_from_content(resp.content) or ""
                plan = json.loads(text)
                raw_sqs = plan.get("sub_queries", [])
            except Exception as e:
                logger.warning("plan_search_failed", error=str(e))
                # Fallback: use the original query as a single sub-query
                raw_sqs = [{"query": query, "type": "exploratory", "provider": "tavily", "priority": 3}]

            sub_queries: list[SubQuery] = []
            for i, sq in enumerate(raw_sqs[:max_sq]):
                sub_queries.append({
                    "id": f"sq{i+1}",
                    "query": sq.get("query", query),
                    "type": sq.get("type", "factual"),
                    "provider": sq.get("provider", "tavily"),
                    "priority": sq.get("priority", 2),
                    "status": "pending",
                })

            knowledge = dict(state.get("knowledge", {}))
            knowledge["sub_queries"] = sub_queries

            pending_events.append(
                agent_events.search_plan(
                    sub_queries=[
                        {"id": sq["id"], "query": sq["query"], "type": sq["type"], "provider": sq["provider"]}
                        for sq in sub_queries
                    ]
                )
            )
            pending_events.append(
                agent_events.stage("planning", f"Planned {len(sub_queries)} sub-queries", "completed"),
            )

            return {
                "knowledge": knowledge,
                "pending_events": pending_events,
            }

        # ---- Node: execute_search ----
        async def execute_search(state: AgenticSearchState) -> dict:
            """Execute pending sub-queries in parallel."""
            knowledge = dict(state.get("knowledge", {}))
            sub_queries = knowledge.get("sub_queries", [])
            pending_events: list[dict] = [
                agent_events.stage("searching", "Executing searches in parallel...", "running"),
            ]

            from app.services.search_providers import get_search_registry
            from app.services.search_providers.credibility import score_domain

            registry = get_search_registry()
            max_sources = int(state.get("input_params", {}).get("max_sources", 5))

            # Find pending sub-queries
            pending_sqs = [sq for sq in sub_queries if sq["status"] == "pending"]

            async def _search_one(sq: SubQuery) -> tuple[str, list[SearchResult]]:
                pending_events.append(
                    agent_events.sub_query_status(id=sq["id"], status="searching")
                )
                try:
                    results = await registry.search(
                        sq["query"],
                        query_type=sq["type"],
                        max_results=max_sources,
                    )
                    return sq["id"], results
                except Exception as e:
                    logger.error("sub_query_search_failed", sq_id=sq["id"], error=str(e))
                    return sq["id"], []

            # Fan out searches
            tasks = [_search_one(sq) for sq in pending_sqs]
            results_by_sq = await asyncio.gather(*tasks)

            # Process results into knowledge
            sources = dict(knowledge.get("sources", {}))
            for sq_id, results in results_by_sq:
                for r in results:
                    url = r.url
                    if url not in sources:
                        sources[url] = {
                            "title": r.title,
                            "url": url,
                            "snippet": r.snippet,
                            "content": r.content,
                            "domain": url.split("/")[2] if len(url.split("/")) > 2 else "",
                            "credibility": score_domain(url),
                            "used_for": [sq_id],
                        }
                        pending_events.append(
                            agent_events.source(
                                title=r.title, url=url, snippet=r.snippet,
                                relevance_score=r.relevance_score,
                            )
                        )
                    else:
                        if sq_id not in sources[url]["used_for"]:
                            sources[url]["used_for"].append(sq_id)

                # Mark sub-query as searched
                for sq in sub_queries:
                    if sq["id"] == sq_id:
                        sq["status"] = "searched"

            knowledge["sources"] = sources
            knowledge["sub_queries"] = sub_queries

            pending_events.append(
                agent_events.knowledge_update(
                    facts_count=len(knowledge.get("facts", [])),
                    sources_count=len(sources),
                )
            )
            pending_events.append(
                agent_events.stage("searching", f"Found {len(sources)} sources", "completed"),
            )

            return {
                "knowledge": knowledge,
                "pending_events": pending_events,
            }

        # ---- Node: evaluate ----
        async def evaluate(state: AgenticSearchState) -> dict:
            """Evaluate search coverage and decide if refinement is needed."""
            query = state.get("query", "")
            knowledge = dict(state.get("knowledge", {}))
            provider = state.get("provider")
            pending_events: list[dict] = [
                agent_events.stage("evaluating", "Evaluating search coverage...", "running"),
            ]

            from app.config import settings

            # Build sub-query summary for the LLM
            sub_queries = knowledge.get("sub_queries", [])
            sources = knowledge.get("sources", {})
            summary_parts = []
            for sq in sub_queries:
                sq_sources = [
                    f"  - {s['title']}: {s['snippet'][:150]}"
                    for s in sources.values()
                    if sq["id"] in s.get("used_for", [])
                ]
                summary_parts.append(
                    f"[{sq['id']}] {sq['query']} (type: {sq['type']}, status: {sq['status']})\n"
                    + ("\n".join(sq_sources) if sq_sources else "  No results found.")
                )

            sub_query_summary = "\n\n".join(summary_parts)

            try:
                llm = llm_service.get_llm_for_tier(ModelTier.PRO, provider=provider)
                prompt = EVALUATE_PROMPT.format(
                    query=query, sub_query_summary=sub_query_summary
                )
                resp = await llm.ainvoke([
                    SystemMessage(content="You are a research evaluator. Respond in JSON only."),
                    HumanMessage(content=prompt),
                ])
                text = extract_text_from_content(resp.content) or ""
                evaluation = json.loads(text)
            except Exception as e:
                logger.warning("evaluate_failed", error=str(e))
                evaluation = {"coverage": {}, "contradictions": [], "sufficient": True, "gap_analysis": ""}

            # Update knowledge
            coverage = evaluation.get("coverage", {})
            knowledge["coverage_scores"] = coverage
            knowledge["contradictions"] = evaluation.get("contradictions", [])

            # Calculate overall confidence (weighted by priority)
            total_weight = 0.0
            weighted_sum = 0.0
            for sq in sub_queries:
                sq_coverage = coverage.get(sq["id"], 0.5)
                weight = sq["priority"]
                weighted_sum += sq_coverage * weight
                total_weight += weight
                # Update sub-query status
                sq["status"] = "sufficient" if sq_coverage >= 0.6 else "gap"

            overall = weighted_sum / total_weight if total_weight > 0 else 0.0
            knowledge["overall_confidence"] = overall
            knowledge["sub_queries"] = sub_queries

            # Store gap analysis for refine node
            knowledge["_gap_analysis"] = evaluation.get("gap_analysis", "")
            knowledge["_sufficient"] = evaluation.get("sufficient", True)

            pending_events.append(
                agent_events.confidence_update(
                    confidence=overall, coverage_summary=coverage
                )
            )
            pending_events.append(
                agent_events.stage("evaluating", f"Confidence: {overall:.0%}", "completed"),
            )

            return {
                "knowledge": knowledge,
                "pending_events": pending_events,
            }

        # ---- Node: refine ----
        async def refine(state: AgenticSearchState) -> dict:
            """Generate follow-up queries for gaps."""
            query = state.get("query", "")
            knowledge = dict(state.get("knowledge", {}))
            provider = state.get("provider")
            current_refinement = state.get("current_refinement", 0)

            pending_events: list[dict] = []

            gap_analysis = knowledge.pop("_gap_analysis", "")
            sub_queries = knowledge.get("sub_queries", [])
            coverage = knowledge.get("coverage_scores", {})

            coverage_summary = "\n".join(
                f"  [{sq['id']}] {sq['query']} — coverage: {coverage.get(sq['id'], 0):.1%}"
                for sq in sub_queries
            )

            try:
                llm = llm_service.get_llm_for_tier(ModelTier.PRO, provider=provider)
                prompt = REFINE_PROMPT.format(
                    query=query,
                    gap_analysis=gap_analysis,
                    coverage_summary=coverage_summary,
                )
                resp = await llm.ainvoke([
                    SystemMessage(content="You are a search strategist. Respond in JSON only."),
                    HumanMessage(content=prompt),
                ])
                text = extract_text_from_content(resp.content) or ""
                result = json.loads(text)
                follow_ups = result.get("follow_up_queries", [])
            except Exception as e:
                logger.warning("refine_failed", error=str(e))
                follow_ups = []

            # Add follow-up sub-queries
            existing_count = len(sub_queries)
            for i, sq in enumerate(follow_ups[:3]):
                sub_queries.append({
                    "id": f"sq{existing_count + i + 1}",
                    "query": sq.get("query", ""),
                    "type": sq.get("type", "factual"),
                    "provider": sq.get("provider", "tavily"),
                    "priority": sq.get("priority", 2),
                    "status": "pending",
                })

            knowledge["sub_queries"] = sub_queries
            knowledge["refinement_round"] = current_refinement + 1

            pending_events.append(
                agent_events.refinement_start(
                    round=current_refinement + 1,
                    follow_up_queries=[sq.get("query", "") for sq in follow_ups[:3]],
                )
            )

            return {
                "knowledge": knowledge,
                "current_refinement": current_refinement + 1,
                "pending_events": pending_events,
            }

        # ---- Node: synthesize ----
        async def synthesize(state: AgenticSearchState) -> dict:
            """Build structured output from accumulated knowledge."""
            query = state.get("query", "")
            knowledge = dict(state.get("knowledge", {}))
            complexity = state.get("complexity", "trivial")
            provider = state.get("provider")

            pending_events: list[dict] = [
                agent_events.stage("synthesizing", "Building structured findings...", "running"),
            ]

            sources = knowledge.get("sources", {})
            sub_queries = knowledge.get("sub_queries", [])
            contradictions = knowledge.get("contradictions", [])
            overall_confidence = knowledge.get("overall_confidence", 0.0)

            # For trivial queries, skip LLM synthesis
            if complexity == "trivial":
                source_list = [
                    {"title": s["title"], "url": s["url"], "snippet": s["snippet"], "credibility": s["credibility"]}
                    for s in sources.values()
                ]
                pending_events.append(
                    agent_events.stage("synthesizing", "Complete", "completed"),
                )
                return {
                    "output": {
                        "facts": [],
                        "sources": source_list,
                        "overall_confidence": overall_confidence,
                        "coverage": knowledge.get("coverage_scores", {}),
                        "unanswered": [],
                        "contradictions": [],
                        "complexity": complexity,
                        "summary": "",
                        "sub_queries_count": 0,
                        "sources_count": len(source_list),
                        "refinement_rounds": 0,
                        "provider_usage": {},
                    },
                    "pending_events": pending_events,
                }

            # Complex: use LLM to synthesize
            facts_summary = "\n".join(
                f"- [{s['title']}] ({s['url']}): {s['snippet']}"
                for s in sources.values()
            )
            sources_summary = "\n".join(
                f"[{i+1}] {s['title']} — {s['url']} (credibility: {s['credibility']:.1f})"
                for i, s in enumerate(sources.values())
            )
            contradictions_summary = (
                "\n".join(
                    f"- {c.get('claim_a', '')} vs {c.get('claim_b', '')} ({c.get('source_a', '')} vs {c.get('source_b', '')})"
                    for c in contradictions
                )
                if contradictions
                else "No contradictions detected."
            )

            try:
                llm = llm_service.get_llm_for_tier(ModelTier.PRO, provider=provider)
                prompt = SYNTHESIZE_PROMPT.format(
                    query=query,
                    facts_summary=facts_summary,
                    sources_summary=sources_summary,
                    contradictions_summary=contradictions_summary,
                )
                resp = await llm.ainvoke([
                    SystemMessage(content="You are a research synthesizer. Respond in JSON only."),
                    HumanMessage(content=prompt),
                ])
                text = extract_text_from_content(resp.content) or ""
                synthesis = json.loads(text)
            except Exception as e:
                logger.warning("synthesize_failed", error=str(e))
                synthesis = {"facts": [], "summary": "", "unanswered": [], "confidence": overall_confidence}

            source_list = [
                {"title": s["title"], "url": s["url"], "snippet": s["snippet"], "credibility": s["credibility"]}
                for s in sources.values()
            ]

            # Count provider usage
            provider_usage: dict[str, int] = {}
            for sq in sub_queries:
                p = sq.get("provider", "tavily")
                provider_usage[p] = provider_usage.get(p, 0) + 1

            unanswered = synthesis.get("unanswered", [])
            # Also add gap sub-queries
            for sq in sub_queries:
                if sq["status"] == "gap":
                    unanswered.append(sq["query"])

            pending_events.append(
                agent_events.stage("synthesizing", "Complete", "completed"),
            )

            return {
                "output": {
                    "facts": synthesis.get("facts", []),
                    "sources": source_list,
                    "overall_confidence": synthesis.get("confidence", overall_confidence),
                    "coverage": knowledge.get("coverage_scores", {}),
                    "unanswered": unanswered,
                    "contradictions": [
                        {"claim_a": c.get("claim_a"), "claim_b": c.get("claim_b"),
                         "source_a": c.get("source_a"), "source_b": c.get("source_b")}
                        for c in contradictions
                    ],
                    "complexity": complexity,
                    "summary": synthesis.get("summary", ""),
                    "sub_queries_count": len(sub_queries),
                    "sources_count": len(source_list),
                    "refinement_rounds": knowledge.get("refinement_round", 0),
                    "provider_usage": provider_usage,
                },
                "pending_events": pending_events,
            }

        # ---- Conditional edges ----
        def after_classify(state: AgenticSearchState) -> str:
            if state.get("complexity") == "trivial":
                return "quick_search"
            return "plan_search"

        def after_evaluate(state: AgenticSearchState) -> str:
            knowledge = state.get("knowledge", {})
            current_ref = state.get("current_refinement", 0)
            max_ref = state.get("max_refinements", 3)

            from app.config import settings

            threshold = settings.agentic_search_confidence_threshold
            sufficient = knowledge.get("_sufficient", True)
            confidence = knowledge.get("overall_confidence", 0.0)

            if sufficient or confidence >= threshold or current_ref >= max_ref:
                return "synthesize"
            return "refine"

        # ---- Build graph ----
        graph.add_node("classify", classify)
        graph.add_node("quick_search", quick_search)
        graph.add_node("plan_search", plan_search)
        graph.add_node("execute_search", execute_search)
        graph.add_node("evaluate", evaluate)
        graph.add_node("refine", refine)
        graph.add_node("synthesize", synthesize)

        graph.set_entry_point("classify")
        graph.add_conditional_edges(
            "classify",
            after_classify,
            {"quick_search": "quick_search", "plan_search": "plan_search"},
        )
        graph.add_edge("quick_search", "synthesize")
        graph.add_edge("plan_search", "execute_search")
        graph.add_edge("execute_search", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            after_evaluate,
            {"synthesize": "synthesize", "refine": "refine"},
        )
        graph.add_edge("refine", "execute_search")
        graph.add_edge("synthesize", END)

        return graph.compile()
```

**Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_agentic_search.py -v`
Expected: All tests PASS

---

### Task 9: Register Skill in Builtin Package

**Files:**
- Modify: `backend/app/agents/skills/builtin/__init__.py`
- Modify: `backend/app/services/skill_registry.py:257-281`

**Step 1: Update `__init__.py` to import AgenticSearchSkill**

In `backend/app/agents/skills/builtin/__init__.py`, add:

```python
from app.agents.skills.builtin.agentic_search_skill import AgenticSearchSkill
```

And add `"AgenticSearchSkill"` to `__all__`.

**Step 2: Add to skill registry initialization**

In `backend/app/services/skill_registry.py`, in `_register_builtin_skills()`, add the import:

```python
from app.agents.skills.builtin import (
    AgenticSearchSkill,
    AppBuilderSkill,
    ...
)
```

And add `AgenticSearchSkill` to the registration loop list.

**Step 3: Verify registration**

Run: `cd backend && python -c "from app.agents.skills.builtin import AgenticSearchSkill; print(AgenticSearchSkill().metadata.id)"`
Expected: `agentic_search`

---

### Task 10: Task Agent Integration

**Files:**
- Modify: `backend/app/agents/subagents/task.py:309-312`

**Step 1: Update `_DIRECT_SKILL_MODES`**

In `backend/app/agents/subagents/task.py`, change line 309-312 from:

```python
            "research": {
                "skill_id": "deep_research",
                "param_key": "query",
            },
```

To:

```python
            "research": {
                "skill_id": "agentic_search",
                "param_key": "query",
            },
```

**Step 2: Verify the change**

Run: `cd backend && python -c "
from app.agents.subagents.task import _get_cached_task_tools
# Just verify module imports without error
print('task agent imports OK')
"`
Expected: `task agent imports OK`

---

### Task 11: Deprecation Aliases for Old Skills

**Files:**
- Modify: `backend/app/agents/skills/builtin/web_research_skill.py` (add deprecation)
- Modify: `backend/app/agents/skills/builtin/deep_research_skill.py` (add deprecation)

**Step 1: Add deprecation notice to web_research_skill.py**

Add at the top of the `WebResearchSkill` class docstring:

```python
"""DEPRECATED: Use AgenticSearchSkill instead.
Kept as alias for backward compatibility."""
```

**Step 2: Add deprecation notice to deep_research_skill.py**

Add at the top of the `DeepResearchSkill` class docstring:

```python
"""DEPRECATED: Use AgenticSearchSkill instead.
Kept as alias for backward compatibility."""
```

No code changes — these files remain importable but are no longer the primary research path.

---

### Task 12: Run Full Test Suite

**Step 1: Run all new tests**

Run: `cd backend && python -m pytest tests/test_search_providers.py tests/test_agentic_search.py -v`
Expected: All tests PASS

**Step 2: Run existing tests to verify no regressions**

Run: `cd backend && python -m pytest tests/ -v --timeout=60`
Expected: All existing tests still PASS

**Step 3: Lint check**

Run: `cd backend && make lint-backend`
Expected: No new lint errors

---

### Task 13: Install New Dependency (aiohttp)

**Step 1: Check if aiohttp is already installed**

Run: `cd backend && python -c "import aiohttp; print(aiohttp.__version__)"`

If not installed:

Run: `cd backend && pip install aiohttp`

Then add to `requirements.txt` or `pyproject.toml` depending on which the project uses.

**Step 2: Verify**

Run: `cd backend && python -c "from app.services.search_providers.serper_provider import SerperSearchProvider; print('OK')"`
Expected: `OK`
