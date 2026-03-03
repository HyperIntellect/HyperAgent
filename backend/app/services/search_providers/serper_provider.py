"""Serper search provider — Google Search via Serper API."""

import httpx

from app.config import settings
from app.core.logging import get_logger
from app.services.search import SearchResult
from app.services.search_providers.base import SearchProvider

logger = get_logger(__name__)

SERPER_API_URL = "https://google.serper.dev/search"


class SerperSearchProvider(SearchProvider):
    """Search provider using Google Search via Serper API."""

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

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(SERPER_API_URL, json=payload, headers=headers)
            if resp.status_code != 200:
                logger.error("serper_search_failed", status=resp.status_code, body=resp.text[:200])
                raise RuntimeError(f"Serper API error: {resp.status_code}")
            data = resp.json()

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
