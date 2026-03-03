"""Jina Reader provider — deep page content extraction."""

import httpx

from app.config import settings
from app.core.logging import get_logger
from app.services.search import SearchResult
from app.services.search_providers.base import SearchProvider

logger = get_logger(__name__)

JINA_READER_URL = "https://r.jina.ai/"


class JinaReaderProvider(SearchProvider):
    """Deep content extraction using Jina Reader.

    Not a search engine — fetches and extracts full markdown content
    from a given URL. The `query` parameter should be a URL.
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
        api_key = getattr(settings, "jina_api_key", "")
        if not api_key:
            raise ValueError("JINA_API_KEY not configured")

        url = query
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/markdown",
        }

        logger.info("jina_reader_started", url=url[:100])

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{JINA_READER_URL}{url}", headers=headers)
            if resp.status_code != 200:
                logger.error("jina_reader_failed", status=resp.status_code, body=resp.text[:200])
                raise RuntimeError(f"Jina Reader error: {resp.status_code}")
            content = resp.text

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
                content=content[:15000],
                relevance_score=None,
            )
        ]
