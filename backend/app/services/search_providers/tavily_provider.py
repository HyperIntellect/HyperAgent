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
