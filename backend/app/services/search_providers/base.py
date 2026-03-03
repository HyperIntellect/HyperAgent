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
        """Execute a search query."""
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
        """Get the best provider for a query type. Falls back to default."""
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
        """Route a search to the appropriate provider."""
        provider = self.get_provider_for_type(query_type)
        return await provider.search(query, max_results=max_results, **kwargs)
