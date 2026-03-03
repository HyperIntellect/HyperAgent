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
