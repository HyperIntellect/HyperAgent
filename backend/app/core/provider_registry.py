"""Registry for LLM providers, supporting both built-in and custom OpenAI-compatible providers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CustomProviderConfig:
    """Configuration for a custom OpenAI-compatible provider."""

    name: str  # e.g. "deepseek"
    api_key: str  # Provider API key (empty string for local providers like Ollama)
    base_url: str  # e.g. "https://api.deepseek.com"
    default_model: str = ""  # e.g. "deepseek-chat" (optional if tier_models provided)
    display_name: str = ""  # Human-readable name for UI (defaults to name.title())
    tier_models: dict[str, str] = field(default_factory=dict)
    # Optional: {"max": "deepseek-reasoner", "pro": "deepseek-chat", "lite": "deepseek-chat"}
    image_model: str = ""  # Image generation model for this provider
    vision_model: str = ""  # Vision/image understanding model for this provider
    enable_thinking: bool = False  # Enable thinking/reasoning mode (e.g. Kimi, DeepSeek)

    def __post_init__(self):
        if not self.default_model and self.tier_models:
            # Pick from tier_models in priority order: pro > lite > max
            for tier_key in ("pro", "lite", "max"):
                if tier_key in self.tier_models:
                    self.default_model = self.tier_models[tier_key]
                    break


BUILTIN_PROVIDERS = {"anthropic", "openai", "gemini"}


class ProviderRegistry:
    """Registry that manages both built-in and custom LLM providers."""

    def __init__(self) -> None:
        self._custom_providers: dict[str, CustomProviderConfig] = {}

    def register(self, config: CustomProviderConfig) -> None:
        """Register a custom provider configuration."""
        if config.name in BUILTIN_PROVIDERS:
            raise ValueError(
                f"Cannot register '{config.name}' as a custom provider: "
                f"it conflicts with a built-in provider name."
            )
        self._custom_providers[config.name] = config

    def get_custom(self, name: str) -> CustomProviderConfig | None:
        """Get a custom provider config by name, or None if not found."""
        return self._custom_providers.get(name)

    def is_builtin(self, name: str) -> bool:
        """Check if a provider name is a built-in provider."""
        return name in BUILTIN_PROVIDERS

    def is_known(self, name: str) -> bool:
        """Check if a provider name is either built-in or registered as custom."""
        return name in BUILTIN_PROVIDERS or name in self._custom_providers

    def all_provider_names(self) -> list[str]:
        """Return all known provider names (built-in + custom)."""
        return sorted(BUILTIN_PROVIDERS | set(self._custom_providers.keys()))

    def all_custom_providers(self) -> list[CustomProviderConfig]:
        """Return all registered custom provider configs."""
        return list(self._custom_providers.values())


# Singleton instance
provider_registry = ProviderRegistry()
