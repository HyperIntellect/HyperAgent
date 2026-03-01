from typing import Any, Sequence

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.ai.model_tiers import (
    ModelTier,
    resolve_model,
    resolve_model_for_task,
)
from app.ai.thinking import ThinkingAwareChatOpenAI
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def extract_text_from_content(content: Any) -> str:
    """Extract text content from structured LLM chunk content.

    Handles various content formats:
    - String: returns as-is
    - List of content blocks: extracts text from text blocks, skips tool_use blocks
    - Dict: extracts text field if present
    - Other: converts to string

    Args:
        content: Content from LLM chunk (can be str, list, dict, etc.)

    Returns:
        Extracted text as string
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                # Extract text from text blocks
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(str(item["text"]))
                # Skip tool_use blocks (not meant for display)
                elif item.get("type") == "tool_use":
                    continue
                # Skip input_json_delta (intermediate states)
                elif item.get("type") == "input_json_delta":
                    continue
                # Try to extract text field if present
                elif "text" in item:
                    text_parts.append(str(item["text"]))
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                # Fallback: convert to string
                text_parts.append(str(item))
        return "".join(text_parts)

    if isinstance(content, dict):
        # Try to extract text field
        if "text" in content:
            return str(content["text"])
        # Skip tool_use blocks
        if content.get("type") == "tool_use":
            return ""
        # Skip input_json_delta
        if content.get("type") == "input_json_delta":
            return ""

    # Fallback: convert to string
    return str(content)


class LLMService:
    """Unified LLM service supporting multiple providers."""

    # Cache LLM clients by (provider, model) to avoid re-creating HTTP clients
    _cache: dict[tuple[str, str], BaseChatModel] = {}

    def _get_anthropic(self, model: str | None = None) -> ChatAnthropic:
        """Get Anthropic client."""
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")
        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=model or settings.tier_pro_anthropic,
            streaming=True,
            stream_usage=True,
            timeout=float(settings.llm_request_timeout),
            max_retries=settings.llm_max_retries,
        )

    def _get_openai(self, model: str | None = None) -> ChatOpenAI:
        """Get OpenAI client."""
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=model or settings.tier_pro_openai,
            streaming=True,
            stream_usage=True,
            request_timeout=float(settings.llm_request_timeout),
            max_retries=settings.llm_max_retries,
        )

    def _get_gemini(self, model: str | None = None) -> ChatGoogleGenerativeAI:
        """Get Google Gemini client (supports both API key and Vertex AI)."""
        if settings.gemini_use_vertex_ai:
            if not settings.gcp_project_id:
                raise ValueError("GCP_PROJECT_ID must be set when using Vertex AI")
            return ChatGoogleGenerativeAI(
                model=model or settings.tier_pro_gemini,
                vertexai=True,
                project=settings.gcp_project_id,
                location=settings.gcp_location,
                streaming=True,
                timeout=float(settings.llm_request_timeout),
                max_retries=settings.llm_max_retries,
            )
        if not settings.gemini_api_key:
            raise ValueError("Gemini API key not configured")
        return ChatGoogleGenerativeAI(
            api_key=settings.gemini_api_key,
            model=model or settings.tier_pro_gemini,
            streaming=True,
            timeout=float(settings.llm_request_timeout),
            max_retries=settings.llm_max_retries,
        )

    def _get_custom_openai_compatible(
        self, provider_name: str, model: str | None = None
    ) -> ChatOpenAI:
        """Get an OpenAI-compatible client for a custom provider."""
        from app.core.provider_registry import provider_registry

        config = provider_registry.get_custom(provider_name)
        if config is None:
            raise ValueError(f"Unknown custom provider: {provider_name}")
        effective_model = model or config.default_model
        if not effective_model:
            raise ValueError(
                f"No model specified for custom provider '{provider_name}' "
                f"and no default_model configured"
            )

        # Pass enable_thinking via extra_body (OpenAI SDK's extra_body param)
        extra_body: dict[str, Any] | None = None
        if config.enable_thinking:
            extra_body = {"enable_thinking": True}

        client = ThinkingAwareChatOpenAI(
            api_key=config.api_key or "not-needed",
            base_url=config.base_url,
            model=effective_model,
            streaming=True,
            request_timeout=float(settings.llm_request_timeout),
            max_retries=settings.llm_max_retries,
            extra_body=extra_body,
            thinking_mode=config.enable_thinking,
        )

        return client

    def get_llm(self, provider: str = "anthropic", model: str | None = None) -> BaseChatModel:
        """Get LLM instance for the specified provider."""
        from app.core.provider_registry import provider_registry

        # Resolve the effective model name for cache lookup
        if provider == "anthropic":
            effective_model = model or settings.tier_pro_anthropic
        elif provider == "openai":
            effective_model = model or settings.tier_pro_openai
        elif provider == "gemini":
            effective_model = model or settings.tier_pro_gemini
        else:
            # Custom provider
            config = provider_registry.get_custom(provider)
            if config is None:
                raise ValueError(f"Unknown provider: {provider}")
            effective_model = model or config.default_model
            if not effective_model:
                raise ValueError(
                    f"No model specified for custom provider '{provider}' "
                    f"and no default_model configured"
                )

        cache_key = (provider, effective_model)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if provider == "anthropic":
            client = self._get_anthropic(model)
        elif provider == "openai":
            client = self._get_openai(model)
        elif provider == "gemini":
            client = self._get_gemini(model)
        else:
            client = self._get_custom_openai_compatible(provider, model)

        logger.debug("llm_client_created", provider=provider, model=effective_model)
        self._cache[cache_key] = client
        return client

    def get_llm_for_tier(
        self,
        tier: ModelTier,
        provider: str | None = None,
        model_override: str | None = None,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
    ) -> BaseChatModel:
        """Get LLM for a specific tier, with optional model override.

        Args:
            tier: The model tier to use
            provider: The LLM provider (if None, auto-selects based on tier configuration)
            model_override: Optional model override (bypasses tier selection)
            callbacks: Optional callback handlers (e.g. UsageTracker)

        Returns:
            LLM instance configured for the specified tier
        """
        p, m = resolve_model(tier, provider, model_override)
        logger.info("llm_invocation", tier=tier.value, provider=p, model=m)
        llm = self.get_llm(p, m)
        if callbacks:
            return llm.with_config(callbacks=list(callbacks))
        return llm

    def choose_llm_for_task(
        self,
        task_type: str,
        provider: str | None = None,
        tier_override: ModelTier | None = None,
        model_override: str | None = None,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
    ) -> BaseChatModel:
        """Choose LLM based on task type with auto-routing.

        Priority: model_override > tier_override > auto-routing

        Args:
            task_type: Type of task (e.g., "research", "task", "code")
            provider: The LLM provider (if None, auto-selects based on tier configuration)
            tier_override: Optional tier override (bypasses auto-routing)
            model_override: Optional model override (bypasses tier and auto-routing)
            callbacks: Optional callback handlers (e.g. UsageTracker)

        Returns:
            LLM instance configured for the task
        """
        tier, p, m = resolve_model_for_task(task_type, provider, tier_override, model_override)
        logger.info("llm_invocation", task_type=task_type, tier=tier.value, provider=p, model=m)
        llm = self.get_llm(p, m)
        if callbacks:
            return llm.with_config(callbacks=list(callbacks))
        return llm

    async def generate_title(
        self,
        message: str,
        provider: str = "anthropic",
        model: str | None = None,
    ) -> str:
        """Generate a concise title for a conversation based on the user's message.

        Args:
            message: The user's input message
            provider: LLM provider to use
            model: Optional model override

        Returns:
            A concise title (max 5-6 words)
        """
        logger.info("llm_invocation", task_type="naming", provider=provider, model=model)
        llm = self.get_llm(provider, model)

        prompt = (
            "Generate a very concise, meaningful title for an AI conversation "
            "that starts with the following user message. "
            "Respond ONLY with the title text (no quotes, no period, max 6 words):\n\n"
            f"'{message}'"
        )

        messages = [HumanMessage(content=prompt)]
        response = await llm.ainvoke(messages)
        title = extract_text_from_content(response.content).strip()

        # Clean up the title
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        elif title.startswith("'") and title.endswith("'"):
            title = title[1:-1]

        return title


# Global instance
llm_service = LLMService()
