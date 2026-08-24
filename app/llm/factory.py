from __future__ import annotations

from app.core.config import settings
from app.llm.base import BaseLLM
from app.llm.gateway import LLMGateway
from app.llm.model_capabilities import ModelCapabilities, ModelCapabilityRegistry
from app.llm.provider_config import (
    effective_llm_timeout,
    normalize_provider,
    resolve_chat_model,
    resolve_llm_base_url,
)
from app.llm.registry import LLMAdapterRegistry

# Load adapters so they self-register. Add a new provider by creating a
# BaseLLM subclass with @LLMAdapterRegistry.register(...) — do not expand
# this factory with per-vendor if/elif blocks.
from app.llm.anthropic import AnthropicLLM as _AnthropicLLM  # noqa: F401
from app.llm.ollama import OllamaLLM as _OllamaLLM  # noqa: F401
from app.llm.openai_compatible import OpenAICompatibleLLM as _OpenAICompatibleLLM  # noqa: F401


class LLMFactory:
    """Build provider adapters and wrap them in LLMGateway."""

    @staticmethod
    def create() -> BaseLLM:
        provider = settings.LLM_PROVIDER
        primary = LLMFactory._build_provider(
            provider=provider,
            base_url=settings.LLM_URL,
            model=settings.CHAT_MODEL,
            api_key=settings.api_key_for_provider(provider),
        )
        fallback = None
        if settings.LLM_FALLBACK_PROVIDER:
            fallback = LLMFactory._build_provider(
                provider=settings.LLM_FALLBACK_PROVIDER,
                base_url=settings.LLM_FALLBACK_URL or settings.LLM_URL,
                model=settings.LLM_FALLBACK_MODEL or settings.CHAT_MODEL,
                api_key=(
                    settings.LLM_FALLBACK_API_KEY
                    or settings.api_key_for_provider(settings.LLM_FALLBACK_PROVIDER)
                ),
            )
        timeout = effective_llm_timeout(
            settings.LLM_PROVIDER,
            settings.LLM_TIMEOUT,
        )
        return LLMGateway(primary, fallback=fallback, timeout_seconds=timeout)

    @staticmethod
    def create_raw(
        *,
        provider: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> BaseLLM:
        """Create an unwrapped provider adapter (tests / diagnostics)."""
        resolved_provider = provider or settings.LLM_PROVIDER
        return LLMFactory._build_provider(
            provider=resolved_provider,
            base_url=base_url or settings.LLM_URL,
            model=model or settings.CHAT_MODEL,
            api_key=(
                api_key
                if api_key is not None
                else settings.api_key_for_provider(resolved_provider)
            ),
        )

    @staticmethod
    def capabilities(
        *,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> ModelCapabilities:
        return ModelCapabilityRegistry.resolve(
            provider=provider,
            model_name=model_name,
        )

    @staticmethod
    def registered_providers() -> list[str]:
        return LLMAdapterRegistry.registered_names()

    @staticmethod
    def _build_provider(
        *,
        provider: str,
        base_url: str,
        model: str,
        api_key: str | None,
    ) -> BaseLLM:
        name = normalize_provider(provider)
        resolved_model = resolve_chat_model(name, model)
        timeout = effective_llm_timeout(name, settings.LLM_TIMEOUT)
        return LLMAdapterRegistry.connect(
            provider=name,
            base_url=resolve_llm_base_url(name, base_url),
            model=resolved_model,
            api_key=api_key,
            timeout_seconds=timeout,
        )
