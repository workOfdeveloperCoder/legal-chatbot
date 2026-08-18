from __future__ import annotations

from app.core.config import settings
from app.llm.base import BaseLLM
from app.llm.gateway import LLMGateway
from app.llm.model_capabilities import ModelCapabilities, ModelCapabilityRegistry
from app.llm.ollama import OllamaLLM
from app.llm.openai_compatible import OpenAICompatibleLLM


class LLMFactory:
    """Build provider adapters and wrap them in LLMGateway."""

    @staticmethod
    def create() -> BaseLLM:
        primary = LLMFactory._build_provider(
            provider=settings.LLM_PROVIDER,
            base_url=settings.LLM_URL,
            model=settings.CHAT_MODEL,
            api_key=settings.LLM_API_KEY,
        )
        fallback = None
        if settings.LLM_FALLBACK_PROVIDER:
            fallback = LLMFactory._build_provider(
                provider=settings.LLM_FALLBACK_PROVIDER,
                base_url=settings.LLM_FALLBACK_URL or settings.LLM_URL,
                model=settings.LLM_FALLBACK_MODEL or settings.CHAT_MODEL,
                api_key=settings.LLM_FALLBACK_API_KEY or settings.LLM_API_KEY,
            )
        return LLMGateway(primary, fallback=fallback)

    @staticmethod
    def create_raw(
        *,
        provider: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> BaseLLM:
        """Create an unwrapped provider adapter (tests / diagnostics)."""
        return LLMFactory._build_provider(
            provider=provider or settings.LLM_PROVIDER,
            base_url=base_url or settings.LLM_URL,
            model=model or settings.CHAT_MODEL,
            api_key=api_key if api_key is not None else settings.LLM_API_KEY,
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
    def _build_provider(
        *,
        provider: str,
        base_url: str,
        model: str,
        api_key: str | None,
    ) -> BaseLLM:
        name = (provider or "ollama").lower().strip()
        if name == "ollama":
            return OllamaLLM(base_url=base_url, model=model)
        if name in {"openai", "openai_compatible", "openai-compatible"}:
            return OpenAICompatibleLLM(
                base_url=base_url,
                api_key=api_key,
                model=model,
                provider_name="openai" if name == "openai" else "openai_compatible",
            )
        raise ValueError(f"Unsupported LLM provider: {provider}")
