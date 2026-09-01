from __future__ import annotations

from app.core.config import settings
from app.llm.errors import LLMPermanentError
from app.llm.openai_compatible import OpenAICompatibleLLM
from app.llm.provider_config import nonempty_secret
from app.llm.registry import LLMAdapterRegistry


@LLMAdapterRegistry.register(
    "openrouter",
    online=True,
    default_url="https://openrouter.ai/api/v1",
)
class OpenRouterLLM(OpenAICompatibleLLM):
    """
    OpenRouter adapter (OpenAI-compatible Chat Completions).

    Model id comes from OPENROUTER_MODEL / CHAT_MODEL via the factory.
    Switching free → paid is a config change, not a code change.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: int | None = None,
        connect_timeout: int | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        resolved = (
            api_key
            if api_key is not None
            else settings.api_key_for_provider("openrouter")
        )
        if not nonempty_secret(resolved):
            raise LLMPermanentError(
                "OpenRouter requires an API key. "
                "Set OPENROUTER_API_KEY or LLM_API_KEY."
            )
        headers = {
            "HTTP-Referer": "http://localhost",
            "X-Title": settings.APP_NAME,
        }
        if extra_headers:
            headers.update(extra_headers)
        super().__init__(
            base_url=base_url,
            api_key=resolved,
            model=model,
            timeout_seconds=timeout_seconds,
            connect_timeout=connect_timeout,
            provider_name="openrouter",
            extra_headers=headers,
            send_stream_options=False,
        )

    @classmethod
    def connect(
        cls,
        *,
        provider: str,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_seconds: int,
    ) -> OpenRouterLLM:
        del provider
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )
