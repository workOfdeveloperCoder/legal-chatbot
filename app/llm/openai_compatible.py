from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.config import settings
from app.llm.base import BaseLLM
from app.llm.errors import LLMPermanentError, LLMTransientError
from app.llm.model_capabilities import ModelCapabilities, ModelCapabilityRegistry
from app.llm.registry import LLMAdapterRegistry
from app.llm.provider_config import adapter_provider_name
from app.rag.reasoning_cleanup import strip_reasoning_output
from app.schemas.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

logger = logging.getLogger(__name__)


@LLMAdapterRegistry.register(
    "openai",
    "openai_compatible",
    "deepseek",
    "groq",
    "openrouter",
    "gemini",
    "mistral",
    "together",
    "fireworks",
    online=True,
    default_urls={
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
        "mistral": "https://api.mistral.ai/v1",
        "together": "https://api.together.xyz/v1",
        "fireworks": "https://api.fireworks.ai/inference/v1",
    },
)
class OpenAICompatibleLLM(BaseLLM):
    """
    OpenAI Chat Completions-compatible adapter.

    Use this for OpenAI and any hosted API that speaks the same protocol
    (DeepSeek, Groq, OpenRouter, Gemini OpenAI mode, Mistral, Together).
    For a different wire protocol, add a new BaseLLM subclass instead.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: int | None = None,
        connect_timeout: int | None = None,
        provider_name: str = "openai_compatible",
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model or settings.CHAT_MODEL
        self._api_key = (
            api_key
            if api_key is not None
            else settings.api_key_for_provider(provider_name)
        )
        base = (base_url or settings.LLM_URL).rstrip("/")
        # Accept /v1, or a vendor root that already includes the protocol prefix
        # (Gemini: .../v1beta/openai). Do not append a second /v1.
        if base.endswith("/v1") or base.endswith("/openai"):
            self._base_url = base
        else:
            self._base_url = f"{base}/v1"

        read_timeout = timeout_seconds or settings.LLM_TIMEOUT
        connect = connect_timeout or settings.LLM_CONNECT_TIMEOUT
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        self.client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(
                connect=connect,
                read=read_timeout,
                write=connect,
                pool=connect,
            ),
        )
        self._caps = ModelCapabilityRegistry.resolve(
            provider=provider_name,
            model_name=self.model_name,
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
    ) -> OpenAICompatibleLLM:
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            provider_name=adapter_provider_name(provider),
        )

    def capabilities(self) -> ModelCapabilities:
        return self._caps

    def _completion_payload(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [m.model_dump() for m in request.messages],
            "stream": stream,
        }
        if self._caps.supports_temperature:
            payload["temperature"] = request.temperature
        if self._caps.uses_max_completion_tokens:
            payload["max_completion_tokens"] = request.max_tokens
        else:
            payload["max_tokens"] = request.max_tokens
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        payload = self._completion_payload(request, stream=False)
        try:
            response = await self.client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTransientError(f"OpenAI-compatible timeout: {exc}") from exc
        except httpx.TransportError as exc:
            raise LLMTransientError(f"OpenAI-compatible transport error: {exc}") from exc

        if response.status_code in {401, 403}:
            raise LLMPermanentError(
                f"OpenAI-compatible auth failed ({response.status_code})"
            )
        if response.status_code in {400, 404, 422}:
            raise LLMPermanentError(
                f"OpenAI-compatible bad request ({response.status_code}): "
                f"{response.text[:200]}"
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise LLMTransientError(
                f"OpenAI-compatible transient HTTP {response.status_code}"
            )
        if response.status_code >= 400:
            raise LLMPermanentError(
                f"OpenAI-compatible HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMPermanentError("Malformed JSON from OpenAI-compatible API") from exc

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMPermanentError(
                "Malformed OpenAI-compatible response: missing choices/message"
            ) from exc

        if content is None:
            raise LLMPermanentError("Empty content from OpenAI-compatible API")

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if total_tokens is None and prompt_tokens is not None:
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        return ChatCompletionResponse(
            content=strip_reasoning_output(str(content)),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    async def generate_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        if not settings.LLM_ENABLE_STREAMING:
            async for chunk in super().generate_stream(request):
                yield chunk
            return

        payload = self._completion_payload(request, stream=True)
        try:
            async with self.client.stream(
                "POST",
                "/chat/completions",
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise LLMTransientError(
                        f"Stream failed HTTP {response.status_code}: "
                        f"{body[:200]!r}"
                    )
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    # Minimal SSE parse — yield content deltas only.
                    import json

                    try:
                        event = json.loads(data)
                        delta = event["choices"][0].get("delta") or {}
                        piece = delta.get("content")
                        if piece:
                            yield strip_reasoning_output(piece)
                    except (ValueError, KeyError, IndexError, TypeError):
                        continue
        except httpx.TimeoutException as exc:
            raise LLMTransientError(f"Stream timeout: {exc}") from exc
        except httpx.TransportError as exc:
            raise LLMTransientError(f"Stream transport error: {exc}") from exc

    async def aclose(self) -> None:
        await self.client.aclose()
