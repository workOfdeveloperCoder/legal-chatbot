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
from app.rag.reasoning_cleanup import strip_reasoning_output
from app.schemas.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

logger = logging.getLogger(__name__)

_ANTHROPIC_VERSION = "2023-06-01"


@LLMAdapterRegistry.register(
    "anthropic",
    "claude",
    online=True,
    default_url="https://api.anthropic.com",
)
class AnthropicLLM(BaseLLM):
    """
    Anthropic Messages API adapter (Claude).

    Wire protocol is not OpenAI-compatible. RAG still speaks
    ChatCompletionRequest — this class translates.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: int | None = None,
        connect_timeout: int | None = None,
        provider_name: str = "anthropic",
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model or settings.CHAT_MODEL
        self._api_key = (
            api_key
            if api_key is not None
            else settings.api_key_for_provider("anthropic")
        )
        base = (base_url or "https://api.anthropic.com").rstrip("/")
        if base.endswith("/v1"):
            self._base_url = base
        else:
            self._base_url = f"{base}/v1"

        read_timeout = timeout_seconds or settings.LLM_TIMEOUT
        connect = connect_timeout or settings.LLM_CONNECT_TIMEOUT
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key

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
            provider="anthropic",
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
    ) -> AnthropicLLM:
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            provider_name="anthropic",
        )

    def capabilities(self) -> ModelCapabilities:
        return self._caps

    def _messages_payload(
        self,
        request: ChatCompletionRequest,
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        messages: list[dict[str, str]] = []
        for message in request.messages:
            if message.role == "system":
                if message.content:
                    system_parts.append(message.content)
                continue
            role = "assistant" if message.role == "assistant" else "user"
            messages.append({"role": role, "content": message.content})
        if not messages:
            messages = [{"role": "user", "content": "."}]

        payload: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max(1, request.max_tokens),
            "messages": messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if self._caps.supports_temperature:
            payload["temperature"] = request.temperature
        return payload

    async def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        payload = self._messages_payload(request)
        try:
            response = await self.client.post("/messages", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTransientError(f"Anthropic timeout: {exc}") from exc
        except httpx.TransportError as exc:
            raise LLMTransientError(f"Anthropic transport error: {exc}") from exc

        if response.status_code in {401, 403}:
            raise LLMPermanentError(
                f"Anthropic auth failed ({response.status_code})"
            )
        if response.status_code in {400, 404, 422}:
            raise LLMPermanentError(
                f"Anthropic bad request ({response.status_code}): "
                f"{response.text[:200]}"
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise LLMTransientError(
                f"Anthropic transient HTTP {response.status_code}"
            )
        if response.status_code >= 400:
            raise LLMPermanentError(
                f"Anthropic HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMPermanentError("Malformed JSON from Anthropic") from exc

        content = _extract_text(data)
        if not content:
            raise LLMPermanentError("Empty content from Anthropic")

        usage = data.get("usage") or {}
        prompt_tokens = usage.get("input_tokens")
        completion_tokens = usage.get("output_tokens")
        total_tokens = None
        if prompt_tokens is not None or completion_tokens is not None:
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        return ChatCompletionResponse(
            content=strip_reasoning_output(content),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    async def generate_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        async for chunk in super().generate_stream(request):
            yield chunk

    async def aclose(self) -> None:
        await self.client.aclose()


def _extract_text(data: dict[str, Any]) -> str:
    blocks = data.get("content") or []
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if text:
                parts.append(str(text))
    return "".join(parts).strip()
