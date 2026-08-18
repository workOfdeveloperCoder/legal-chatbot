from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.config import settings
from app.llm.base import BaseLLM
from app.llm.errors import LLMPermanentError, LLMTransientError
from app.llm.model_capabilities import ModelCapabilities, ModelCapabilityRegistry
from app.rag.reasoning_cleanup import strip_reasoning_output
from app.schemas.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM(BaseLLM):
    """
    OpenAI Chat Completions-compatible adapter.

    Works with OpenAI and any OpenAI-compatible online API.
    Credentials come from config/env — never hard-coded.
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
        self._api_key = api_key if api_key is not None else settings.LLM_API_KEY
        base = (base_url or settings.LLM_URL).rstrip("/")
        # Accept either root or /v1
        if base.endswith("/v1"):
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

    def capabilities(self) -> ModelCapabilities:
        return self._caps

    async def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
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

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
        }
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
