from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.config import settings
from app.llm.base import BaseLLM
from app.llm.delta import StreamDelta
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


def _parts_to_text(value: object) -> str | None:
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, list):
        bits: list[str] = []
        for item in value:
            if isinstance(item, str):
                bits.append(item)
            elif isinstance(item, dict):
                bits.append(
                    str(item.get("text") or item.get("content") or "")
                )
        joined = "".join(bits)
        return joined if joined.strip() else None
    return None


def _choice_visible_text(choice: dict[str, Any]) -> str | None:
    message = choice.get("message") or {}
    return _parts_to_text(message.get("content")) or _parts_to_text(
        message.get("reasoning") or message.get("reasoning_content")
    )


def _sse_error_status(event: dict[str, Any]) -> tuple[int, str] | None:
    error = event.get("error")
    if not error:
        return None
    if isinstance(error, dict):
        message = str(error.get("message") or error)
        code = error.get("code") or error.get("status") or 400
    else:
        message = str(error)
        code = 400
    try:
        status_code = int(code)
    except (TypeError, ValueError):
        status_code = 400
    return status_code, message


@LLMAdapterRegistry.register(
    "openai",
    "openai_compatible",
    "deepseek",
    "groq",
    "gemini",
    "mistral",
    "together",
    "fireworks",
    online=True,
    default_urls={
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "groq": "https://api.groq.com/openai/v1",
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
    (DeepSeek, Groq, Gemini OpenAI mode, Mistral, Together).
    OpenRouter has its own subclass. For a different wire protocol, add a
    new BaseLLM subclass instead.
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
        extra_headers: dict[str, str] | None = None,
        send_stream_options: bool = True,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model or settings.CHAT_MODEL
        self._api_key = (
            api_key
            if api_key is not None
            else settings.api_key_for_provider(provider_name)
        )
        self._send_stream_options = send_stream_options
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
        if extra_headers:
            headers.update(extra_headers)

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
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if stream and self._send_stream_options:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _raise_for_status(self, status_code: int, body: str) -> None:
        snippet = (body or "")[:200]
        label = self.provider_name
        if status_code in {401, 403}:
            raise LLMPermanentError(f"{label} auth failed ({status_code})")
        if status_code in {400, 404, 422}:
            raise LLMPermanentError(
                f"{label} bad request ({status_code}): {snippet}"
            )
        if status_code == 429 or status_code >= 500:
            raise LLMTransientError(
                f"{label} transient HTTP {status_code}"
            )
        if status_code >= 400:
            raise LLMPermanentError(
                f"{label} HTTP {status_code}: {snippet}"
            )

    async def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        payload = self._completion_payload(request, stream=False)
        try:
            response = await self.client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTransientError(
                f"{self.provider_name} timeout: {exc}"
            ) from exc
        except httpx.TransportError as exc:
            raise LLMTransientError(
                f"{self.provider_name} transport error contacting "
                f"{self._base_url}: {exc}"
            ) from exc

        self._raise_for_status(response.status_code, response.text)

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMPermanentError(
                f"Malformed JSON from {self.provider_name}"
            ) from exc

        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMPermanentError(
                f"Malformed {self.provider_name} response: missing choices/message"
            ) from exc

        content = _choice_visible_text(choice)
        if not content:
            raise LLMPermanentError(
                f"Empty content from {self.provider_name}"
            )

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
    ) -> AsyncIterator[str | StreamDelta]:
        if not settings.LLM_ENABLE_STREAMING:
            async for chunk in super().generate_stream(request):
                yield chunk
            return

        payload = self._completion_payload(request, stream=True)
        yielded = False
        try:
            async with self.client.stream(
                "POST",
                "/chat/completions",
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    text = body.decode("utf-8", errors="replace")
                    self._raise_for_status(response.status_code, text)
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except ValueError:
                        continue
                    sse_error = _sse_error_status(event)
                    if sse_error is not None:
                        self._raise_for_status(sse_error[0], sse_error[1])
                    try:
                        delta = event["choices"][0].get("delta") or {}
                    except (KeyError, IndexError, TypeError):
                        continue
                    reasoning = _parts_to_text(
                        delta.get("reasoning")
                        or delta.get("reasoning_content")
                    )
                    if reasoning:
                        yield StreamDelta(text=reasoning, kind="thinking")
                    piece = _parts_to_text(delta.get("content"))
                    if piece:
                        yielded = True
                        yield strip_reasoning_output(piece)
        except httpx.TimeoutException as exc:
            raise LLMTransientError(f"Stream timeout: {exc}") from exc
        except httpx.TransportError as exc:
            raise LLMTransientError(
                f"Stream transport error contacting {self._base_url}: {exc}"
            ) from exc

        if yielded:
            return
        logger.warning(
            "%s stream returned no visible content (model=%s); "
            "falling back to non-stream generate()",
            self.provider_name,
            self.model_name,
        )
        parsed = await self.generate(request)
        if parsed.content.strip():
            yield parsed.content

    async def aclose(self) -> None:
        await self.client.aclose()
