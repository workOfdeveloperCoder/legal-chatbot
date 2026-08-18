from __future__ import annotations

import logging
from collections.abc import AsyncIterator

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

# DeepSeek R1 spends many tokens on hidden "thinking" before content.
# Too-low num_predict yields empty message.content and a 503 upstream.
_R1_MIN_PREDICT = 4096
_R1_RETRY_PREDICT = 8192


class OllamaLLM(BaseLLM):
    """Local Ollama / DeepSeek adapter."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int | None = None,
        connect_timeout: int | None = None,
    ) -> None:
        self.provider_name = "ollama"
        self.model_name = model or settings.CHAT_MODEL
        read_timeout = timeout_seconds or settings.LLM_TIMEOUT
        connect = connect_timeout or settings.LLM_CONNECT_TIMEOUT
        self.client = httpx.AsyncClient(
            base_url=base_url or settings.LLM_URL,
            timeout=httpx.Timeout(
                connect=connect,
                read=max(read_timeout, 60),
                write=connect,
                pool=connect,
            ),
        )
        self._caps = ModelCapabilityRegistry.resolve(
            provider="ollama",
            model_name=self.model_name,
        )

    def capabilities(self) -> ModelCapabilities:
        return self._caps

    async def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        predict = self._effective_num_predict(request.max_tokens)
        data = await self._chat(request, num_predict=predict)
        parsed = self._parse_chat_payload(data)

        # R1 often exhausts the budget inside thinking and returns blank content.
        if not parsed.content.strip():
            thinking = ((data.get("message") or {}).get("thinking") or "").strip()
            done_reason = data.get("done_reason")
            if thinking and predict < _R1_RETRY_PREDICT:
                logger.warning(
                    "Ollama R1 returned empty content (done_reason=%s, "
                    "predict=%s); retrying with num_predict=%s",
                    done_reason,
                    predict,
                    _R1_RETRY_PREDICT,
                )
                data = await self._chat(request, num_predict=_R1_RETRY_PREDICT)
                parsed = self._parse_chat_payload(data)

            if not parsed.content.strip():
                # Transient so gateway may retry / fallback — never invent an answer.
                raise LLMTransientError(
                    "Ollama returned empty final content "
                    f"(model={self.model_name}, done_reason={done_reason}, "
                    "likely spent all tokens on reasoning). "
                    "Increase TOKEN_RESERVED_OUTPUT_TOKENS / LLM_TIMEOUT."
                )

        return parsed

    async def _chat(
        self,
        request: ChatCompletionRequest,
        *,
        num_predict: int,
    ) -> dict:
        try:
            response = await self.client.post(
                "/api/chat",
                json={
                    "model": self.model_name,
                    "stream": False,
                    "messages": [
                        message.model_dump()
                        for message in request.messages
                    ],
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": num_predict,
                    },
                },
            )
        except httpx.TimeoutException as exc:
            raise LLMTransientError(f"Ollama timeout: {exc}") from exc
        except httpx.TransportError as exc:
            raise LLMTransientError(f"Ollama transport error: {exc}") from exc

        if response.status_code in {401, 403}:
            raise LLMPermanentError(f"Ollama auth failed ({response.status_code})")
        if response.status_code in {400, 404}:
            raise LLMPermanentError(
                f"Ollama bad request ({response.status_code}): {response.text[:200]}"
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise LLMTransientError(f"Ollama transient HTTP {response.status_code}")
        if response.status_code >= 400:
            raise LLMPermanentError(
                f"Ollama HTTP {response.status_code}: {response.text[:200]}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise LLMPermanentError("Malformed JSON from Ollama") from exc

    def _parse_chat_payload(self, data: dict) -> ChatCompletionResponse:
        message = data.get("message")
        if not isinstance(message, dict):
            raise LLMPermanentError("Malformed Ollama response: missing message")

        # Newer Ollama R1 builds put CoT in message.thinking and the answer
        # in message.content. Older builds embed <think> inside content.
        raw_content = message.get("content")
        if raw_content is None:
            raw_content = ""

        cleaned = strip_reasoning_output(str(raw_content))
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        return ChatCompletionResponse(
            content=cleaned,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=(
                (prompt_tokens or 0) + (completion_tokens or 0)
                if prompt_tokens is not None or completion_tokens is not None
                else None
            ),
        )

    def _effective_num_predict(self, requested: int) -> int:
        value = max(int(requested or 0), 256)
        if "deepseek-r1" in (self.model_name or "").lower():
            return max(value, _R1_MIN_PREDICT)
        return value

    async def generate_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        if not settings.LLM_ENABLE_STREAMING:
            async for chunk in super().generate_stream(request):
                yield chunk
            return

        predict = self._effective_num_predict(request.max_tokens)
        try:
            async with self.client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": self.model_name,
                    "stream": True,
                    "messages": [m.model_dump() for m in request.messages],
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": predict,
                    },
                },
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise LLMTransientError(
                        f"Ollama stream HTTP {response.status_code}: {body[:200]!r}"
                    )
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    import json

                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    message = event.get("message") or {}
                    piece = message.get("content")
                    if piece:
                        cleaned = strip_reasoning_output(piece)
                        if cleaned:
                            yield cleaned
                    if event.get("done"):
                        break
        except httpx.TimeoutException as exc:
            raise LLMTransientError(f"Ollama stream timeout: {exc}") from exc
        except httpx.TransportError as exc:
            raise LLMTransientError(f"Ollama stream transport: {exc}") from exc

    async def aclose(self) -> None:
        await self.client.aclose()
