from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator

import httpx

from app.core.config import settings
from app.llm.base import BaseLLM
from app.llm.delta import StreamDelta
from app.llm.errors import LLMPermanentError, LLMTransientError
from app.llm.model_capabilities import ModelCapabilities, ModelCapabilityRegistry
from app.llm.registry import LLMAdapterRegistry
from app.rag.reasoning_cleanup import (
    ReasoningStreamFilter,
    strip_reasoning_output,
)
from app.schemas.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

logger = logging.getLogger(__name__)

# DeepSeek R1 spends many tokens on hidden "thinking" before content.
# Too-low num_predict yields empty message.content; RAG then falls back
# to retrieved passages instead of raising a hard 503.
_R1_MIN_PREDICT = 12288
_R1_RETRY_PREDICT = 16384

_THINKING_ANSWER_MARKERS = (
    re.compile(
        r"(?is)(?:final answer|answer|conclusion|to sum up|in summary)\s*[:\-]\s*(.+)$"
    ),
)


@LLMAdapterRegistry.register(
    "ollama",
    online=False,
    default_url="http://localhost:11434",
)
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

    @classmethod
    def connect(
        cls,
        *,
        provider: str,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_seconds: int,
    ) -> OllamaLLM:
        return cls(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
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
            if predict < _R1_RETRY_PREDICT:
                logger.warning(
                    "Ollama R1 returned empty content (done_reason=%s, "
                    "predict=%s); retrying with num_predict=%s",
                    done_reason,
                    predict,
                    _R1_RETRY_PREDICT,
                )
                data = await self._chat(request, num_predict=_R1_RETRY_PREDICT)
                parsed = self._parse_chat_payload(data)
                thinking = (
                    ((data.get("message") or {}).get("thinking") or "").strip()
                    or thinking
                )
                done_reason = data.get("done_reason") or done_reason

            if not parsed.content.strip():
                salvaged = _salvage_answer_from_thinking(thinking)
                if salvaged:
                    logger.warning(
                        "Ollama R1 empty content (done_reason=%s); "
                        "salvaged %s chars from thinking",
                        done_reason,
                        len(salvaged),
                    )
                    return ChatCompletionResponse(
                        content=salvaged,
                        prompt_tokens=parsed.prompt_tokens,
                        completion_tokens=parsed.completion_tokens,
                        total_tokens=parsed.total_tokens,
                    )

                # Soft-fail for R1 so RAG can show retrieved passages instead
                # of a hard 503. Non-R1 models still raise.
                if self._is_r1_model():
                    logger.warning(
                        "Ollama R1 empty content after retries "
                        "(done_reason=%s, predict=%s); returning blank "
                        "for RAG passage fallback",
                        done_reason,
                        max(predict, _R1_RETRY_PREDICT),
                    )
                    return ChatCompletionResponse(
                        content="",
                        prompt_tokens=parsed.prompt_tokens,
                        completion_tokens=parsed.completion_tokens,
                        total_tokens=parsed.total_tokens,
                    )
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

    def _is_r1_model(self) -> bool:
        name = (self.model_name or "").lower()
        return "deepseek" in name and "r1" in name

    def _effective_num_predict(self, requested: int) -> int:
        value = max(int(requested or 0), 256)
        if self._is_r1_model():
            return max(value, _R1_MIN_PREDICT)
        return value

    async def generate_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str | StreamDelta]:
        if not settings.LLM_ENABLE_STREAMING:
            async for chunk in super().generate_stream(request):
                yield chunk
            return

        predict = self._effective_num_predict(request.max_tokens)
        filter_ = ReasoningStreamFilter()
        yielded_visible = False
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
                    native_thinking = message.get("thinking")
                    if native_thinking:
                        yield StreamDelta(text=str(native_thinking), kind="thinking")
                    piece = message.get("content")
                    if piece:
                        visible, thinking = filter_.feed_parts(piece)
                        if thinking:
                            yield StreamDelta(text=thinking, kind="thinking")
                        if visible and visible.strip():
                            yielded_visible = True
                            yield StreamDelta(text=visible, kind="token")
                    if event.get("done"):
                        break
        except httpx.TimeoutException as exc:
            raise LLMTransientError(f"Ollama stream timeout: {exc}") from exc
        except httpx.TransportError as exc:
            raise LLMTransientError(f"Ollama stream transport: {exc}") from exc

        leftover_visible, leftover_thinking = filter_.finish_parts()
        if leftover_thinking:
            yield StreamDelta(text=leftover_thinking, kind="thinking")
        if leftover_visible and leftover_visible.strip():
            yielded_visible = True
            yield StreamDelta(text=leftover_visible, kind="token")
        if yielded_visible:
            return
        logger.warning(
            "Ollama stream returned no visible content (model=%s); "
            "falling back to non-stream generate()",
            self.model_name,
        )
        parsed = await self.generate(request)
        if parsed.content.strip():
            yield StreamDelta(text=parsed.content, kind="token")

    async def aclose(self) -> None:
        await self.client.aclose()


def _salvage_answer_from_thinking(thinking: str) -> str:
    """Best-effort visible answer when R1 never left the thinking channel."""
    text = (thinking or "").strip()
    if len(text) < 40:
        return ""
    for pattern in _THINKING_ANSWER_MARKERS:
        match = pattern.search(text)
        if not match:
            continue
        candidate = " ".join(match.group(1).split()).strip()
        if len(candidate) >= 40:
            return candidate[:2000]
    paragraphs = [
        " ".join(part.split()).strip()
        for part in re.split(r"\n\s*\n", text)
        if part.strip()
    ]
    for paragraph in reversed(paragraphs):
        if len(paragraph) >= 60 and not paragraph.lower().startswith(
            ("i need", "i should", "let me", "the user", "wait,", "hmm")
        ):
            return paragraph[:2000]
    return ""
