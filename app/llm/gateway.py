from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import settings
from app.llm.base import BaseLLM
from app.llm.delta import StreamDelta, coerce_delta
from app.llm.errors import (
    LLMAllProvidersFailed,
    LLMCancelledError,
    LLMError,
    LLMPermanentError,
    LLMTransientError,
)
from app.llm.model_capabilities import ModelCapabilities
from app.schemas.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GatewayResult:
    response: ChatCompletionResponse
    provider: str
    model: str
    retry_count: int = 0
    fallback_used: bool = False
    latency_ms: float = 0.0
    error_type: str | None = None


@dataclass(slots=True)
class GatewayStats:
    retry_count: int = 0
    fallback_used: bool = False
    provider: str | None = None
    model: str | None = None
    last_error_type: str | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)


class LLMGateway(BaseLLM):
    """
    Provider-agnostic LLM execution with timeout, retry, and fallback.

    Application code should depend on LLMGateway (or BaseLLM), never on
    a concrete Ollama/OpenAI client.
    """

    def __init__(
        self,
        primary: BaseLLM,
        *,
        fallback: BaseLLM | None = None,
        max_retries: int | None = None,
        retry_base_delay: float | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._max_retries = (
            settings.LLM_MAX_RETRIES if max_retries is None else max_retries
        )
        self._retry_base_delay = (
            settings.LLM_RETRY_BASE_DELAY_SECONDS
            if retry_base_delay is None
            else retry_base_delay
        )
        self._timeout = float(
            settings.LLM_TIMEOUT if timeout_seconds is None else timeout_seconds
        )
        self.provider_name = getattr(primary, "provider_name", "gateway")
        self.model_name = getattr(primary, "model_name", settings.CHAT_MODEL)
        self.last_stats = GatewayStats()

    def capabilities(self) -> ModelCapabilities | None:
        caps = self._primary.capabilities()
        if caps is not None:
            return caps
        if self._fallback is not None:
            return self._fallback.capabilities()
        return None

    async def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        result = await self.generate_with_meta(request)
        return result.response

    async def generate_with_meta(
        self,
        request: ChatCompletionRequest,
    ) -> GatewayResult:
        started = asyncio.get_event_loop().time()
        stats = GatewayStats()
        providers: list[tuple[str, BaseLLM]] = [
            ("primary", self._primary),
        ]
        if self._fallback is not None:
            providers.append(("fallback", self._fallback))

        errors: list[str] = []
        for index, (label, provider) in enumerate(providers):
            if index > 0:
                stats.fallback_used = True
            try:
                response, retries = await self._call_with_retries(
                    provider,
                    request,
                    stats=stats,
                    label=label,
                )
                elapsed = (asyncio.get_event_loop().time() - started) * 1000
                stats.provider = getattr(provider, "provider_name", label)
                stats.model = getattr(provider, "model_name", self.model_name)
                stats.retry_count = retries
                self.last_stats = stats
                return GatewayResult(
                    response=response,
                    provider=stats.provider or label,
                    model=stats.model or self.model_name,
                    retry_count=retries,
                    fallback_used=stats.fallback_used,
                    latency_ms=elapsed,
                )
            except LLMCancelledError:
                raise
            except LLMPermanentError as exc:
                errors.append(f"{label}: {exc}")
                stats.last_error_type = "permanent"
                # Permanent on primary → try fallback once; do not retry same provider.
                continue
            except LLMTransientError as exc:
                errors.append(f"{label}: {exc}")
                stats.last_error_type = "transient"
                continue
            except LLMError as exc:
                errors.append(f"{label}: {exc}")
                stats.last_error_type = "error"
                continue

        self.last_stats = stats
        detail = "; ".join(errors) or "unknown provider failure"
        raise LLMAllProvidersFailed(
            "The assistant is temporarily unavailable. "
            f"All LLM providers failed ({detail})."
        )

    async def generate_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str | StreamDelta]:
        # Streaming uses primary only for now; retries remain non-stream.
        try:
            async for piece in self._primary.generate_stream(request):
                delta = coerce_delta(piece)
                if delta.text:
                    yield delta
        except asyncio.CancelledError as exc:
            raise LLMCancelledError() from exc
        except LLMError:
            if self._fallback is None:
                raise
            self.last_stats.fallback_used = True
            async for piece in self._fallback.generate_stream(request):
                delta = coerce_delta(piece)
                if delta.text:
                    yield delta

    async def aclose(self) -> None:
        await self._primary.aclose()
        if self._fallback is not None:
            await self._fallback.aclose()

    async def _call_with_retries(
        self,
        provider: BaseLLM,
        request: ChatCompletionRequest,
        *,
        stats: GatewayStats,
        label: str,
    ) -> tuple[ChatCompletionResponse, int]:
        attempts = 0
        last_error: Exception | None = None
        max_attempts = max(1, self._max_retries + 1)

        while attempts < max_attempts:
            attempts += 1
            try:
                response = await asyncio.wait_for(
                    provider.generate(request),
                    timeout=self._timeout,
                )
                if response is None or response.content is None:
                    raise LLMPermanentError("Provider returned empty response")
                # Strip already applied in providers; still reject blank.
                if not str(response.content).strip():
                    raise LLMPermanentError("Provider returned blank content")
                stats.attempts.append(
                    {
                        "provider": label,
                        "attempt": attempts,
                        "ok": True,
                    }
                )
                return response, attempts - 1
            except asyncio.CancelledError as exc:
                raise LLMCancelledError() from exc
            except asyncio.TimeoutError as exc:
                last_error = LLMTransientError(
                    f"{label} exceeded timeout ({self._timeout}s)"
                )
                stats.attempts.append(
                    {
                        "provider": label,
                        "attempt": attempts,
                        "ok": False,
                        "error": "timeout",
                    }
                )
                # A local R1 call can already take tens of minutes; retrying
                # the same timeout would double the wait then still 503.
                break
            except httpx.TimeoutException as exc:
                last_error = LLMTransientError(f"{label} HTTP timeout: {exc}")
                stats.attempts.append(
                    {
                        "provider": label,
                        "attempt": attempts,
                        "ok": False,
                        "error": "http_timeout",
                    }
                )
                break
            except httpx.TransportError as exc:
                last_error = LLMTransientError(f"{label} transport: {exc}")
                stats.attempts.append(
                    {
                        "provider": label,
                        "attempt": attempts,
                        "ok": False,
                        "error": "transport",
                    }
                )
            except LLMPermanentError as exc:
                stats.attempts.append(
                    {
                        "provider": label,
                        "attempt": attempts,
                        "ok": False,
                        "error": "permanent",
                    }
                )
                raise
            except LLMTransientError as exc:
                last_error = exc
                stats.attempts.append(
                    {
                        "provider": label,
                        "attempt": attempts,
                        "ok": False,
                        "error": "transient",
                    }
                )
            except LLMError as exc:
                if exc.permanent:
                    raise LLMPermanentError(str(exc)) from exc
                last_error = exc
            except Exception as exc:  # noqa: BLE001 — classify unknown
                # Treat unexpected as transient once, then surface.
                last_error = LLMTransientError(f"{label} unexpected: {exc}")
                stats.attempts.append(
                    {
                        "provider": label,
                        "attempt": attempts,
                        "ok": False,
                        "error": type(exc).__name__,
                    }
                )
                logger.exception("Unexpected LLM provider error label=%s", label)

            if attempts >= max_attempts:
                break
            delay = self._retry_base_delay * (2 ** (attempts - 1))
            delay *= 0.8 + random.random() * 0.4  # jitter
            logger.warning(
                "LLM retry provider=%s attempt=%s/%s delay=%.2fs error=%s",
                label,
                attempts,
                max_attempts,
                delay,
                last_error,
            )
            await asyncio.sleep(delay)

        if last_error is None:
            raise LLMTransientError(f"{label} failed with no error detail")
        if isinstance(last_error, LLMError):
            raise last_error
        raise LLMTransientError(str(last_error))
