from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.llm.base import BaseLLM
from app.llm.errors import (
    LLMAllProvidersFailed,
    LLMPermanentError,
    LLMTransientError,
)
from app.llm.gateway import LLMGateway
from app.schemas.llm import ChatCompletionRequest, ChatCompletionResponse, ChatMessage


class FlakyLLM(BaseLLM):
    def __init__(
        self,
        *,
        fail_times: int = 0,
        error: Exception | None = None,
        content: str = "ok",
        provider_name: str = "flaky",
        model_name: str = "test-model",
    ) -> None:
        self.fail_times = fail_times
        self.calls = 0
        self.error = error or LLMTransientError("transient")
        self.content = content
        self.provider_name = provider_name
        self.model_name = model_name

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.error
        return ChatCompletionResponse(
            content=self.content,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hello")],
        max_tokens=64,
    )


@pytest.mark.asyncio
async def test_gateway_retries_transient_then_succeeds():
    primary = FlakyLLM(fail_times=1)
    gateway = LLMGateway(
        primary,
        max_retries=2,
        retry_base_delay=0.01,
        timeout_seconds=5,
    )
    result = await gateway.generate_with_meta(_request())
    assert result.response.content == "ok"
    assert result.retry_count == 1
    assert primary.calls == 2
    assert result.fallback_used is False


@pytest.mark.asyncio
async def test_gateway_does_not_retry_permanent_on_same_provider():
    primary = FlakyLLM(
        fail_times=5,
        error=LLMPermanentError("bad auth"),
    )
    fallback = FlakyLLM(content="fallback-answer", provider_name="fallback")
    gateway = LLMGateway(
        primary,
        fallback=fallback,
        max_retries=3,
        retry_base_delay=0.01,
        timeout_seconds=5,
    )
    result = await gateway.generate_with_meta(_request())
    assert result.response.content == "fallback-answer"
    assert result.fallback_used is True
    assert primary.calls == 1


@pytest.mark.asyncio
async def test_gateway_all_providers_fail_raises_clean_error():
    primary = FlakyLLM(fail_times=5, error=LLMTransientError("down"))
    fallback = FlakyLLM(
        fail_times=5,
        error=LLMTransientError("also down"),
        provider_name="fallback",
    )
    gateway = LLMGateway(
        primary,
        fallback=fallback,
        max_retries=1,
        retry_base_delay=0.01,
        timeout_seconds=5,
    )
    with pytest.raises(LLMAllProvidersFailed):
        await gateway.generate(_request())


@pytest.mark.asyncio
async def test_gateway_timeout_is_transient():
    class SlowLLM(BaseLLM):
        provider_name = "slow"
        model_name = "slow-model"

        async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
            await asyncio.sleep(0.2)
            return ChatCompletionResponse(content="late")

    gateway = LLMGateway(
        SlowLLM(),
        max_retries=0,
        retry_base_delay=0.01,
        timeout_seconds=0.05,
    )
    with pytest.raises(LLMAllProvidersFailed):
        await gateway.generate(_request())


@pytest.mark.asyncio
async def test_generate_stream_default_yields_full_answer():
    primary = FlakyLLM(content="final answer only")
    gateway = LLMGateway(primary, max_retries=0, timeout_seconds=5)
    pieces: list[str] = []
    async for piece in gateway.generate_stream(_request()):
        pieces.append(piece.text if hasattr(piece, "text") else piece)
    assert "".join(pieces) == "final answer only"
