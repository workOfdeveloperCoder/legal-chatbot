from __future__ import annotations

from app.llm.execution_profile import ExecutionProfile
from app.llm.model_capabilities import ModelCapabilities
from app.llm.token_counter import FixedTokenCounter
from app.schemas.chat import ChatResponse
from app.services.token_usage_service import (
    TokenUsageService,
    apply_provider_usage,
    token_usage_from_budget,
)


class _StubLLM:
    provider_name = "openai"
    model_name = "gpt-4o-mini"

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model_name="gpt-4o-mini",
            provider="openai",
            context_window=128000,
            max_output_tokens=16384,
            tokenizer_id="test_whitespace",
            input_price_per_1m=0.15,
            output_price_per_1m=0.60,
            returns_usage=True,
        )


def test_status_exposes_cursor_style_limits():
    service = TokenUsageService(_StubLLM())
    service.counter = FixedTokenCounter()
    status = service.status()
    assert status.model == "gpt-4o-mini"
    assert status.provider == "openai"
    assert status.online is True
    assert status.context_window == 128000
    assert status.available_input_tokens < status.context_window
    assert status.available_input_tokens > 0
    assert status.warning_percent == 80.0


def test_count_tokens_tracks_remaining_against_window():
    service = TokenUsageService(_StubLLM())
    service.counter = FixedTokenCounter()
    counted = service.count(
        "section 54-C clog on discretion",
        include_system_prompt=False,
    )
    assert counted.tokens == 5
    assert counted.context_window == 128000
    assert counted.remaining_tokens == counted.available_input_tokens - 5
    assert counted.over_limit is False
    assert counted.usage_percent < 1


def test_count_includes_system_prompt_when_requested():
    service = TokenUsageService(_StubLLM())
    service.counter = FixedTokenCounter()
    without_system = service.count("hello world", include_system_prompt=False)
    with_system = service.count("hello world", include_system_prompt=True)
    assert with_system.tokens > without_system.tokens
    profile = ExecutionProfile.for_capabilities(_StubLLM().capabilities())
    assert with_system.tokens == without_system.tokens + len(
        profile.system_prompt.split()
    )


def test_apply_provider_usage_overwrites_estimate():
    budget = {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "context_window": 128000,
        "input_tokens": 900,
        "output_tokens": 4096,
        "total_tokens": 4996,
        "available_input_tokens": 120000,
        "usage_source": "estimate",
    }
    updated = apply_provider_usage(
        budget,
        prompt_tokens=1200,
        completion_tokens=80,
        total_tokens=1280,
        capabilities=_StubLLM().capabilities(),
    )
    assert updated is not None
    assert updated["input_tokens"] == 1200
    assert updated["output_tokens"] == 80
    assert updated["total_tokens"] == 1280
    assert updated["usage_source"] == "provider"
    assert updated["remaining_input_tokens"] == 120000 - 1200
    assert updated["usage_percent"] == round(100.0 * 1280 / 128000, 2)
    assert updated["estimated_cost"] is not None


def test_token_usage_view_and_chat_response_field():
    usage = token_usage_from_budget(
        {
            "model": "gpt-4o-mini",
            "provider": "openai",
            "context_window": 128000,
            "input_tokens": 100000,
            "output_tokens": 200,
            "total_tokens": 100200,
            "available_input_tokens": 110000,
            "remaining_input_tokens": 10000,
            "usage_percent": 78.28,
            "tokenizer_id": "o200k_base",
            "tokenizer_exact": True,
            "usage_source": "provider",
            "system_tokens": 800,
            "query_tokens": 40,
        },
        capabilities=_StubLLM().capabilities(),
    )
    assert usage is not None
    assert usage.context_window == 128000
    assert usage.remaining_input_tokens == 10000
    assert usage.breakdown.query_tokens == 40
    from uuid import uuid4

    response = ChatResponse(
        conversation_id=uuid4(),
        response="Section 54-C can clog discretion.",
        token_usage=usage,
    )
    dumped = response.model_dump()
    assert dumped["token_usage"]["model"] == "gpt-4o-mini"
    assert dumped["token_usage"]["context_window"] == 128000
