from __future__ import annotations

import pytest

from app.llm.anthropic import AnthropicLLM
from app.llm.base import BaseLLM
from app.llm.execution_profile import ExecutionProfile
from app.llm.factory import LLMFactory
from app.llm.model_capabilities import ModelCapabilityRegistry
from app.llm.openai_compatible import OpenAICompatibleLLM
from app.llm.provider_config import resolve_chat_model, resolve_llm_base_url
from app.llm.registry import LLMAdapterRegistry
from app.rag.prompt_builder import PromptBuilder
from app.rag.prompts import LEGAL_SYSTEM_PROMPT, LEGAL_SYSTEM_PROMPT_COMPACT
from app.rag.query_complexity import QueryComplexity
from app.rag.reasoning_pipeline import ReasoningPipeline
from app.rag.token_budget.manager import TokenBudgetManager
from app.schemas.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)


def test_factory_builds_local_ollama_adapter():
    from app.llm.ollama import OllamaLLM

    llm = LLMFactory.create_raw(
        provider="ollama",
        base_url="http://127.0.0.1:11434",
        model="deepseek-r1:32b",
    )
    assert isinstance(llm, OllamaLLM)
    assert llm.model_name == "deepseek-r1:32b"
    assert "11434" in str(llm.client.base_url)


def test_factory_builds_anthropic_adapter():
    llm = LLMFactory.create_raw(
        provider="anthropic",
        base_url="http://localhost:11434",
        model="deepseek-r1:32b",
        api_key="sk-ant-test",
    )
    assert isinstance(llm, AnthropicLLM)
    assert llm.model_name == "claude-sonnet-4-5"
    assert str(llm.client.base_url).rstrip("/") == "https://api.anthropic.com/v1"


def test_gemini_openai_compat_url_is_not_double_suffixed():
    url = resolve_llm_base_url("gemini", "http://localhost:11434")
    assert url.endswith("/openai")
    llm = LLMFactory.create_raw(
        provider="gemini",
        base_url="http://localhost:11434",
        model="gemini-2.0-flash",
        api_key="test-key",
    )
    assert isinstance(llm, OpenAICompatibleLLM)
    assert str(llm.client.base_url).rstrip("/") == url


def test_anthropic_payload_uses_top_level_system():
    llm = AnthropicLLM(
        base_url="https://api.anthropic.com",
        api_key="sk-ant-test",
        model="claude-sonnet-4-5",
    )
    payload = llm._messages_payload(
        ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="Be concise."),
                ChatMessage(role="user", content="What is s.54-C?"),
            ],
            temperature=0.1,
            max_tokens=512,
        )
    )
    assert payload["system"] == "Be concise."
    assert payload["messages"] == [
        {"role": "user", "content": "What is s.54-C?"}
    ]
    assert payload["max_tokens"] == 512
    assert "model" in payload


def test_custom_adapter_registers_without_changing_factory():
    @LLMAdapterRegistry.register(
        "unit_test_vendor",
        online=True,
        default_url="https://llm.example.test/v1",
    )
    class UnitTestLLM(BaseLLM):
        def __init__(self, *, model: str, base_url: str, api_key: str | None) -> None:
            self.provider_name = "unit_test_vendor"
            self.model_name = model
            self.base_url = base_url
            self.api_key = api_key

        @classmethod
        def connect(
            cls,
            *,
            provider: str,
            base_url: str,
            model: str,
            api_key: str | None,
            timeout_seconds: int,
        ) -> UnitTestLLM:
            return cls(model=model, base_url=base_url, api_key=api_key)

        async def generate(
            self,
            request: ChatCompletionRequest,
        ) -> ChatCompletionResponse:
            return ChatCompletionResponse(content="ok")

    try:
        assert "unit_test_vendor" in LLMFactory.registered_providers()
        llm = LLMFactory.create_raw(
            provider="unit_test_vendor",
            base_url="http://localhost:11434",
            model="example-large",
            api_key="k",
        )
        assert isinstance(llm, UnitTestLLM)
        assert llm.base_url == "https://llm.example.test/v1"
        assert llm.model_name == "example-large"
    finally:
        LLMAdapterRegistry._adapters.pop("unit_test_vendor", None)
        LLMAdapterRegistry._online.pop("unit_test_vendor", None)
        LLMAdapterRegistry._default_urls.pop("unit_test_vendor", None)


def test_unknown_provider_lists_registered_adapters():
    with pytest.raises(ValueError, match="Registered adapters"):
        LLMFactory.create_raw(provider="not_a_real_vendor", api_key="k")


def test_compact_system_prompt_is_shorter_and_keeps_legal_rules():
    compact = LEGAL_SYSTEM_PROMPT_COMPACT
    full = LEGAL_SYSTEM_PROMPT
    assert len(compact.split()) < len(full.split()) * 0.45
    assert "LegalGPT" in compact
    assert "[Source N]" in compact
    assert "Punjabi" in compact
    assert "invent" in compact.lower() or "Never invent" in compact


def test_online_execution_profile_saves_tokens():
    caps = ModelCapabilityRegistry.resolve(
        provider="openai",
        model_name="gpt-4o-mini",
    )
    online = ExecutionProfile.online(caps)
    local = ExecutionProfile.local(caps)
    assert online.compact_prompts is True
    assert online.embed_system_in_user_prompt is False
    assert online.two_stage_reasoning is False
    assert online.reserved_output_tokens <= local.reserved_output_tokens
    assert online.max_legal_evidence_tokens < local.max_legal_evidence_tokens
    assert len(online.system_prompt) < len(local.system_prompt)


def test_prompt_builder_online_does_not_duplicate_system_prompt():
    builder = PromptBuilder()
    prompt = builder.build(
        question="What is section 54-C?",
        history=[],
        chunks=[],
        memories=[],
        include_system_in_user=False,
        compact=True,
    )
    assert "You are LegalGPT" not in prompt
    assert "Be token-efficient" in prompt
    assert "USER QUESTION" in prompt


@pytest.mark.asyncio
async def test_online_profile_skips_two_stage_reasoning():
    class RecordingLLM(BaseLLM):
        def __init__(self) -> None:
            self.calls = 0
            self.provider_name = "openai"
            self.model_name = "gpt-4o-mini"

        def capabilities(self):
            return ModelCapabilityRegistry.resolve(
                provider="openai",
                model_name="gpt-4o-mini",
            )

        async def generate(
            self,
            request: ChatCompletionRequest,
        ) -> ChatCompletionResponse:
            self.calls += 1
            return ChatCompletionResponse(
                content="Direct answer.",
                prompt_tokens=20,
                completion_tokens=8,
                total_tokens=28,
            )

    llm = RecordingLLM()
    profile = ExecutionProfile.online(llm.capabilities())
    result = await ReasoningPipeline(llm, profile=profile).generate(
        user_prompt="Analyse section 54-C on injunctions.",
        complexity=QueryComplexity.COMPLEX,
    )
    assert llm.calls == 1
    assert result.two_stage is False
    assert "Direct answer" in result.content


def test_online_token_budget_uses_tighter_caps():
    from app.llm.token_counter import FixedTokenCounter

    caps = ModelCapabilityRegistry.resolve(
        provider="openai",
        model_name="gpt-4o-mini",
    )
    profile = ExecutionProfile.online(caps)
    mgr = TokenBudgetManager(
        capabilities=caps,
        limits=profile.token_limits(caps),
        counter=FixedTokenCounter(),
    )
    packed = mgr.prepare(
        question="What is section 54-C?",
        history=[],
        chunks=[],
        memories=[],
        system_prompt=profile.system_prompt,
        duplicate_system_in_user=False,
    )
    assert packed.reserved_output_tokens == profile.reserved_output_tokens
    assert packed.metadata is not None
    assert packed.metadata.scaffolding_tokens == profile.scaffolding_overhead_tokens


def test_local_ollama_tag_maps_for_anthropic():
    assert resolve_chat_model("anthropic", "deepseek-r1:32b") == "claude-sonnet-4-5"
    assert resolve_chat_model("claude", "deepseek-r1:32b") == "claude-sonnet-4-5"


def test_r1_32b_reserves_thinking_headroom():
    from app.llm.ollama import OllamaLLM

    caps = ModelCapabilityRegistry.resolve(
        provider="ollama",
        model_name="deepseek-r1:32b",
    )
    assert caps.max_output_tokens >= 8192
    llm = OllamaLLM(
        base_url="http://127.0.0.1:11434",
        model="deepseek-r1:32b",
        timeout_seconds=5,
    )
    try:
        assert llm._effective_num_predict(512) >= 12288
    finally:
        pass


def test_salvage_answer_from_thinking():
    from app.llm.ollama import _salvage_answer_from_thinking

    thinking = (
        "I need to reason about this carefully.\n\n"
        "Final answer: Section 54-C of the Electricity Act, 1910 can "
        "restrict interim injunctions in certain electricity disputes."
    )
    salvaged = _salvage_answer_from_thinking(thinking)
    assert "Section 54-C" in salvaged
    assert "Electricity Act" in salvaged


def test_blank_llm_chunk_fallback_includes_passages():
    from types import SimpleNamespace

    from app.rag.rag_service import RAGService

    text = RAGService._blank_llm_chunk_fallback(
        question="What is Section 54-c?",
        chunks=[
            SimpleNamespace(
                filename="CLOG ON DISCRETION",
                text="Section 54-C operates as a clog on judicial discretion.",
            )
        ],
    )
    assert "Section 54-c" in text or "54-c" in text.lower()
    assert "CLOG ON DISCRETION" in text
    assert "Resources" in text or "resources" in text.lower()


def test_finalize_response_never_empty():
    from app.rag.rag_service import RAGService

    payload = RAGService._finalize_response({"answer": "[Source 1]"})
    assert payload["answer"].strip()
    payload2 = RAGService._finalize_response({"answer": ""})
    assert payload2["answer"].strip()
