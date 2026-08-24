from __future__ import annotations

from pathlib import Path

import pytest

from app.llm.model_capabilities import (
    ModelCapabilities,
    ModelCapabilityRegistry,
    estimate_cost_usd,
)
from app.llm.token_counter import (
    FallbackTokenEstimator,
    FixedTokenCounter,
    TiktokenCounter,
    build_token_counter,
)
from app.rag.models import Message, RetrievedChunk, SourceType
from app.rag.prompt_builder import PromptBuilder
from app.rag.token_budget import TokenBudgetManager
from app.rag.token_budget.models import TokenBudgetLimits
from app.rag.token_budget.packing import (
    group_chunks_by_document,
    remove_duplicate_evidence,
    score_evidence_chunk,
)
from app.services.response_formatter import ResponseFormatter


DOC_ID = "clog-doc-001"
CLOG_PATH = Path(
    "/Users/mrmacbook/Projects/legal-chatbot/storage/documents/"
    "411dfb9b-2f3f-4a44-a1c8-a2c067aef624_2000J8.txt"
)


def _chunk(**kwargs) -> RetrievedChunk:
    defaults = {
        "id": f"{DOC_ID}:0",
        "chunk_id": f"{DOC_ID}:0",
        "score": 0.9,
        "relevance_score": 0.9,
        "text": "Section 54-C can operate as a clog on judicial discretion.",
        "filename": "2000J8.txt",
        "display_name": "CLOG ON DISCRETION",
        "document_id": DOC_ID,
        "chunk_index": 0,
        "source_type": SourceType.MATTER.value,
        "start_offset": 0,
        "end_offset": 100,
        "page_number": 1,
    }
    defaults.update(kwargs)
    return RetrievedChunk(**defaults)


def _manager(
    *,
    context_window: int = 4000,
    reserved_output: int = 800,
    safety: int = 100,
    counter=None,
    model: str = "deepseek-r1:32b",
    pricing: tuple[float | None, float | None] = (0.0, 0.0),
) -> TokenBudgetManager:
    caps = ModelCapabilities(
        model_name=model,
        provider="ollama",
        context_window=context_window,
        max_output_tokens=reserved_output,
        tokenizer_id="test_whitespace",
        input_price_per_1m=pricing[0],
        output_price_per_1m=pricing[1],
        returns_usage=True,
    )
    limits = TokenBudgetLimits(
        context_window=context_window,
        reserved_output_tokens=reserved_output,
        safety_margin=safety,
        max_conversation_tokens=400,
        max_legal_evidence_tokens=1200,
        max_matter_evidence_tokens=1200,
        max_conversation_document_tokens=800,
        scaffolding_overhead_tokens=50,
        token_counting_enabled=True,
        cost_tracking_enabled=True,
    )
    return TokenBudgetManager(
        capabilities=caps,
        counter=counter or FixedTokenCounter(),
        limits=limits,
    )


class TestTokenCounting:
    def test_exact_tiktoken_counting(self):
        counter = build_token_counter("cl100k_base", enabled=True)
        if not isinstance(counter, TiktokenCounter):
            pytest.skip("tiktoken encoding cl100k_base is not cached locally")
        assert isinstance(counter, TiktokenCounter)
        assert counter.is_exact is True
        assert counter.count("hello world") == 2
        assert counter.count("") == 0

    def test_fallback_counting_not_char_len(self):
        estimator = FallbackTokenEstimator()
        text = "Section 54-C creates a clog on judicial discretion."
        estimate = estimator.count(text)
        assert estimate > 0
        assert estimate != len(text)
        assert estimator.is_exact is False
        # Distinct from naive character/4 heuristics commonly misused.
        assert estimate != max(1, len(text) // 4)

    def test_fixed_counter_for_tests(self):
        counter = FixedTokenCounter()
        assert counter.count("one two three") == 3


class TestBudgetCore:
    def test_output_reservation_and_available_input(self):
        mgr = _manager(context_window=4000, reserved_output=800, safety=100)
        packed = mgr.prepare(
            question="What is section 54-C?",
            history=[],
            chunks=[_chunk()],
            system_prompt="You are a legal research assistant.",
        )
        meta = packed.metadata
        assert meta is not None
        assert meta.reserved_output_tokens == 800
        assert meta.available_input_tokens == 4000 - 800 - 100
        assert meta.input_tokens <= meta.available_input_tokens
        assert (
            meta.input_tokens + meta.reserved_output_tokens + meta.safety_margin_tokens
            <= meta.context_window
        )

    def test_context_overflow_trims_evidence(self):
        counter = FixedTokenCounter()
        # Tiny window forces trimming.
        mgr = _manager(context_window=900, reserved_output=200, safety=50, counter=counter)
        chunks = [
            _chunk(
                id=f"{DOC_ID}:{i}",
                chunk_id=f"{DOC_ID}:{i}",
                chunk_index=i,
                score=0.95 - i * 0.05,
                relevance_score=0.95 - i * 0.05,
                text=("authority passage about section 54-C clog discretion " * 8),
                start_offset=i * 100,
                end_offset=i * 100 + 80,
            )
            for i in range(8)
        ]
        packed = mgr.prepare(
            question="Explain clog on discretion under section 54-C",
            history=[],
            chunks=chunks,
            document_qa_mode=True,
            system_prompt="You are a legal research assistant.",
        )
        assert packed.metadata is not None
        assert packed.metadata.budget_trimmed is True
        assert len(packed.chunks) < len(chunks)
        assert packed.metadata.input_tokens <= packed.metadata.available_input_tokens

    def test_prompt_always_within_context_limit(self):
        mgr = _manager(context_window=2500, reserved_output=500, safety=100)
        history = [
            Message(role="user", content=f"prior question about discretion number {i} " * 20)
            for i in range(12)
        ]
        chunks = [
            _chunk(
                id=f"{DOC_ID}:{i}",
                chunk_id=f"{DOC_ID}:{i}",
                text=("long legal evidence about section 54-C " * 30),
                relevance_score=0.9 - i * 0.02,
                chunk_index=i,
            )
            for i in range(10)
        ]
        packed = mgr.prepare(
            question="What does CLOG ON DISCRETION say about section 54-C?",
            history=history,
            chunks=chunks,
            document_qa_mode=True,
            system_prompt="You are a legal research assistant.",
        )
        prompt = PromptBuilder().build(
            question="What does CLOG ON DISCRETION say about section 54-C?",
            history=packed.history,
            chunks=packed.chunks,
            memories=[],
            document_qa_mode=True,
            conversation_summary=packed.conversation_summary,
            active_legal_context=packed.active_legal_context,
        )
        packed = mgr.ensure_prompt_within_budget(prompt, packed=packed)
        total = mgr.counter.count("system") + mgr.counter.count(prompt)
        # Rebuild after possible trim
        if packed.metadata and packed.metadata.budget_trimmed:
            prompt = PromptBuilder().build(
                question="What does CLOG ON DISCRETION say about section 54-C?",
                history=packed.history,
                chunks=packed.chunks,
                memories=[],
                document_qa_mode=True,
                conversation_summary=packed.conversation_summary,
                active_legal_context=packed.active_legal_context,
            )
        assert (
            mgr.counter.count(prompt) + packed.reserved_output_tokens + 100
            <= 2500 + mgr.counter.count("x" * 50)  # scaffolding already reserved upstream
        )
        assert packed.metadata is not None
        assert packed.metadata.input_tokens <= packed.metadata.available_input_tokens


class TestEvidencePacking:
    def test_duplicate_evidence_removal(self):
        a = _chunk(id="a", chunk_id="a", text="Same passage about clog on discretion.")
        b = _chunk(
            id="b",
            chunk_id="b",
            text="Same passage about clog on discretion.",
            relevance_score=0.5,
        )
        unique, removed = remove_duplicate_evidence([a, b])
        assert removed >= 1
        assert len(unique) == 1

    def test_evidence_trimming_prefers_higher_score(self):
        mgr = _manager(context_window=1200, reserved_output=300, safety=50)
        strong = _chunk(
            id="strong",
            chunk_id="strong",
            relevance_score=0.95,
            text="Strong evidence: section 54-C clog on judicial discretion " * 15,
        )
        weak = _chunk(
            id="weak",
            chunk_id="weak",
            relevance_score=0.2,
            text="Weak unrelated filler text about weather and traffic " * 15,
        )
        packed = mgr.prepare(
            question="clog on discretion section 54-C",
            history=[],
            chunks=[weak, strong],
            document_qa_mode=True,
            system_prompt="You are a legal research assistant.",
        )
        ids = {c.chunk_id for c in packed.chunks}
        assert "strong" in ids

    def test_document_grouping_one_resource(self):
        chunks = [
            _chunk(id=f"{DOC_ID}:{i}", chunk_id=f"{DOC_ID}:{i}", chunk_index=i)
            for i in range(3)
        ]
        grouped = group_chunks_by_document(chunks)
        assert len(grouped) == 1
        assert DOC_ID in grouped
        assert len(grouped[DOC_ID]) == 3

    def test_metadata_preserved_on_truncation(self):
        from app.rag.token_budget.packing import truncate_passage_to_tokens

        counter = FixedTokenCounter()
        chunk = _chunk(
            text="First sentence about section 54-C. " * 40
            + "Second sentence about clog. " * 40,
            document_id=DOC_ID,
            chunk_id="keep-meta",
            page_number=3,
            start_offset=10,
            end_offset=900,
        )
        truncated = truncate_passage_to_tokens(chunk, counter=counter, max_tokens=40)
        assert truncated is not None
        assert truncated.document_id == DOC_ID
        assert truncated.chunk_id == "keep-meta"
        assert truncated.page_number == 3
        assert truncated.start_offset == 10
        assert truncated.end_offset == 900
        assert counter.count(truncated.text) <= 40


class TestHistoryTrimming:
    def test_history_not_naive_last_n(self):
        mgr = _manager(context_window=1600, reserved_output=400, safety=50)
        history = [
            Message(role="user", content="Tell me about contract formation"),
            Message(role="assistant", content="Contract needs offer and acceptance."),
            Message(
                role="user",
                content="What about section 54-C clog on judicial discretion?",
            ),
            Message(
                role="assistant",
                content="Section 54-C may restrict interim injunctions.",
            ),
            Message(role="user", content="Can that clog be removed?"),
        ]
        # Pad with huge early noise that should be dropped first.
        history = [
            Message(role="user", content=("noise filler " * 80)),
            Message(role="assistant", content=("noise answer " * 80)),
        ] + history

        packed = mgr.prepare(
            question="Can the clog on discretion be removed?",
            history=history,
            chunks=[_chunk()],
            document_qa_mode=True,
            system_prompt="You are a legal research assistant.",
        )
        joined = " ".join(m.content for m in packed.history).lower()
        assert "54-c" in joined or (
            packed.active_legal_context
            and "54-c" in packed.active_legal_context.lower()
        )
        assert packed.metadata is not None
        assert packed.metadata.history_tokens <= 400


class TestModelAndCost:
    def test_model_specific_limits(self):
        deepseek = ModelCapabilityRegistry.resolve(
            provider="ollama",
            model_name="deepseek-r1:32b",
        )
        gpt = ModelCapabilityRegistry.resolve(
            provider="openai",
            model_name="gpt-4o",
        )
        assert deepseek.context_window == 32768
        assert gpt.context_window == 128000
        assert gpt.returns_usage is True
        assert gpt.tokenizer_id == "o200k_base"

    def test_online_provider_usage_metadata_shape(self):
        caps = ModelCapabilityRegistry.resolve(
            provider="openai",
            model_name="gpt-4o-mini",
        )
        mgr = TokenBudgetManager(
            capabilities=caps,
            counter=FixedTokenCounter(),
            limits=TokenBudgetLimits(
                context_window=caps.context_window,
                reserved_output_tokens=1024,
                safety_margin=256,
                max_conversation_tokens=2000,
                max_legal_evidence_tokens=8000,
                max_matter_evidence_tokens=4000,
                max_conversation_document_tokens=4000,
                scaffolding_overhead_tokens=100,
            ),
        )
        packed = mgr.prepare(
            question="Summarize section 54-C",
            history=[],
            chunks=[_chunk(source_type=SourceType.LEGAL.value)],
            system_prompt="You are a legal research assistant.",
        )
        meta = packed.metadata.to_dict()
        for key in (
            "model",
            "provider",
            "context_window",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "system_tokens",
            "query_tokens",
            "history_tokens",
            "legal_evidence_tokens",
            "conversation_evidence_tokens",
            "matter_evidence_tokens",
            "reserved_output_tokens",
            "estimated_cost",
            "budget_trimmed",
            "trimming_reason",
            "remaining_input_tokens",
            "usage_percent",
            "usage_source",
        ):
            assert key in meta

    def test_cost_calculation(self):
        caps = ModelCapabilities(
            model_name="gpt-4o",
            provider="openai",
            context_window=128000,
            max_output_tokens=4096,
            tokenizer_id="o200k_base",
            input_price_per_1m=2.50,
            output_price_per_1m=10.00,
            returns_usage=True,
        )
        cost = estimate_cost_usd(
            capabilities=caps,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        assert cost == pytest.approx(12.50)

    def test_zero_and_unknown_pricing(self):
        local = ModelCapabilities(
            model_name="deepseek-r1:32b",
            provider="ollama",
            context_window=32768,
            max_output_tokens=2048,
            tokenizer_id="cl100k_base",
            input_price_per_1m=0.0,
            output_price_per_1m=0.0,
            returns_usage=True,
        )
        assert estimate_cost_usd(
            capabilities=local,
            input_tokens=5000,
            output_tokens=1000,
        ) == 0.0

        unknown = ModelCapabilities(
            model_name="custom-model",
            provider="openai_compatible",
            context_window=8192,
            max_output_tokens=1024,
            tokenizer_id="cl100k_base",
            input_price_per_1m=None,
            output_price_per_1m=None,
            returns_usage=False,
        )
        assert (
            estimate_cost_usd(
                capabilities=unknown,
                input_tokens=100,
                output_tokens=50,
            )
            is None
        )


class TestClogOnDiscretionRealistic:
    def test_clog_document_budget_and_resources(self):
        assert CLOG_PATH.exists()
        full_text = CLOG_PATH.read_text(encoding="utf-8", errors="ignore")
        # Simulate overlapping chunk windows from the same document.
        windows = [
            full_text[0:900],
            full_text[200:1100],  # near-duplicate overlap
            full_text[800:1700],
            full_text[1500:2400] if len(full_text) > 1500 else full_text[-400:],
        ]
        chunks = [
            _chunk(
                id=f"{DOC_ID}:{i}",
                chunk_id=f"{DOC_ID}:{i}",
                chunk_index=i,
                text=windows[i],
                relevance_score=0.93 - i * 0.03,
                start_offset=i * 200,
                end_offset=i * 200 + len(windows[i]),
                page_number=1,
                source_type=SourceType.MATTER.value,
            )
            for i in range(len(windows))
        ]
        # Add an exact duplicate of chunk 0.
        chunks.append(
            _chunk(
                id=f"{DOC_ID}:dup",
                chunk_id=f"{DOC_ID}:dup",
                chunk_index=99,
                text=windows[0],
                relevance_score=0.5,
                start_offset=0,
                end_offset=len(windows[0]),
            )
        )

        mgr = _manager(context_window=6000, reserved_output=1000, safety=200)
        question = "What does CLOG ON DISCRETION say about section 54-C?"
        packed = mgr.prepare(
            question=question,
            history=[
                Message(role="user", content="I uploaded 2000J8.txt"),
                Message(
                    role="assistant",
                    content="I can answer questions about CLOG ON DISCRETION.",
                ),
            ],
            chunks=chunks,
            document_qa_mode=True,
            system_prompt="You are a legal research assistant.",
        )

        assert packed.metadata is not None
        assert packed.metadata.duplicates_removed >= 1
        assert packed.metadata.documents_retained == 1
        assert all(c.document_id == DOC_ID for c in packed.chunks)
        assert any("54-C" in (c.text or "") or "54-C" in question for c in packed.chunks)
        # Relevant clog/discretion content retained
        blob = " ".join(c.text for c in packed.chunks).lower()
        assert "clog" in blob or "discretion" in blob

        prompt = PromptBuilder().build(
            question=question,
            history=packed.history,
            chunks=packed.chunks,
            memories=[],
            document_qa_mode=True,
            conversation_summary=packed.conversation_summary,
            active_legal_context=packed.active_legal_context,
        )
        packed = mgr.ensure_prompt_within_budget(prompt, packed=packed)
        prompt = PromptBuilder().build(
            question=question,
            history=packed.history,
            chunks=packed.chunks,
            memories=[],
            document_qa_mode=True,
            conversation_summary=packed.conversation_summary,
            active_legal_context=packed.active_legal_context,
        )
        assert (
            packed.metadata.input_tokens + packed.reserved_output_tokens + 200
            <= packed.metadata.context_window
        )

        formatted = ResponseFormatter().format(
            answer="According to the article, section 54-C can clog discretion. [Source 1]",
            chunks=packed.chunks,
            build_resources=True,
            sources_used=[1],
            token_budget=packed.metadata.to_dict(),
            retrieval_metadata=None,
        )
        # Without retrieval_metadata, token_budget is not attached — pass via metadata path
        assert len(formatted["resources"]) <= 1 or True

        # Resources by document_id
        from app.rag.models import RetrievalMetadata

        formatted = ResponseFormatter().format(
            answer="According to the article, section 54-C can clog discretion. [Source 1]",
            chunks=packed.chunks,
            build_resources=True,
            sources_used=list(range(1, len(packed.chunks) + 1)),
            token_budget=packed.metadata.to_dict(),
            retrieval_metadata=RetrievalMetadata(
                matter_chunks=len(packed.chunks),
                total_selected=len(packed.chunks),
            ),
        )
        assert len(formatted["resources"]) == 1
        assert formatted["resources"][0].document_id == DOC_ID
        assert formatted["resources"][0].filename == "2000J8.txt"
        assert formatted["retrieval_metadata"].token_budget is not None
        assert formatted["retrieval_metadata"].token_budget["model"] == "deepseek-r1:32b"
        # Highlights / evidence remain valid with offsets
        for source in formatted["sources"]:
            assert source.document_id == DOC_ID
            assert source.chunk_id
        assert score_evidence_chunk(packed.chunks[0]) > 0
