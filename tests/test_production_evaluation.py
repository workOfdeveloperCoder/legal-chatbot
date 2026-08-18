"""
Legal RAG evaluation runner — offline checks without live LLM/Qdrant.

Covers planner modes, follow-up context, claim grounding, resource identity,
token budget overflow, and provider failure classification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.llm.base import BaseLLM
from app.llm.errors import LLMAllProvidersFailed, LLMPermanentError, LLMTransientError
from app.llm.gateway import LLMGateway
from app.llm.token_counter import FixedTokenCounter
from app.llm.model_capabilities import ModelCapabilities
from app.rag.answer_guard import AnswerGuard
from app.rag.context_resolver import ContextResolver
from app.rag.document_identity import normalize_document_id
from app.rag.legal_query_planner import LegalQueryPlanner
from app.rag.models import Message, RetrievedChunk, SourceType
from app.rag.token_budget import TokenBudgetManager
from app.rag.token_budget.models import TokenBudgetLimits
from app.schemas.llm import ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from app.services.response_formatter import ResponseFormatter
from tests.evaluation.legal_regression_cases import LEGAL_REGRESSION_CASES


CLOG_PATH = Path(
    "/Users/mrmacbook/Projects/legal-chatbot/storage/documents/"
    "411dfb9b-2f3f-4a44-a1c8-a2c067aef624_2000J8.txt"
)


def _chunk(**kwargs) -> RetrievedChunk:
    defaults = {
        "id": "doc:0",
        "chunk_id": "doc:0",
        "score": 0.9,
        "relevance_score": 0.9,
        "text": "placeholder",
        "filename": "2000J8.txt",
        "display_name": "CLOG ON DISCRETION",
        "document_id": "clog-doc",
        "source_type": SourceType.MATTER.value,
        "start_offset": 0,
        "end_offset": 100,
    }
    defaults.update(kwargs)
    return RetrievedChunk(**defaults)


class TestPlannerRegression:
    @pytest.mark.asyncio
    async def test_expected_answer_modes(self):
        planner = LegalQueryPlanner()
        for case in LEGAL_REGRESSION_CASES:
            if not case.expected_answer_mode:
                continue
            plan = await planner.plan(
                question=case.question,
                has_uploaded_documents=case.has_uploaded_documents,
            )
            assert plan.answer_mode.value == case.expected_answer_mode, case.question


class TestFollowUpRegression:
    def test_follow_up_cases_retain_context(self):
        resolver = ContextResolver()
        history = [
            Message(role="user", content="Explain Section 54-C."),
            Message(role="assistant", content="It can restrict interim relief."),
        ]
        for case in LEGAL_REGRESSION_CASES:
            if not case.follow_up:
                continue
            resolved = resolver.resolve(case.question, history=history)
            assert "54" in resolved.lower() or "section" in resolved.lower()


class TestClaimGroundingRegression:
    def test_hallucinated_citation_unsupported(self):
        from app.rag.answer_guard import EvidenceStrength

        chunks = [
            _chunk(
                text=(
                    "The article argues section 54-C can operate as a clog "
                    "on judicial discretion."
                ),
                source_type=SourceType.LEGAL.value,
            )
        ]
        sources = ResponseFormatter()._build_sources(chunks)
        result = AnswerGuard().process(
            answer=(
                "The Supreme Court in PLD 2099 SC 999 held that section 54-C "
                "is unconstitutional. [Source 1]"
            ),
            chunks=chunks,
            sources=sources,
            evidence_strength=EvidenceStrength.PARTIAL,
        )
        assert (
            any("PLD 2099" in c for c in result.grounding.unsupported_claims)
            or "not verified" in result.answer.lower()
            or result.grounding_status.value in {"partial", "insufficient"}
        )

    def test_unsupported_legal_claim_qualified(self):
        from app.rag.answer_guard import EvidenceStrength

        chunks = [
            _chunk(
                text="Section 302 PPC provides the punishment for murder.",
                source_type=SourceType.LEGAL.value,
                filename="ppc.txt",
                display_name="PPC",
                document_id="ppc",
            )
        ]
        sources = ResponseFormatter()._build_sources(chunks)
        result = AnswerGuard().process(
            answer=(
                "Under section 999 XYZ Act the accused has absolute immunity "
                "from all prosecution forever. [Source 1]"
            ),
            chunks=chunks,
            sources=sources,
            evidence_strength=EvidenceStrength.PARTIAL,
        )
        assert result.grounding_status.value in {"partial", "insufficient"}


class TestResourceDedupRegression:
    def test_one_document_one_resource_for_clog(self):
        assert CLOG_PATH.exists()
        text = CLOG_PATH.read_text(encoding="utf-8", errors="ignore")
        chunks = [
            _chunk(
                id=f"clog:{i}",
                chunk_id=f"clog:{i}",
                text=text[i * 200 : i * 200 + 500],
                start_offset=i * 200,
                end_offset=i * 200 + 500,
                chunk_index=i,
            )
            for i in range(3)
        ]
        # Exact duplicate chunk
        chunks.append(
            _chunk(
                id="clog:dup",
                chunk_id="clog:0",
                text=chunks[0].text,
                start_offset=0,
                end_offset=500,
            )
        )
        formatted = ResponseFormatter().format(
            answer="According to the article, section 54-C can clog discretion.",
            chunks=chunks,
            build_resources=True,
            sources_used=[1, 2, 3],
        )
        assert len(formatted["resources"]) == 1
        assert formatted["resources"][0].document_id == "clog-doc"
        assert formatted["resources"][0].filename == "2000J8.txt"
        keys = {normalize_document_id(c.document_id) for c in chunks}
        assert len(keys) == 1


class TestTokenBudgetOverflowRegression:
    def test_large_context_is_trimmed(self):
        caps = ModelCapabilities(
            model_name="deepseek-r1:32b",
            provider="ollama",
            context_window=1200,
            max_output_tokens=300,
            tokenizer_id="test",
            input_price_per_1m=0.0,
            output_price_per_1m=0.0,
            returns_usage=True,
        )
        mgr = TokenBudgetManager(
            capabilities=caps,
            counter=FixedTokenCounter(),
            limits=TokenBudgetLimits(
                context_window=1200,
                reserved_output_tokens=300,
                safety_margin=50,
                max_conversation_tokens=200,
                max_legal_evidence_tokens=400,
                max_matter_evidence_tokens=400,
                max_conversation_document_tokens=200,
                scaffolding_overhead_tokens=20,
            ),
        )
        chunks = [
            _chunk(
                id=f"c:{i}",
                chunk_id=f"c:{i}",
                text=("section 54-C clog discretion evidence passage " * 25),
                relevance_score=0.9 - i * 0.05,
            )
            for i in range(12)
        ]
        packed = mgr.prepare(
            question="What is the problem identified with Section 54-C?",
            history=[Message(role="user", content="noise " * 100)] * 5,
            chunks=chunks,
            document_qa_mode=True,
            system_prompt="You are a legal assistant.",
        )
        assert packed.metadata is not None
        assert packed.metadata.budget_trimmed is True
        assert packed.metadata.input_tokens <= packed.metadata.available_input_tokens


class TestProviderFailureRegression:
    @pytest.mark.asyncio
    async def test_llm_timeout_and_failure_do_not_fabricate(self):
        class Boom(BaseLLM):  # type: ignore[misc]
            provider_name = "boom"
            model_name = "boom"

            async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
                raise LLMTransientError("timeout")

        gateway = LLMGateway(Boom(), max_retries=0, timeout_seconds=1)
        with pytest.raises(LLMAllProvidersFailed):
            await gateway.generate(
                ChatCompletionRequest(
                    messages=[ChatMessage(role="user", content="hi")],
                )
            )

    @pytest.mark.asyncio
    async def test_permanent_error_not_retried_blindly(self):
        class AuthFail(BaseLLM):  # type: ignore[misc]
            provider_name = "auth"
            model_name = "auth"
            calls = 0

            async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
                type(self).calls += 1
                raise LLMPermanentError("401")

        AuthFail.calls = 0
        gateway = LLMGateway(AuthFail(), max_retries=3, retry_base_delay=0.01)
        with pytest.raises(LLMAllProvidersFailed):
            await gateway.generate(
                ChatCompletionRequest(
                    messages=[ChatMessage(role="user", content="hi")],
                )
            )
        assert AuthFail.calls == 1


class TestClogDocumentQaCase:
    def test_clog_problem_with_section_54c_expected_behavior(self):
        assert CLOG_PATH.exists()
        text = CLOG_PATH.read_text(encoding="utf-8", errors="ignore")
        chunks = [
            _chunk(
                id="clog:0",
                chunk_id="clog:0",
                text=text[:1200],
                start_offset=0,
                end_offset=min(1200, len(text)),
            ),
            _chunk(
                id="clog:1",
                chunk_id="clog:1",
                text=text[400:1600] if len(text) > 400 else text,
                start_offset=400,
                end_offset=min(1600, len(text)),
                relevance_score=0.85,
            ),
        ]
        from app.rag.answer_guard import EvidenceStrength

        answer = (
            "According to the article CLOG ON DISCRETION, the problem with "
            "Section 54-C is that it can operate as a clog on judicial "
            "discretion, especially where bills are unreasonable or the same "
            "period is charged again (double jeopardy). This is the author's "
            "argument, not an independent statement of binding law. [Source 1]"
        )
        formatted = ResponseFormatter().format(
            answer=answer,
            chunks=chunks,
            build_resources=True,
            sources_used=[1, 2],
        )
        guarded = AnswerGuard().process(
            answer=formatted["answer"],
            chunks=chunks,
            sources=formatted["sources"],
            evidence_strength=EvidenceStrength.STRONG,
            document_evidence_mode=True,
            question="What is the problem identified with Section 54-C?",
        )
        assert len(formatted["resources"]) == 1
        assert "sources:" not in guarded.answer.lower().split("\n")[-1]
        assert "clog" in guarded.answer.lower()
        assert "discretion" in guarded.answer.lower()
        # Must attribute to article/document language ideally; at least not invent cites
        assert "PLD 2099" not in guarded.answer
