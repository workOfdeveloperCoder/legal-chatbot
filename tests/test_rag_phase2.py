from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from app.rag.answer_guard import (
    AnswerGuard,
    CitationValidator,
    ClaimGroundingValidator,
    EvidenceStrength,
    GroundingStatus,
    assess_evidence_strength,
    requires_legal_authority,
)
from app.rag.context_resolver import ContextResolver
from app.rag.models import Message, RetrievedChunk, SourceType
from app.rag.query_rewriter import QueryRewriter
from app.schemas.chat import SourceReference
from app.services.response_formatter import ResponseFormatter
from app.vector.filters import QdrantFilterBuilder
from tests.evaluation.legal_regression_cases import LEGAL_REGRESSION_CASES


def _chunk(**kwargs) -> RetrievedChunk:
    defaults = {
        "id": "chunk-1",
        "chunk_id": "chunk-1",
        "score": 0.7,
        "text": "Section 302 of the Pakistan Penal Code deals with murder.",
        "source_type": SourceType.LEGAL.value,
        "sections": ["302"],
        "law_name": "Pakistan Penal Code",
    }
    defaults.update(kwargs)
    return RetrievedChunk(**defaults)


def _source(
    source_number: int,
    *,
    source_type: str = SourceType.LEGAL.value,
    excerpt: str | None = None,
) -> SourceReference:
    return SourceReference(
        id=f"source-{source_number}",
        source_number=source_number,
        source_type=source_type,
        excerpt=excerpt,
    )


class TestCitationValidator:
    def test_valid_citation(self):
        validator = CitationValidator()
        result = validator.validate(
            "Murder is defined under Section 302 [Source 1].",
            source_count=3,
        )
        assert result.valid == [1]
        assert result.invalid == []
        assert result.unused == [2, 3]
        assert result.status.value == "valid"

    def test_invalid_citation(self):
        validator = CitationValidator()
        result = validator.validate(
            "See [Source 99] for authority.",
            source_count=4,
        )
        assert result.valid == []
        assert result.invalid == [99]
        assert result.status.value == "partial"

    def test_duplicate_citation(self):
        validator = CitationValidator()
        result = validator.validate(
            "Murder [Source 1] and again [Source 1].",
            source_count=2,
        )
        assert result.valid == [1]
        assert 1 in result.valid

    def test_repair_removes_invalid_citation(self):
        validator = CitationValidator()
        validation = validator.validate(
            "Murder [Source 1] and fake [Source 99].",
            source_count=2,
        )
        repaired, revalidated = validator.repair(
            "Murder [Source 1] and fake [Source 99].",
            validation,
            source_count=2,
        )
        assert "[Source 99]" not in repaired
        assert "[Source 1]" in repaired
        assert revalidated.status.value == "repaired"

    def test_stable_source_number_mapping(self):
        formatter = ResponseFormatter()
        chunks = [
            _chunk(chunk_id="a", text="first"),
            _chunk(chunk_id="b", text="second"),
        ]
        result = formatter.format(answer="Answer", chunks=chunks)
        assert result["sources"][0].source_number == 1
        assert result["sources"][1].source_number == 2
        assert result["sources"][0].id == "source-1"
        assert result["sources"][1].id == "source-2"


class TestEvidenceStrength:
    def test_strong_legal_evidence(self):
        chunks = [
            _chunk(
                source_type=SourceType.LEGAL.value,
                relevance_score=0.82,
            )
        ]
        strength = assess_evidence_strength(
            chunks,
            requires_legal_authority=True,
        )
        assert strength == EvidenceStrength.STRONG

    def test_partial_evidence(self):
        chunks = [
            _chunk(
                source_type=SourceType.LEGAL.value,
                score=0.35,
                relevance_score=0.35,
            )
        ]
        strength = assess_evidence_strength(
            chunks,
            requires_legal_authority=True,
        )
        assert strength == EvidenceStrength.PARTIAL

    def test_no_legal_evidence(self):
        strength = assess_evidence_strength(
            [],
            requires_legal_authority=True,
        )
        assert strength == EvidenceStrength.NONE

    def test_private_document_only_for_legal_question(self):
        chunks = [
            _chunk(
                source_type=SourceType.MATTER.value,
                filename="plaint.pdf",
                text="The plaintiff claims Section 302 applies.",
            )
        ]
        strength = assess_evidence_strength(
            chunks,
            requires_legal_authority=True,
        )
        assert strength == EvidenceStrength.WEAK


class TestClaimGrounding:
    def test_supported_section_claim(self):
        validator = ClaimGroundingValidator()
        chunks = [_chunk(text="Section 302 of the Pakistan Penal Code")]
        sources = [_source(1, excerpt=chunks[0].text)]
        result = validator.validate(
            "Section 302 provides punishment for murder.",
            chunks=chunks,
            sources=sources,
        )
        assert result.status == GroundingStatus.STRONG
        assert result.unsupported_claims == []

    def test_unsupported_section_claim(self):
        validator = ClaimGroundingValidator()
        chunks = [_chunk(text="Section 497 CrPC deals with bail.")]
        sources = [_source(1, excerpt=chunks[0].text)]
        result = validator.validate(
            "Section 999 provides absolute immunity.",
            chunks=chunks,
            sources=sources,
        )
        assert result.status in {
            GroundingStatus.PARTIAL,
            GroundingStatus.INSUFFICIENT,
        }
        assert result.unsupported_claims

    def test_fabricated_case_citation(self):
        validator = ClaimGroundingValidator()
        chunks = [_chunk(text="Section 302 PPC")]
        sources = [_source(1)]
        result = validator.validate(
            "See PLD 2099 SC 999 for the holding.",
            chunks=chunks,
            sources=sources,
        )
        assert result.unsupported_claims


class TestAnswerGuard:
    def test_insufficient_evidence_template(self):
        guard = AnswerGuard()
        answer = guard.build_insufficient_evidence_answer(
            evidence_strength=EvidenceStrength.NONE,
        )
        assert "sufficient authoritative" in answer.lower()

    def test_invalid_citations_repaired_before_response(self):
        guard = AnswerGuard()
        chunks = [_chunk()]
        sources = [_source(1)]
        result = guard.process(
            answer="Murder under Section 302 [Source 1] and [Source 99].",
            chunks=chunks,
            sources=sources,
            evidence_strength=EvidenceStrength.STRONG,
        )
        assert "[Source 99]" not in result.answer
        assert result.citation_validation.status.value == "repaired"

    def test_weak_evidence_adds_qualification(self):
        guard = AnswerGuard()
        result = guard.process(
            answer="The contract permits termination.",
            chunks=[_chunk(source_type=SourceType.MATTER.value)],
            sources=[_source(1, source_type=SourceType.MATTER.value)],
            evidence_strength=EvidenceStrength.WEAK,
        )
        assert "limited" in result.answer.lower() or "Note:" in result.answer


class TestFollowUpContext:
    @pytest.mark.asyncio
    async def test_follow_up_inherits_context(self):
        resolver = ContextResolver()
        history = [
            Message(
                role="user",
                content="What is the limitation period for this claim?",
                created_at=datetime.now(timezone.utc),
            ),
            Message(
                role="assistant",
                content="The limitation period is three years.",
                created_at=datetime.now(timezone.utc),
            ),
        ]
        resolved = resolver.resolve("What about appeals?", history)
        assert "appeal" in resolved.lower()
        assert "limitation" in resolved.lower()

    @pytest.mark.asyncio
    async def test_explicit_query_overrides_context(self):
        resolver = ContextResolver()
        history = [
            Message(
                role="user",
                content="What is murder under Section 302?",
                created_at=datetime.now(timezone.utc),
            ),
        ]
        resolved = resolver.resolve(
            "What about limitation under the Limitation Act?",
            history,
        )
        assert resolved == "What about limitation under the Limitation Act?"

    @pytest.mark.asyncio
    async def test_rewriter_uses_history_flag(self):
        rewriter = QueryRewriter()
        history = [
            Message(
                role="user",
                content="What is the limitation period for filing a suit?",
                created_at=datetime.now(timezone.utc),
            ),
        ]
        result = await rewriter.rewrite(
            question="What about appeals?",
            history=history,
        )
        assert result.used_conversation_context is True
        assert "appeal" in result.resolved_query.lower()


class TestAuthorizationUnchanged:
    def test_conversation_filter_still_isolated(self):
        user_id = str(uuid.uuid4())
        conversation_a = str(uuid.uuid4())
        conversation_b = str(uuid.uuid4())

        filt = QdrantFilterBuilder.conversation_documents(
            user_id=user_id,
            conversation_id=conversation_b,
        )
        conversation_condition = next(
            cond for cond in filt.must if cond.key == "conversation_id"
        )
        assert conversation_condition.match.value == conversation_b
        assert conversation_condition.match.value != conversation_a

    def test_matter_filter_still_isolated(self):
        user_id = str(uuid.uuid4())
        matter_a = str(uuid.uuid4())
        matter_b = str(uuid.uuid4())

        filt = QdrantFilterBuilder.matter_documents(
            user_id=user_id,
            matter_id=matter_b,
        )
        matter_condition = next(
            cond for cond in filt.must if cond.key == "matter_id"
        )
        assert matter_condition.match.value == matter_b
        assert matter_condition.match.value != matter_a


class TestRequiresLegalAuthority:
    def test_legal_question_detected(self):
        assert requires_legal_authority(
            "What is the limitation period under the Limitation Act?"
        )

    def test_document_question_not_requires_legal(self):
        assert not requires_legal_authority(
            "Summarize this contract clause.",
            document_task=True,
        )


class TestLegalRegressionDataset:
    def test_dataset_has_minimum_cases(self):
        assert len(LEGAL_REGRESSION_CASES) >= 20

    def test_dataset_covers_topics(self):
        topics = {
            topic
            for case in LEGAL_REGRESSION_CASES
            for topic in case.expected_topics
        }
        assert "limitation" in topics
        assert "appeal" in topics
        assert "contract" in topics

    @pytest.mark.parametrize(
        "case",
        LEGAL_REGRESSION_CASES,
        ids=lambda case: case.question[:40],
    )
    def test_case_metadata_shape(self, case):
        assert case.expected_source_type in {
            "legal",
            "conversation",
            "matter",
        }
        assert len(case.expected_topics) >= 1

    def test_follow_up_case_marked(self):
        followups = [c for c in LEGAL_REGRESSION_CASES if c.follow_up]
        assert followups

    def test_explicit_override_case_marked(self):
        overrides = [
            c for c in LEGAL_REGRESSION_CASES if c.explicit_override
        ]
        assert overrides
