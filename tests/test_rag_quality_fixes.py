"""Production RAG quality regression: resources, citations, attribution."""

from __future__ import annotations

import uuid

import pytest

from app.rag.answer_cleanup import clean_answer_for_display
from app.rag.answer_guard import (
    AnswerGuard,
    CitationValidator,
    ClaimGroundingValidator,
    EvidenceStrength,
    GroundingStatus,
)
from app.rag.context_resolver import ContextResolver
from app.rag.document_identity import recover_document_id
from app.rag.models import Message, RetrievedChunk, SourceType
from app.rag.prompt_builder import PromptBuilder
from app.rag.query_complexity import QueryComplexity
from app.services.query_router import QueryRouter, Task
from app.services.response_formatter import ResponseFormatter


DOC_A = "11111111-1111-1111-1111-111111111111"
DOC_B = "22222222-2222-2222-2222-222222222222"


def _chunk(
    *,
    document_id: str | None,
    filename: str,
    index: int,
    score: float,
    text: str | None = None,
    source_type: str = SourceType.MATTER.value,
) -> RetrievedChunk:
    chunk_id = f"{document_id}:{index}" if document_id else f"legal:{index}"
    return RetrievedChunk(
        id=chunk_id,
        chunk_id=chunk_id,
        score=score,
        relevance_score=score,
        text=text or f"Chunk {index} from {filename}",
        filename=filename,
        document_id=document_id,
        chunk_index=index,
        source_type=source_type,
        display_name="CLOG ON DISCRETION" if "2000J8" in filename else None,
        start_offset=index * 100,
        end_offset=(index + 1) * 100,
    )


class TestResourceDeduplication:
    def test_one_document_multiple_chunks_one_resource(self):
        chunks = [
            _chunk(document_id=DOC_A, filename="2000J8.txt", index=0, score=0.91),
            _chunk(document_id=DOC_A, filename="2000J8.txt", index=1, score=0.88),
            _chunk(document_id=DOC_A, filename="2000J8.txt", index=2, score=0.86),
        ]
        result = ResponseFormatter().format(
            answer="Answer [Source 1] [Source 2] [Source 3]",
            chunks=chunks,
            sources_used=[1, 2, 3],
            build_resources=True,
        )
        assert len(result["resources"]) == 1
        assert len(result["sources"]) == 3
        assert len(result["resources"][0].evidence) == 3

    def test_duplicate_chunk_identity_one_evidence(self):
        chunks = [
            _chunk(document_id=DOC_A, filename="2000J8.txt", index=0, score=0.91),
            RetrievedChunk(
                id=f"{DOC_A}:0-dup",
                chunk_id=f"{DOC_A}:0",
                document_id=str(uuid.UUID(DOC_A)).upper(),
                chunk_index=0,
                score=0.90,
                relevance_score=0.90,
                text="Same",
                filename="2000J8.txt",
                source_type=SourceType.MATTER.value,
            ),
        ]
        result = ResponseFormatter().format(
            answer="Answer [Source 1, Source 2]",
            chunks=chunks,
            sources_used=[1, 2],
            build_resources=True,
        )
        assert len(result["resources"]) == 1
        assert len(result["sources"]) == 1

    def test_same_filename_different_document_id_two_resources(self):
        chunks = [
            _chunk(document_id=DOC_A, filename="2000J8.txt", index=0, score=0.9),
            _chunk(document_id=DOC_B, filename="2000J8.txt", index=0, score=0.8),
        ]
        result = ResponseFormatter().format(
            answer="Answer [Source 1] [Source 2]",
            chunks=chunks,
            sources_used=[1, 2],
            build_resources=True,
        )
        assert len(result["resources"]) == 2
        assert {r.document_id for r in result["resources"]} == {DOC_A, DOC_B}

    def test_missing_document_id_recovered_from_chunk_id(self):
        chunks = [
            RetrievedChunk(
                id=f"{DOC_A}:0",
                chunk_id=f"{DOC_A}:0",
                document_id=None,
                chunk_index=0,
                score=0.9,
                relevance_score=0.9,
                text="Section 54-C discussion",
                filename="2000J8.txt",
                display_name="CLOG ON DISCRETION",
                source_type=SourceType.MATTER.value,
            ),
            RetrievedChunk(
                id=f"{DOC_A}:1",
                chunk_id=f"{DOC_A}:1",
                document_id=None,
                chunk_index=1,
                score=0.85,
                relevance_score=0.85,
                text="Double jeopardy discussion",
                filename="2000J8.txt",
                display_name="CLOG ON DISCRETION",
                source_type=SourceType.MATTER.value,
            ),
        ]
        assert recover_document_id(chunk_id=f"{DOC_A}:0") == DOC_A
        result = ResponseFormatter().format(
            answer="Answer [Source 1] [Source 2]",
            chunks=chunks,
            sources_used=[1, 2],
            build_resources=True,
        )
        assert len(result["resources"]) == 1
        assert result["resources"][0].document_id == DOC_A
        assert len(result["resources"][0].evidence) == 2


class TestCitationCleanup:
    def test_combined_marker_stripped_from_display(self):
        cleaned = clean_answer_for_display(
            "The article argues this [Source 1, Source 2].\n\nSources:\n-"
        )
        assert "[Source" not in cleaned
        assert "Sources:" not in cleaned

    def test_citation_validator_parses_combined_marker(self):
        validator = CitationValidator()
        result = validator.validate(
            "See [Source 1, Source 2] for support.",
            source_count=3,
        )
        assert result.valid == [1, 2]
        assert result.invalid == []

    def test_invalid_citation_removed(self):
        guard = AnswerGuard()
        result = guard.process(
            answer="Claim [Source 1] and fake [Source 99].",
            chunks=[_chunk(document_id=DOC_A, filename="a.txt", index=0, score=0.8)],
            sources=ResponseFormatter()._build_sources(
                [_chunk(document_id=DOC_A, filename="a.txt", index=0, score=0.8)]
            ),
            evidence_strength=EvidenceStrength.STRONG,
        )
        assert "[Source 99]" not in result.answer


class TestDocumentAttributionPrompt:
    def test_document_qa_prompt_requires_attribution(self):
        prompt = PromptBuilder().build(
            question="What is the problem identified with section 54-C?",
            history=[],
            chunks=[
                _chunk(
                    document_id=DOC_A,
                    filename="2000J8.txt",
                    index=0,
                    score=0.9,
                    text=(
                        "The author argues that Section 54-C operates as a "
                        "clog on judicial discretion and may prevent interim "
                        "relief in double jeopardy situations."
                    ),
                )
            ],
            memories=[],
            document_qa_mode=True,
            complexity=QueryComplexity.SIMPLE,
        )
        assert "According to the article" in prompt
        assert "Do NOT present document arguments as independently" in prompt


class TestDocumentQaGuardWarnings:
    def test_no_quote_warning_for_paraphrase_mismatch(self):
        guard = AnswerGuard()
        chunk = _chunk(
            document_id=DOC_A,
            filename="2000J8.txt",
            index=0,
            score=0.9,
            text="Section 54-C creates a clog on judicial discretion.",
        )
        sources = ResponseFormatter()._build_sources([chunk])
        result = guard.process(
            answer=(
                "According to the article, the main problem with Section 54-C "
                "is that it can operate as a clog on judicial discretion."
            ),
            chunks=[chunk],
            sources=sources,
            evidence_strength=EvidenceStrength.STRONG,
            document_evidence_mode=True,
        )
        assert "quoted passages could not be matched" not in result.answer.lower()


class TestClaimGroundingNotKeywordOnly:
    def test_section_mention_not_enough_for_double_jeopardy_claim(self):
        validator = ClaimGroundingValidator()
        chunks = [
            _chunk(
                document_id=DOC_A,
                filename="2000J8.txt",
                index=0,
                score=0.9,
                text="Section 54-C is discussed in the statute.",
                source_type=SourceType.LEGAL.value,
            )
        ]
        result = validator.validate(
            "Section 54-C prevents injunctions in double jeopardy cases.",
            chunks=chunks,
            sources=[],
            question=(
                "Can section 54-C prevent an injunction in a double jeopardy situation?"
            ),
        )
        assert result.status in {
            GroundingStatus.PARTIAL,
            GroundingStatus.INSUFFICIENT,
        }


class TestFollowUpContext:
    def test_double_jeopardy_preserves_section_context(self):
        resolver = ContextResolver()
        history = [
            Message(role="user", content="Explain section 54-C."),
            Message(
                role="assistant",
                content="Section 54-C restricts judicial discretion.",
            ),
        ]
        resolved = resolver.resolve("What about double jeopardy?", history)
        assert "double jeopardy" in resolved.lower()
        assert "54-c" in resolved.lower() or "section" in resolved.lower()


class TestQueryRouting:
    @pytest.mark.asyncio
    async def test_document_problem_question(self):
        router = QueryRouter()
        task = await router.detect(
            "What is the problem identified with section 54-C?",
            has_uploaded_documents=True,
        )
        assert task == Task.DOCUMENT_QA

    @pytest.mark.asyncio
    async def test_mixed_legal_correctness(self):
        router = QueryRouter()
        task = await router.detect(
            "Is the author's argument about section 54-C legally correct?",
            has_uploaded_documents=True,
        )
        assert task == Task.MIXED_QA
