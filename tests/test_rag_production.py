"""Production RAG tests: resource dedup, sources separation, routing."""

from __future__ import annotations

import uuid

import pytest

from app.rag.models import RetrievedChunk, SourceType
from app.services.query_router import QueryRouter, Task
from app.services.response_formatter import ResponseFormatter


DOC_A = "11111111-1111-1111-1111-111111111111"
DOC_B = "22222222-2222-2222-2222-222222222222"


def _chunk(
    *,
    document_id: str,
    filename: str,
    index: int,
    score: float,
    text: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{document_id}:{index}",
        chunk_id=f"{document_id}:{index}",
        score=score,
        relevance_score=score,
        text=text or f"Chunk {index} from {filename}",
        filename=filename,
        document_id=document_id,
        chunk_index=index,
        source_type=SourceType.MATTER.value,
        start_offset=index * 100,
        end_offset=(index + 1) * 100,
    )


class TestResourceDeduplication:
    def test_same_document_id_produces_one_resource(self):
        chunks = [
            _chunk(document_id=DOC_A, filename="2000J8.txt", index=0, score=0.91),
            _chunk(document_id=DOC_A, filename="2000J8.txt", index=1, score=0.88),
            _chunk(document_id=DOC_A, filename="2000J8.txt", index=2, score=0.86),
        ]
        formatter = ResponseFormatter()
        result = formatter.format(
            answer="Answer [Source 1] [Source 2] [Source 3]",
            chunks=chunks,
            sources_used=[1, 2, 3],
            build_resources=True,
        )

        assert len(result["resources"]) == 1
        resource = result["resources"][0]
        assert resource.document_id == DOC_A
        assert len(resource.source_ids) == 3
        assert len(resource.evidence) == 3
        assert len(result["sources"]) == 3

    def test_same_filename_different_document_id_produces_two_resources(self):
        chunks = [
            _chunk(document_id=DOC_A, filename="contract.pdf", index=0, score=0.90),
            _chunk(document_id=DOC_A, filename="contract.pdf", index=1, score=0.85),
            _chunk(document_id=DOC_B, filename="contract.pdf", index=0, score=0.80),
            _chunk(document_id=DOC_B, filename="contract.pdf", index=1, score=0.75),
        ]
        formatter = ResponseFormatter()
        result = formatter.format(
            answer="Answer [Source 1] [Source 2] [Source 3] [Source 4]",
            chunks=chunks,
            sources_used=[1, 2, 3, 4],
            build_resources=True,
        )

        # HARD INVARIANT: different document_id => separate resources
        assert len(result["resources"]) == 2
        doc_ids = {resource.document_id for resource in result["resources"]}
        assert doc_ids == {DOC_A, DOC_B}
        assert len(result["sources"]) == 4

    def test_uncited_answer_keeps_only_top_relevant_resources(self):
        chunks = [
            _chunk(document_id=DOC_A, filename="2000J8.txt", index=0, score=0.92),
            _chunk(document_id=DOC_A, filename="2000J8.txt", index=1, score=0.88),
            _chunk(document_id=DOC_B, filename="unrelated.pdf", index=0, score=0.31),
        ]
        formatter = ResponseFormatter()
        result = formatter.format(
            answer="According to the article, section 54-C creates a clog.",
            chunks=chunks,
            sources_used=[],
            build_resources=True,
        )

        assert len(result["resources"]) == 1
        assert result["resources"][0].filename == "2000J8.txt"

    def test_same_document_duplicate_chunk_produces_one_source(self):
        chunks = [
            _chunk(document_id=DOC_A, filename="2000J8.txt", index=0, score=0.91),
            RetrievedChunk(
                id=f"{DOC_A}:0-dup",
                chunk_id=f"{DOC_A}:0",
                document_id=str(uuid.UUID(DOC_A)).upper(),
                chunk_index=0,
                score=0.90,
                relevance_score=0.90,
                text="Chunk 0 from 2000J8.txt",
                filename="2000J8.txt",
                source_type=SourceType.MATTER.value,
            ),
        ]
        formatter = ResponseFormatter()
        result = formatter.format(
            answer="Answer [Source 1] [Source 2]",
            chunks=chunks,
            sources_used=[1, 2],
            build_resources=True,
        )

        assert len(result["resources"]) == 1
        assert len(result["sources"]) == 1
        assert len(result["resources"][0].evidence) == 1

    def test_relevance_percent_normalized_on_resources(self):
        chunks = [
            _chunk(document_id=DOC_A, filename="a.txt", index=0, score=0.50),
            _chunk(document_id=DOC_B, filename="b.txt", index=0, score=1.00),
        ]
        formatter = ResponseFormatter()
        result = formatter.format(
            answer="Answer [Source 1] [Source 2]",
            chunks=chunks,
            sources_used=[1, 2],
            build_resources=True,
        )

        by_doc = {r.document_id: r for r in result["resources"]}
        assert by_doc[DOC_B].relevance_percent == 100
        assert by_doc[DOC_A].relevance_percent == 50


class TestQueryRouting:
    @pytest.mark.asyncio
    async def test_mixed_document_law_question(self):
        router = QueryRouter()
        task = await router.detect(
            "Does the argument in this document reflect Pakistani law on section 54-C?",
            has_uploaded_documents=True,
        )
        assert task == Task.MIXED_QA

    @pytest.mark.asyncio
    async def test_document_content_question_stays_document_qa(self):
        router = QueryRouter()
        task = await router.detect(
            "What is the problem identified with section 54-C in this document?",
            has_uploaded_documents=True,
        )
        assert task == Task.DOCUMENT_QA

    @pytest.mark.asyncio
    async def test_general_law_question_without_uploads(self):
        router = QueryRouter()
        task = await router.detect(
            "What is the law regarding section 54-C?",
            has_uploaded_documents=False,
        )
        assert task == Task.STATUTE_SEARCH


class TestFollowUpContext:
    def test_double_jeopardy_follow_up(self):
        from app.rag.context_resolver import ContextResolver
        from app.rag.models import Message

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
