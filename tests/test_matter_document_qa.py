from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.answer_guard import AnswerGuard, EvidenceStrength, assess_evidence_strength
from app.rag.document_metadata import derive_display_name, enrich_chunks_from_documents
from app.rag.models import RetrievedChunk, SourceType
from app.rag.rag_service import RAGService
from app.services.query_router import QueryRouter, Task
from app.services.response_formatter import ResponseFormatter


CLOG_TEXT = """CLOG ON DISCRETION
By M. Qasim Khan Khattak, Advocate.

Albeit section 54-C of the Act prima facie contains a bar against
interim injunctions unless the impugned amount is deposited, blind
application can restrict judicial discretion and cause hardship where
bills are unreasonable. The article discusses double jeopardy where the
same period is charged again. The author argues that such a clog can be
removed in appropriate cases involving unreasonableness or double jeopardy.
"""

DOCUMENT_ID = "document-123"
FILENAME = "2000J8.txt"
MATTER_ID = str(uuid.uuid4())


def _chunk(**kwargs) -> RetrievedChunk:
    defaults = {
        "id": f"{DOCUMENT_ID}:0",
        "chunk_id": f"{DOCUMENT_ID}:0",
        "score": 0.91,
        "relevance_score": 0.91,
        "text": CLOG_TEXT,
        "filename": FILENAME,
        "document_id": DOCUMENT_ID,
        "chunk_index": 0,
        "source_type": SourceType.MATTER.value,
        "matter_id": MATTER_ID,
        "start_offset": 0,
        "end_offset": 500,
    }
    defaults.update(kwargs)
    return RetrievedChunk(**defaults)


def _document():
    return SimpleNamespace(
        id=uuid.UUID(DOCUMENT_ID) if False else DOCUMENT_ID,
        filename=FILENAME,
        extracted_text=CLOG_TEXT,
        matter_id=MATTER_ID,
    )


class TestDisplayNameDerivation:
    def test_derives_title_from_first_line(self):
        name = derive_display_name(CLOG_TEXT, FILENAME)
        assert name == "CLOG ON DISCRETION"

    def test_falls_back_to_filename_stem(self):
        name = derive_display_name("", FILENAME)
        assert name == "2000J8"


class TestQueryRouterMatterDocument:
    @pytest.mark.asyncio
    async def test_section_problem_question_routes_to_document_qa(self):
        router = QueryRouter()
        task = await router.detect(
            "What is the problem identified with section 54-C?",
            has_uploaded_documents=True,
        )
        assert task == Task.DOCUMENT_QA

    @pytest.mark.asyncio
    async def test_general_section_question_stays_statute(self):
        router = QueryRouter()
        task = await router.detect(
            "What is section 54-C under the Act?",
            has_uploaded_documents=True,
        )
        assert task == Task.STATUTE_SEARCH


class TestDynamicSourceIdentity:
    def test_source_contains_document_identity(self):
        document = SimpleNamespace(
            id=DOCUMENT_ID,
            filename=FILENAME,
            extracted_text=CLOG_TEXT,
            matter_id=MATTER_ID,
        )
        chunk = _chunk()
        enriched = enrich_chunks_from_documents(
            [chunk],
            {DOCUMENT_ID: document},
        )[0]

        formatter = ResponseFormatter()
        result = formatter.format(
            answer="According to the article, section 54-C is problematic. [Source 1]",
            chunks=[enriched],
        )

        source = result["sources"][0]
        assert source.document_id == DOCUMENT_ID
        assert source.filename == FILENAME
        assert source.display_name == "CLOG ON DISCRETION"
        assert source.document_name == "CLOG ON DISCRETION"
        assert source.chunk_id == f"{DOCUMENT_ID}:0"
        assert source.source_number == 1
        assert source.source_type == SourceType.MATTER.value
        assert "section 54-C" in (source.text or "")
        assert source.matter_id == MATTER_ID

    def test_citation_mapping_source_one(self):
        chunk = _chunk(display_name="CLOG ON DISCRETION")
        formatter = ResponseFormatter()
        preliminary = formatter.format(
            answer="The author considers section 54-C a clog. [Source 1]",
            chunks=[chunk],
        )
        guard = AnswerGuard()
        guarded = guard.process(
            answer=preliminary["answer"],
            chunks=[chunk],
            sources=preliminary["sources"],
            evidence_strength=EvidenceStrength.STRONG,
            document_evidence_mode=True,
        )

        assert guarded.citation_validation.valid == [1]
        assert guarded.citation_validation.invalid == []
        assert "[Source 99]" not in guarded.answer
        assert "could not verify all legal propositions" not in guarded.answer.lower()
        assert preliminary["sources"][0].document_id == DOCUMENT_ID

    def test_invalid_citation_removed(self):
        chunk = _chunk()
        formatter = ResponseFormatter()
        preliminary = formatter.format(answer="Text [Source 99]", chunks=[chunk])
        guard = AnswerGuard()
        guarded = guard.process(
            answer=preliminary["answer"],
            chunks=[chunk],
            sources=preliminary["sources"],
            evidence_strength=EvidenceStrength.STRONG,
            document_evidence_mode=True,
        )
        assert "[Source 99]" not in guarded.answer


class TestDocumentEvidenceMode:
    def test_document_qa_strong_with_matter_chunks(self):
        strength = assess_evidence_strength(
            [_chunk()],
            document_qa_mode=True,
            requires_legal_authority=False,
        )
        assert strength == EvidenceStrength.STRONG

    def test_document_section_claim_not_marked_insufficient(self):
        chunk = _chunk()
        formatter = ResponseFormatter()
        preliminary = formatter.format(
            answer=(
                "According to the article, section 54-C creates a bar against "
                "interim injunctions unless the impugned amount is deposited. "
                "[Source 1]"
            ),
            chunks=[chunk],
        )
        guard = AnswerGuard()
        guarded = guard.process(
            answer=preliminary["answer"],
            chunks=[chunk],
            sources=preliminary["sources"],
            evidence_strength=EvidenceStrength.STRONG,
            document_evidence_mode=True,
        )
        assert guarded.grounding_status.value == "strong"
        assert "could not verify all legal propositions" not in guarded.answer.lower()


class TestRAGServiceDocumentEnrichment:
    @pytest.mark.asyncio
    async def test_enriches_sources_from_document_repository(self):
        chunk = _chunk()
        from app.rag.models import RetrievalMetadata, RetrievalOutcome

        retriever = AsyncMock()
        retriever.search = AsyncMock(
            return_value=RetrievalOutcome(
                chunks=[chunk],
                metadata=RetrievalMetadata(matter_chunks=1, total_selected=1),
            )
        )

        document = SimpleNamespace(
            id=DOCUMENT_ID,
            filename=FILENAME,
            extracted_text=CLOG_TEXT,
            matter_id=MATTER_ID,
        )
        repo = AsyncMock()
        repo.get_by_id = AsyncMock(return_value=document)

        llm = AsyncMock()
        llm.generate = AsyncMock(
            return_value=SimpleNamespace(
                content=(
                    "According to the article, section 54-C can operate as a "
                    "clog on discretion. [Source 1]"
                ),
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            )
        )

        from app.rag.prompt_builder import PromptBuilder
        from app.rag.query_rewriter import RewriteResult

        query_rewriter = AsyncMock()
        query_rewriter.rewrite = AsyncMock(
            return_value=RewriteResult(
                original_query="What is the problem identified with section 54-C?",
                rewritten_query="section 54-C problem",
                resolved_query="What is the problem identified with section 54-C?",
                legal_terms=["section"],
                filters={},
                intent="document_qa",
                used_conversation_context=False,
            )
        )

        service = RAGService(
            llm=llm,
            retriever=retriever,
            query_rewriter=query_rewriter,
            prompt_builder=PromptBuilder(),
            response_formatter=ResponseFormatter(),
            document_repository=repo,
        )

        result = await service.execute(
            question="What is the problem identified with section 54-C?",
            history=[],
            memories=[],
            task=Task.DOCUMENT_QA,
            user_id=str(uuid.uuid4()),
            matter_id=MATTER_ID,
            conversation_id=str(uuid.uuid4()),
        )

        source = result["sources"][0]
        assert source.document_id == DOCUMENT_ID
        assert source.filename == FILENAME
        assert source.display_name == "CLOG ON DISCRETION"
        assert source.chunk_id == f"{DOCUMENT_ID}:0"
        assert "section 54-C" in (source.text or "")
        assert result["grounding_status"] == "strong"
        assert "could not verify all legal propositions" not in result["answer"].lower()
        repo.get_by_id.assert_awaited()

        meta = result["retrieval_metadata"]
        assert meta is not None
        assert meta.token_budget is not None
        assert meta.token_budget["reserved_output_tokens"] > 0
        assert meta.token_budget["input_tokens"] <= meta.token_budget["available_input_tokens"]
        assert (
            meta.token_budget["input_tokens"]
            + meta.token_budget["reserved_output_tokens"]
            + meta.token_budget.get("safety_margin_tokens", 0)
            <= meta.token_budget["context_window"]
        )
