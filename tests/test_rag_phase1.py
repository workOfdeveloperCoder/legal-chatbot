from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.evidence import dedupe_chunks, select_evidence
from app.rag.models import RetrievedChunk, RetrievalMetadata, SourceType
from app.rag.reranker import HybridReranker
from app.rag.retrieval_pipeline import build_source_reference, finalize_retrieval
from app.services.response_formatter import ResponseFormatter
from app.vector.filters import QdrantFilterBuilder


def _chunk(
    *,
    chunk_id: str,
    source_type: str,
    score: float = 0.5,
    document_id: str | None = None,
    chunk_index: int | None = None,
    text: str = "sample legal text",
    **kwargs,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        chunk_id=chunk_id,
        score=score,
        text=text,
        source_type=source_type,
        document_id=document_id,
        chunk_index=chunk_index,
        **kwargs,
    )


class TestEvidenceSelection:
    def test_dedupe_keeps_highest_score(self):
        chunks = [
            _chunk(chunk_id="a", source_type="legal", score=0.4),
            _chunk(chunk_id="a", source_type="legal", score=0.9),
        ]

        result = dedupe_chunks(chunks)

        assert len(result) == 1
        assert result[0].score == 0.9

    def test_select_evidence_preserves_tier_priority(self):
        chunks = [
            _chunk(
                chunk_id="matter-1",
                source_type=SourceType.MATTER.value,
                score=0.99,
                document_id="m1",
                chunk_index=0,
            ),
            _chunk(
                chunk_id="legal-1",
                source_type=SourceType.LEGAL.value,
                score=0.70,
                law_name="Pakistan Penal Code",
                sections=["302"],
            ),
            _chunk(
                chunk_id="conv-1",
                source_type=SourceType.CONVERSATION.value,
                score=0.95,
                document_id="c1",
                chunk_index=0,
                filename="contract.pdf",
            ),
            _chunk(
                chunk_id="legal-2",
                source_type=SourceType.LEGAL.value,
                score=0.65,
                law_name="CrPC",
                sections=["497"],
            ),
        ]

        selected = select_evidence(
            chunks,
            limit=4,
            legal_limit=2,
            conversation_limit=1,
            matter_limit=1,
        )

        types = [c.source_type for c in selected]
        assert types.count(SourceType.LEGAL.value) == 2
        assert SourceType.CONVERSATION.value in types
        assert SourceType.MATTER.value in types
        assert types.index(SourceType.LEGAL.value) < types.index(
            SourceType.CONVERSATION.value
        )


class TestHybridReranker:
    @pytest.mark.asyncio
    async def test_boosts_legal_tier_for_general_queries(self):
        reranker = HybridReranker()
        chunks = [
            _chunk(
                chunk_id="legal",
                source_type=SourceType.LEGAL.value,
                score=0.50,
                law_name="PPC",
                sections=["302"],
                text="murder section 302 pakistan penal code",
            ),
            _chunk(
                chunk_id="matter",
                source_type=SourceType.MATTER.value,
                score=0.55,
                filename="plaint.pdf",
                text="plaint facts about murder",
            ),
        ]

        ranked = await reranker.rerank(
            query="murder section 302",
            chunks=chunks,
            filters={"section": "302"},
        )

        assert ranked[0].source_type == SourceType.LEGAL.value
        assert ranked[0].relevance_score is not None

    @pytest.mark.asyncio
    async def test_boosts_private_docs_for_document_tasks(self):
        reranker = HybridReranker()
        chunks = [
            _chunk(
                chunk_id="legal",
                source_type=SourceType.LEGAL.value,
                score=0.80,
                text="statute text",
            ),
            _chunk(
                chunk_id="matter",
                source_type=SourceType.MATTER.value,
                score=0.60,
                filename="contract.pdf",
                text="contract termination clause thirty days",
            ),
        ]

        ranked = await reranker.rerank(
            query="contract termination thirty days",
            chunks=chunks,
            document_task=True,
        )

        assert ranked[0].source_type == SourceType.MATTER.value


class TestIsolationFilters:
    def test_conversation_documents_filter_structure(self):
        user_id = str(uuid.uuid4())
        conversation_id = str(uuid.uuid4())

        filt = QdrantFilterBuilder.conversation_documents(
            user_id=user_id,
            conversation_id=conversation_id,
        )

        keys = {cond.key for cond in filt.must}
        assert keys == {"user_id", "scope", "conversation_id"}

    def test_matter_documents_filter_structure(self):
        user_id = str(uuid.uuid4())
        matter_id = str(uuid.uuid4())

        filt = QdrantFilterBuilder.matter_documents(
            user_id=user_id,
            matter_id=matter_id,
        )

        keys = {cond.key for cond in filt.must}
        assert keys == {"user_id", "scope", "matter_id"}

    def test_matter_filter_excludes_other_matter(self):
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


class TestResponseFormatter:
    def test_builds_chunk_level_sources_with_highlight_metadata(self):
        formatter = ResponseFormatter()
        chunks = [
            _chunk(
                chunk_id="doc:0",
                source_type=SourceType.MATTER.value,
                score=0.88,
                document_id=str(uuid.uuid4()),
                chunk_index=0,
                filename="Plaint.pdf",
                page_number=12,
                start_offset=100,
                end_offset=250,
                text="The agreement permits termination after thirty days.",
            )
        ]

        result = formatter.format(answer="Answer text.", chunks=chunks)

        assert len(result["sources"]) == 1
        source = result["sources"][0]
        assert source.source_type == SourceType.MATTER.value
        assert source.page == 12
        assert source.start_offset == 100
        assert source.end_offset == 250
        assert source.chunk_id == "doc:0"
        assert source.document_name == "Plaint.pdf"

    def test_citations_include_source_type(self):
        formatter = ResponseFormatter()
        chunks = [
            _chunk(
                chunk_id="legal:1",
                source_type=SourceType.LEGAL.value,
                score=0.77,
                law_name="Pakistan Penal Code",
                sections=["302"],
            )
        ]

        result = formatter.format(answer="Answer.", chunks=chunks)
        assert result["citations"][0].source_type == SourceType.LEGAL.value


class TestSourceReference:
    def test_builds_legal_reference(self):
        chunk = _chunk(
            chunk_id="x",
            source_type=SourceType.LEGAL.value,
            law_name="Pakistan Penal Code",
            sections=["302"],
            court="Supreme Court",
            year=2020,
        )

        ref = build_source_reference(chunk)

        assert "Pakistan Penal Code" in ref
        assert "302" in ref
        assert "2020" in ref


class TestTieredRetriever:
    @pytest.fixture(autouse=True)
    def _nomic_embedding_for_qdrant(self, monkeypatch):
        monkeypatch.setattr(
            "app.rag.qdrant_retriever.settings.EMBEDDING_MODEL",
            "nomic-embed-text:latest",
        )

    @pytest.mark.asyncio
    async def test_search_queries_legal_before_private_tiers(self):
        from app.rag.qdrant_retriever import QdrantRetriever

        retriever = QdrantRetriever()
        call_order: list[str] = []

        async def fake_embed(_query):
            return [0.1, 0.2, 0.3]

        async def fake_query_points(*, collection_name, **kwargs):
            call_order.append(collection_name)
            point = SimpleNamespace(
                id="p1",
                score=0.8,
                payload={
                    "text": "retrieved text",
                    "document_id": str(uuid.uuid4()),
                    "chunk_index": 0,
                    "scope": (
                        "matter"
                        if collection_name == "chatbot_documents"
                        else None
                    ),
                    "matter_id": str(uuid.uuid4()),
                },
            )
            return SimpleNamespace(points=[point])

        retriever.embedding.embed_query = fake_embed
        retriever.client.query_points = fake_query_points

        outcome = await retriever.search(
            query="bail section 497",
            user_id=str(uuid.uuid4()),
            matter_id=str(uuid.uuid4()),
            conversation_id=str(uuid.uuid4()),
            filters={"section": "497"},
        )

        assert call_order[0] == "legal_documents"
        assert "chatbot_documents" in call_order
        assert outcome.metadata.legal_chunks >= 0
        assert outcome.chunks

    @pytest.mark.asyncio
    async def test_prefer_legal_corpus_skips_private_uploads(self):
        from app.rag.qdrant_retriever import QdrantRetriever

        retriever = QdrantRetriever()
        call_order: list[str] = []

        async def fake_embed(_query):
            return [0.1]

        async def fake_query_points(*, collection_name, **kwargs):
            call_order.append(collection_name)
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="legal-1",
                        score=0.88,
                        payload={
                            "text": "Statutes are divided into sections and articles.",
                            "filename": "structure.txt",
                            "chunk_index": 0,
                        },
                    )
                ]
            )

        retriever.embedding.embed_query = fake_embed
        retriever.client.query_points = fake_query_points

        outcome = await retriever.search(
            query="what is sections and articles in law",
            user_id=str(uuid.uuid4()),
            matter_id=str(uuid.uuid4()),
            conversation_id=str(uuid.uuid4()),
            task="statute_search",
            prefer_legal_corpus=True,
        )

        assert call_order == ["legal_documents"]
        assert "chatbot_documents" not in call_order
        assert outcome.metadata.legal_chunks >= 1
        assert all(
            (c.source_type or "legal") == "legal" for c in outcome.chunks
        )

    @pytest.mark.asyncio
    async def test_document_scoped_skips_legal_corpus(self):
        from app.rag.qdrant_retriever import QdrantRetriever

        retriever = QdrantRetriever()
        collections: list[str] = []

        async def fake_embed(_query):
            return [0.1]

        async def fake_query_points(*, collection_name, **kwargs):
            collections.append(collection_name)
            doc_id = kwargs["query_filter"].must[1].match.value
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="p1",
                        score=0.9,
                        payload={
                            "text": "doc chunk",
                            "document_id": doc_id,
                            "chunk_index": 0,
                            "scope": "conversation",
                            "conversation_id": str(uuid.uuid4()),
                        },
                    )
                ]
            )

        retriever.embedding.embed_query = fake_embed
        retriever.client.query_points = fake_query_points

        document_id = str(uuid.uuid4())
        outcome = await retriever.search(
            query="summarize",
            user_id=str(uuid.uuid4()),
            conversation_id=str(uuid.uuid4()),
            task="document_qa",
            document_id=document_id,
        )

        assert collections == ["chatbot_documents"]
        assert "legal_documents" not in collections
        assert outcome.chunks[0].document_id == document_id


class TestFinalizeRetrieval:
    @pytest.mark.asyncio
    async def test_finalize_applies_rerank_and_metadata(self):
        metadata = RetrievalMetadata(filters_applied={"section": "302"})
        chunks = [
            _chunk(
                chunk_id="1",
                source_type=SourceType.LEGAL.value,
                score=0.6,
                law_name="PPC",
                sections=["302"],
                text="section 302 murder",
            ),
            _chunk(
                chunk_id="2",
                source_type=SourceType.MATTER.value,
                score=0.9,
                document_id="d1",
                chunk_index=1,
                filename="plaint.pdf",
            ),
        ]

        outcome = await finalize_retrieval(
            query="section 302 murder",
            chunks=chunks,
            filters={"section": "302"},
            metadata=metadata,
            limit=2,
            legal_limit=1,
            conversation_limit=0,
            matter_limit=1,
        )

        assert outcome.metadata.total_selected <= 2
        assert outcome.metadata.legal_chunks >= 0
