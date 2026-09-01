from __future__ import annotations

import asyncio
import logging

from app.core.config import settings
from app.embeddings.service import EmbeddingService

from app.rag.document_identity import normalize_document_id
from app.rag.models import (
    RetrievedChunk,
    RetrievalMetadata,
    RetrievalOutcome,
    SourceType,
)
from app.rag.repository import BaseRetriever
from app.rag.retrieval_pipeline import finalize_retrieval

from app.vector.qdrant import qdrant_service
from app.vector.filters import QdrantFilterBuilder


logger = logging.getLogger(__name__)

_DOCUMENT_TASKS = {"summarization", "document_qa"}
_MIXED_TASKS = {"mixed_qa"}


def _task_value(task: str | None) -> str | None:
    if task is None:
        return None
    return task.value if hasattr(task, "value") else str(task)


class QdrantRetriever(BaseRetriever):
    """
    Tiered legal RAG retriever.

    Retrieval priority (general legal chat):
        1. legal_documents      — authoritative legal corpus
        2. conversation docs    — current conversation uploads
        3. matter documents     — matter workspace uploads

    Document-scoped tasks query only the requested uploaded document.
    """

    def __init__(self) -> None:
        self.client = qdrant_service.client
        self.embedding = EmbeddingService()

    async def search(
        self,
        *,
        query: str,
        user_id: str,
        matter_id: str | None = None,
        conversation_id: str | None = None,
        task: str | None = None,
        document_id: str | None = None,
        limit: int | None = None,
        filters: dict[str, str] | None = None,
        prefer_legal_corpus: bool = False,
    ) -> RetrievalOutcome:
        task_name = _task_value(task)
        document_scoped = task_name in _DOCUMENT_TASKS and bool(document_id)
        skip_legal_corpus = (
            task_name in _DOCUMENT_TASKS
            and task_name not in _MIXED_TASKS
        )
        # Legal research: answer from legal_documents first; skip private
        # uploads unless the user scoped a document or mixed-doc task.
        corpus_only = (
            prefer_legal_corpus
            and not document_scoped
            and not skip_legal_corpus
            and not document_id
            and task_name not in _MIXED_TASKS
        )

        total_limit = limit or settings.RETRIEVAL_LIMIT
        legal_limit = settings.RETRIEVAL_LEGAL_LIMIT
        conversation_limit = settings.RETRIEVAL_CONVERSATION_LIMIT
        matter_limit = settings.RETRIEVAL_MATTER_LIMIT

        if corpus_only:
            legal_limit = max(legal_limit, min(total_limit, legal_limit + 2))
            conversation_limit = 0
            matter_limit = 0
        elif skip_legal_corpus and not document_scoped:
            legal_limit = 0
            matter_limit = max(matter_limit, 6)
            conversation_limit = min(conversation_limit, 2)

        metadata = RetrievalMetadata(
            filters_applied=dict(filters or {}),
        )

        if not settings.qdrant_embedding_compatible:
            logger.warning(
                "Skipping Qdrant search: embedding model %s is not nomic-compatible "
                "with the %s-d legal corpus index",
                settings.EMBEDDING_MODEL,
                settings.VECTOR_SIZE,
            )
            metadata.degraded_sources.append("qdrant_embedding_model")
            return await finalize_retrieval(
                query=query,
                chunks=[],
                filters=filters,
                metadata=metadata,
                limit=total_limit,
                legal_limit=legal_limit,
                conversation_limit=conversation_limit,
                matter_limit=matter_limit,
            )

        logger.info(
            "Tiered Qdrant search user=%s matter=%s conversation=%s "
            "document_id=%s task=%s document_scoped=%s skip_legal=%s "
            "prefer_legal_corpus=%s corpus_only=%s legal_collection=%s",
            user_id,
            matter_id,
            conversation_id,
            document_id,
            task_name,
            document_scoped,
            skip_legal_corpus,
            prefer_legal_corpus,
            corpus_only,
            qdrant_service.legal_collection,
        )

        vector = await self.embedding.embed_query(query)
        tier_chunks: list[RetrievedChunk] = []

        if document_scoped:
            tier_chunks.extend(
                await self._search_document_scoped(
                    vector=vector,
                    user_id=user_id,
                    document_id=document_id,
                    matter_id=matter_id,
                    conversation_id=conversation_id,
                    limit=total_limit,
                    metadata=metadata,
                )
            )
            if not tier_chunks and metadata.degraded_sources:
                pass
        elif skip_legal_corpus:
            tier_chunks.extend(
                await self._search_private_tiers(
                    vector=vector,
                    user_id=user_id,
                    matter_id=matter_id,
                    conversation_id=conversation_id,
                    legal_limit=0,
                    conversation_limit=conversation_limit,
                    matter_limit=matter_limit,
                    metadata=metadata,
                )
            )
        else:
            tier_chunks.extend(
                await self._search_legal(
                    vector=vector,
                    limit=legal_limit,
                    filters=filters,
                    metadata=metadata,
                )
            )
            if not corpus_only:
                tier_chunks.extend(
                    await self._search_private_tiers(
                        vector=vector,
                        user_id=user_id,
                        matter_id=matter_id,
                        conversation_id=conversation_id,
                        legal_limit=0,
                        conversation_limit=conversation_limit,
                        matter_limit=matter_limit,
                        metadata=metadata,
                    )
                )

        return await finalize_retrieval(
            query=query,
            chunks=tier_chunks,
            filters=filters,
            metadata=metadata,
            limit=total_limit,
            legal_limit=legal_limit,
            conversation_limit=conversation_limit,
            matter_limit=matter_limit,
            document_task=skip_legal_corpus,
        )

    async def _search_document_scoped(
        self,
        *,
        vector,
        user_id: str,
        document_id: str,
        matter_id: str | None,
        conversation_id: str | None,
        limit: int,
        metadata: RetrievalMetadata,
    ) -> list[RetrievedChunk]:
        try:
            private_filter = QdrantFilterBuilder.document_id_scoped(
                user_id=user_id,
                document_id=document_id,
                matter_id=matter_id,
                conversation_id=conversation_id,
            )
            points = await self.client.query_points(
                collection_name=qdrant_service.document_collection,
                query=vector,
                query_filter=private_filter,
                limit=limit,
            )
            metadata.collections_queried.append(
                qdrant_service.document_collection
            )
            return self._convert_points(
                points.points,
                source_type=SourceType.CONVERSATION.value,
            )
        except Exception:
            logger.exception("Document-scoped retrieval failed")
            metadata.degraded_sources.append("document_scoped")
            return []

    async def _search_legal(
        self,
        *,
        vector,
        limit: int,
        filters: dict[str, str] | None,
        metadata: RetrievalMetadata,
    ) -> list[RetrievedChunk]:
        if limit <= 0:
            return []

        try:
            legal_filter = QdrantFilterBuilder.legal_metadata(filters)
            points = await self.client.query_points(
                collection_name=qdrant_service.legal_collection,
                query=vector,
                query_filter=legal_filter,
                limit=limit,
            )
            metadata.collections_queried.append(
                qdrant_service.legal_collection
            )
            return self._convert_points(
                points.points,
                source_type=SourceType.LEGAL.value,
            )
        except Exception:
            logger.exception("Legal corpus retrieval failed")
            metadata.degraded_sources.append(SourceType.LEGAL.value)
            return []

    async def _search_private_tiers(
        self,
        *,
        vector,
        user_id: str,
        matter_id: str | None,
        conversation_id: str | None,
        legal_limit: int,
        conversation_limit: int,
        matter_limit: int,
        metadata: RetrievalMetadata,
    ) -> list[RetrievedChunk]:
        tasks = []

        if conversation_id and conversation_limit > 0:
            tasks.append(
                self._search_collection(
                    vector=vector,
                    collection=qdrant_service.document_collection,
                    qdrant_filter=QdrantFilterBuilder.conversation_documents(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    ),
                    limit=conversation_limit,
                    source_type=SourceType.CONVERSATION.value,
                    metadata=metadata,
                    degrade_key=SourceType.CONVERSATION.value,
                )
            )

        if matter_id and matter_limit > 0:
            tasks.append(
                self._search_collection(
                    vector=vector,
                    collection=qdrant_service.document_collection,
                    qdrant_filter=QdrantFilterBuilder.matter_documents(
                        user_id=user_id,
                        matter_id=matter_id,
                    ),
                    limit=matter_limit,
                    source_type=SourceType.MATTER.value,
                    metadata=metadata,
                    degrade_key=SourceType.MATTER.value,
                )
            )
            # Conversation uploads stamped with this matter (sibling threads).
            tasks.append(
                self._search_collection(
                    vector=vector,
                    collection=qdrant_service.document_collection,
                    qdrant_filter=QdrantFilterBuilder.matter_linked_conversation_documents(
                        user_id=user_id,
                        matter_id=matter_id,
                    ),
                    limit=max(conversation_limit, matter_limit),
                    source_type=SourceType.CONVERSATION.value,
                    metadata=metadata,
                    degrade_key=SourceType.CONVERSATION.value,
                )
            )

        if not tasks:
            return []

        results = await asyncio.gather(*tasks)
        chunks: list[RetrievedChunk] = []
        for batch in results:
            chunks.extend(batch)
        return chunks

    async def _search_collection(
        self,
        *,
        vector,
        collection: str,
        qdrant_filter,
        limit: int,
        source_type: str,
        metadata: RetrievalMetadata,
        degrade_key: str,
    ) -> list[RetrievedChunk]:
        try:
            points = await self.client.query_points(
                collection_name=collection,
                query=vector,
                query_filter=qdrant_filter,
                limit=limit,
            )
            if collection not in metadata.collections_queried:
                metadata.collections_queried.append(collection)
            return self._convert_points(
                points.points,
                source_type=source_type,
            )
        except Exception:
            logger.exception(
                "Retrieval failed collection=%s source_type=%s",
                collection,
                source_type,
            )
            if degrade_key not in metadata.degraded_sources:
                metadata.degraded_sources.append(degrade_key)
            return []

    def _convert_points(
        self,
        points,
        source_type: str,
    ) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []

        for point in points:
            payload = point.payload or {}

            if payload.get("is_parent") is True:
                continue
            if payload.get("embedded") is False:
                continue
            if payload.get("is_retrieval_valid") is False:
                continue

            chunk_id = str(
                payload.get("chunk_id", point.id)
            )
            chunk_index = payload.get("chunk_index")
            if chunk_index is not None:
                chunk_index = int(chunk_index)

            scope = payload.get("scope")
            inferred_source_type = source_type
            if scope == "conversation":
                inferred_source_type = SourceType.CONVERSATION.value
            elif scope == "matter":
                inferred_source_type = SourceType.MATTER.value

            source_block = payload.get("source") or {}
            pub_year = payload.get("publication_year") or source_block.get("publication_year")
            if pub_year is None:
                pub_year = payload.get("year")

            chunks.append(
                RetrievedChunk(
                    id=chunk_id,
                    score=float(point.score),
                    text=payload.get(
                        "text",
                        payload.get("chunk_text", ""),
                    ),
                    filename=payload.get("filename") or source_block.get("filename"),
                    title=payload.get("title"),
                    heading=payload.get("heading"),
                    document_id=normalize_document_id(
                        str(payload["document_id"])
                        if payload.get("document_id") is not None
                        else None
                    ),
                    law_name=payload.get("law_name"),
                    document_type=payload.get("document_type"),
                    category=payload.get("category"),
                    sub_category=payload.get("sub_category"),
                    practice_area=payload.get("practice_area"),
                    court=payload.get("court"),
                    year=pub_year,
                    publication_year=pub_year,
                    jurisdiction=(
                        payload.get("jurisdiction")
                        or payload.get("province")
                    ),
                    sections=payload.get("sections", []) or [],
                    keywords=payload.get("keywords", []) or [],
                    summary=payload.get("summary"),
                    source=source_type,
                    source_type=inferred_source_type,
                    scope=scope,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    parent_chunk_id=payload.get("parent_chunk_id"),
                    validation_status=payload.get("validation_status"),
                    chunk_role=payload.get("chunk_role"),
                    embedded=payload.get("embedded"),
                    page_number=payload.get("page_number"),
                    start_offset=payload.get("start_offset"),
                    end_offset=payload.get("end_offset"),
                    paragraph_index=payload.get("paragraph_index"),
                    section=payload.get("section"),
                    matter_id=(
                        str(payload["matter_id"])
                        if payload.get("matter_id") is not None
                        else None
                    ),
                    conversation_id=(
                        str(payload["conversation_id"])
                        if payload.get("conversation_id") is not None
                        else None
                    ),
                )
            )

        return chunks
