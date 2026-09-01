from __future__ import annotations

from app.rag.context_expansion import expand_parent_context
from app.rag.evidence import select_evidence
from app.rag.models import RetrievedChunk, RetrievalMetadata, RetrievalOutcome, SourceType
from app.rag.reranker import HybridReranker


def build_source_reference(chunk: RetrievedChunk) -> str | None:
    if chunk.source_reference:
        return chunk.source_reference

    parts: list[str] = []

    name = (
        chunk.display_name
        or chunk.filename
        or chunk.title
        or chunk.law_name
        or chunk.document_type
    )
    if name:
        parts.append(str(name))

    if chunk.sections:
        parts.append(f"Section {', '.join(chunk.sections)}")
    elif chunk.section:
        parts.append(f"Section {chunk.section}")

    if chunk.court:
        parts.append(str(chunk.court))

    if chunk.year:
        parts.append(str(chunk.year))

    if chunk.page_number:
        parts.append(f"page {chunk.page_number}")

    if getattr(chunk, "url", None):
        parts.append(str(chunk.url))

    if not parts and chunk.filename:
        parts.append(chunk.filename)

    return ", ".join(parts) if parts else None


def enrich_chunk_metadata(chunk: RetrievedChunk) -> RetrievedChunk:
    updates: dict = {}

    if chunk.source_type is None:
        if chunk.source in ("legal_corpus", "legal"):
            updates["source_type"] = SourceType.LEGAL.value
        elif chunk.scope == "conversation" or chunk.source == "conversation_document":
            updates["source_type"] = SourceType.CONVERSATION.value
        elif chunk.scope == "matter" or chunk.source == "matter_document":
            updates["source_type"] = SourceType.MATTER.value
        elif chunk.source == "web" or chunk.source_type == SourceType.WEB.value:
            updates["source_type"] = SourceType.WEB.value
        else:
            updates["source_type"] = SourceType.LEGAL.value

    if chunk.chunk_id is None:
        updates["chunk_id"] = chunk.id

    if chunk.source_reference is None:
        ref = build_source_reference(chunk)
        if ref:
            updates["source_reference"] = ref

    if updates:
        return chunk.model_copy(update=updates)

    return chunk


async def finalize_retrieval(
    *,
    query: str,
    chunks: list[RetrievedChunk],
    filters: dict[str, str] | None,
    metadata: RetrievalMetadata,
    limit: int,
    legal_limit: int,
    conversation_limit: int,
    matter_limit: int,
    document_task: bool = False,
) -> RetrievalOutcome:
    enriched = [enrich_chunk_metadata(chunk) for chunk in chunks]

    reranker = HybridReranker()
    ranked = await reranker.rerank(
        query=query,
        chunks=enriched,
        filters=filters,
        document_task=document_task,
    )

    selected = select_evidence(
        ranked,
        limit=limit,
        legal_limit=legal_limit,
        conversation_limit=conversation_limit,
        matter_limit=matter_limit,
    )

    selected = await expand_parent_context(selected)

    metadata.total_selected = len(selected)
    metadata.legal_chunks = sum(
        1 for c in selected if c.source_type == SourceType.LEGAL.value
    )
    metadata.conversation_chunks = sum(
        1 for c in selected if c.source_type == SourceType.CONVERSATION.value
    )
    metadata.matter_chunks = sum(
        1 for c in selected if c.source_type == SourceType.MATTER.value
    )
    metadata.web_chunks = sum(
        1 for c in selected if c.source_type == SourceType.WEB.value
    )

    return RetrievalOutcome(chunks=selected, metadata=metadata)
