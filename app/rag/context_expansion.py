"""Expand child retrieval hits with parent context from Qdrant."""

from __future__ import annotations

import logging

from app.rag.models import RetrievedChunk
from app.vector.qdrant import qdrant_service

logger = logging.getLogger(__name__)


def _payload_text(payload: dict) -> str:
    return (payload.get("chunk_text") or payload.get("text") or "").strip()


def _extract_parent_id(chunk: RetrievedChunk) -> str | None:
    if chunk.parent_chunk_id:
        return chunk.parent_chunk_id
    return None


async def expand_parent_context(
    chunks: list[RetrievedChunk],
    *,
    collection: str | None = None,
) -> list[RetrievedChunk]:
    """
    Resolve parent_chunk_id for legal corpus children.

    Prepends parent text when the child is short or structurally incomplete.
    """
    if not chunks:
        return chunks

    collection_name = collection or qdrant_service.legal_collection
    parent_ids = {
        pid
        for chunk in chunks
        if (pid := _extract_parent_id(chunk))
    }
    if not parent_ids:
        return chunks

    try:
        records = await qdrant_service.client.retrieve(
            collection_name=collection_name,
            ids=list(parent_ids),
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        logger.exception("parent context retrieve failed")
        return chunks

    parent_map = {
        str(record.id): record.payload or {}
        for record in records
    }

    expanded: list[RetrievedChunk] = []
    for chunk in chunks:
        parent_id = _extract_parent_id(chunk)
        if not parent_id or parent_id not in parent_map:
            expanded.append(chunk)
            continue

        parent_payload = parent_map[parent_id]
        parent_text = _payload_text(parent_payload)
        chunk.parent_chunk_id = parent_id
        chunk.parent_text = parent_text

        child_text = (chunk.text or "").strip()
        needs_expansion = (
            len(child_text) < 120
            or chunk.validation_status in {"orphan_heading", "micro_heading", "decorative"}
        )
        if needs_expansion and parent_text and parent_text not in child_text:
            merged = f"{parent_text}\n\n{child_text}".strip()
            chunk.expanded_text = merged
            chunk.text = merged
        else:
            chunk.expanded_text = child_text

        expanded.append(chunk)

    return expanded
