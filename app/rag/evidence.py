from __future__ import annotations

import re

from app.rag.document_identity import recover_document_id
from app.rag.models import RetrievedChunk, SourceType


def normalize_passage_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = text.lower().replace("‑", "-").replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def passages_are_near_duplicates(
    left: str | None,
    right: str | None,
    *,
    threshold: float = 0.82,
) -> bool:
    """
    Detect overlapping chunk windows that look like the same passage.

    Common when chunk_overlap causes two adjacent chunks to share a long
    opening stretch (title + author + start of article).
    """
    a = normalize_passage_text(left)
    b = normalize_passage_text(right)
    if not a or not b:
        return False
    if a == b:
        return True

    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) < 60:
        return shorter in longer

    prefix_len = min(180, len(shorter))
    if longer.startswith(shorter[:prefix_len]) or shorter[:prefix_len] in longer:
        overlap = _token_overlap_ratio(shorter, longer)
        return overlap >= threshold

    return _token_overlap_ratio(a, b) >= max(threshold, 0.90)


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return intersection / union if union else 0.0


def offsets_overlap(
    *,
    left_start: int | None,
    left_end: int | None,
    right_start: int | None,
    right_end: int | None,
    min_ratio: float = 0.55,
) -> bool:
    if (
        left_start is None
        or left_end is None
        or right_start is None
        or right_end is None
    ):
        return False
    if left_end <= left_start or right_end <= right_start:
        return False

    overlap_start = max(left_start, right_start)
    overlap_end = min(left_end, right_end)
    overlap = max(0, overlap_end - overlap_start)
    if overlap <= 0:
        return False

    shorter = min(left_end - left_start, right_end - right_start)
    return (overlap / shorter) >= min_ratio


def dedupe_near_duplicate_chunks(
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Keep highest-scoring chunk when passages heavily overlap.

    Survivors are returned in the original input order so retrieval
    tier priority is preserved.
    """
    if len(chunks) <= 1:
        return chunks

    ranked = sorted(
        enumerate(chunks),
        key=lambda pair: _chunk_score(pair[1]),
        reverse=True,
    )
    kept_indexes: set[int] = set()
    kept_chunks: list[RetrievedChunk] = []

    for index, chunk in ranked:
        duplicate = False
        for existing in kept_chunks:
            same_doc = False
            left_doc = recover_document_id(
                document_id=existing.document_id,
                chunk_id=existing.chunk_id or existing.id,
            )
            right_doc = recover_document_id(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id or chunk.id,
            )
            if left_doc and right_doc and left_doc == right_doc:
                same_doc = True
            elif (
                existing.filename
                and chunk.filename
                and existing.filename.strip().lower()
                == chunk.filename.strip().lower()
            ):
                same_doc = True

            if not same_doc:
                continue

            if offsets_overlap(
                left_start=existing.start_offset,
                left_end=existing.end_offset,
                right_start=chunk.start_offset,
                right_end=chunk.end_offset,
            ) or passages_are_near_duplicates(existing.text, chunk.text):
                duplicate = True
                break

        if not duplicate:
            kept_indexes.add(index)
            kept_chunks.append(chunk)

    return [chunk for index, chunk in enumerate(chunks) if index in kept_indexes]


def dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Remove duplicate chunks while keeping the highest-scoring copy.

    Dedup keys (in order):
      1. chunk_id
      2. document_id + chunk_index
      3. id
    Then drop near-duplicate overlapping passages.
    """
    best: dict[str, RetrievedChunk] = {}

    for chunk in chunks:
        key = _dedupe_key(chunk)
        existing = best.get(key)
        if existing is None or _chunk_score(chunk) > _chunk_score(existing):
            best[key] = chunk

    ranked = sorted(best.values(), key=_chunk_score, reverse=True)
    return dedupe_near_duplicate_chunks(ranked)


def select_evidence(
    chunks: list[RetrievedChunk],
    *,
    limit: int,
    legal_limit: int,
    conversation_limit: int,
    matter_limit: int,
) -> list[RetrievedChunk]:
    """
    Select evidence preserving retrieval priority tiers.

    Priority order in the final bundle:
      legal → conversation → matter
    """
    if not chunks:
        return []

    by_type: dict[str, list[RetrievedChunk]] = {
        SourceType.LEGAL.value: [],
        SourceType.CONVERSATION.value: [],
        SourceType.MATTER.value: [],
        SourceType.WEB.value: [],
    }

    for chunk in sorted(chunks, key=_chunk_score, reverse=True):
        source_type = chunk.source_type or SourceType.LEGAL.value
        if source_type not in by_type:
            source_type = SourceType.LEGAL.value
        by_type[source_type].append(chunk)

    selected: list[RetrievedChunk] = []
    web_limit = max(0, min(4, limit - legal_limit))
    tier_limits = (
        (SourceType.LEGAL.value, legal_limit),
        (SourceType.CONVERSATION.value, conversation_limit),
        (SourceType.MATTER.value, matter_limit),
        (SourceType.WEB.value, web_limit or 3),
    )

    for source_type, tier_limit in tier_limits:
        tier_selected = dedupe_chunks(by_type[source_type][: tier_limit * 2])
        selected.extend(tier_selected[:tier_limit])

    selected = _dedupe_preserve_order(selected)
    selected = dedupe_near_duplicate_chunks(selected)
    return selected[:limit]


def _dedupe_preserve_order(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Remove duplicate chunks while preserving the incoming order."""
    best: dict[str, RetrievedChunk] = {}
    order: list[str] = []

    for chunk in chunks:
        key = _dedupe_key(chunk)
        if key not in best:
            best[key] = chunk
            order.append(key)
            continue

        if _chunk_score(chunk) > _chunk_score(best[key]):
            best[key] = chunk

    return [best[key] for key in order]


def _dedupe_key(chunk: RetrievedChunk) -> str:
    if chunk.chunk_id:
        doc = recover_document_id(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
        )
        if doc:
            return f"chunk:{doc}:{chunk.chunk_id.strip().lower()}"
        return f"chunk:{chunk.chunk_id.strip().lower()}"

    doc = recover_document_id(
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id or chunk.id,
    )
    if doc is not None and chunk.chunk_index is not None:
        return f"doc:{doc}:{chunk.chunk_index}"

    return f"id:{chunk.id}"


def _chunk_score(chunk: RetrievedChunk) -> float:
    if chunk.relevance_score is not None:
        return float(chunk.relevance_score)
    return float(chunk.score or 0.0)
