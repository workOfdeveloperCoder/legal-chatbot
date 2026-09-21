"""Public legal-corpus (library) document reads from Qdrant."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from fastapi import HTTPException, status
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.core.config import settings
from app.schemas.library import LibraryDocumentDetail
from app.vector.qdrant import qdrant_service

logger = logging.getLogger(__name__)

_DOC_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{4,128}$")


def _chunk_body(payload: dict) -> str:
    raw = payload.get("chunk_text") or payload.get("text") or payload.get("content") or ""
    return str(raw).strip()


def _looks_like_doc_start(text: str) -> bool:
    """True when chunk likely begins a document (not a mid-window cut)."""
    if not text:
        return False
    ch = text[0]
    if ch.islower():
        return False
    if ch.isupper():
        return True
    if ch in "'\"“‘" and len(text) > 1 and text[1].isupper():
        return True
    return False


def _overlap_join(left: str, right: str, *, min_overlap: int = 24) -> str | None:
    """
    Join ``right`` onto ``left`` when they share a sliding-window overlap.

    Returns None when no reliable overlap is found (caller may try other orders).
    """
    if not left:
        return right
    if not right:
        return left
    if right in left:
        return left
    if left in right:
        return right

    max_check = min(len(left), len(right), 8000)

    for size in range(max_check, min_overlap - 1, -1):
        if left[-size:] == right[:size]:
            return left + right[size:]

    # Mid-word cut: left "...poor person of…", right "erson of…".
    tail = left[-max_check:]
    for size in range(min(len(right), max_check), min_overlap - 1, -1):
        prefix = right[:size]
        idx = tail.rfind(prefix)
        if idx >= 0:
            return left[: len(left) - len(tail) + idx] + right

    return None


def _append_with_overlap(left: str, right: str, *, min_overlap: int = 24) -> str:
    joined = _overlap_join(left, right, min_overlap=min_overlap)
    if joined is not None:
        return joined
    return left + "\n\n" + right


def stitch_chunk_texts(texts: list[str]) -> str:
    """
    Merge overlapping retrieval windows into one full document.

    Picks a clean document-start chunk as the seed, then repeatedly extends
    forward/backward via overlap so mid-word windows (``erson…``) never lead.
    """
    cleaned: list[str] = []
    for text in texts:
        body = (text or "").strip()
        if body and body not in cleaned:
            cleaned.append(body)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]

    def seed_score(text: str) -> tuple[int, int, int]:
        return (
            1 if _looks_like_doc_start(text) else 0,
            0 if text[:1].islower() else 1,
            len(text),
        )

    used = [False] * len(cleaned)
    start = max(range(len(cleaned)), key=lambda i: seed_score(cleaned[i]))
    used[start] = True
    merged = cleaned[start]

    progress = True
    while progress:
        progress = False
        for i, body in enumerate(cleaned):
            if used[i]:
                continue
            appended = _overlap_join(merged, body)
            if appended is not None and len(appended) > len(merged):
                merged = appended
                used[i] = True
                progress = True
                continue
            prepended = _overlap_join(body, merged)
            if prepended is not None and len(prepended) > len(merged):
                merged = prepended
                used[i] = True
                progress = True

    for i, body in enumerate(cleaned):
        if not used[i]:
            merged = _append_with_overlap(merged, body)

    return merged.strip()


class LibraryDocumentService:
    """Rebuild full library document text from ``legal_documents`` chunks."""

    async def get_document(self, *, document_id: str) -> LibraryDocumentDetail:
        doc_id = (document_id or "").strip()
        if not doc_id or not _DOC_ID_RE.fullmatch(doc_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid library document id.",
            )

        points = await self._scroll_by_document_id(doc_id)
        if not points and doc_id.lower().endswith(".txt"):
            points = await self._scroll_by_filename(doc_id)

        if not points:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Library document not found.",
            )

        text, chunk_count = self._merge_points(points)
        if not text:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Library document has no readable text.",
            )

        # Prefer metadata from a parent / document-start chunk when present.
        meta_point = points[0]
        for point in points:
            payload = point.payload or {}
            role = str(payload.get("chunk_role") or "").lower()
            if payload.get("is_parent") is True or role == "parent":
                meta_point = point
                break
            if _looks_like_doc_start(_chunk_body(payload)):
                meta_point = point

        payload = meta_point.payload or {}
        filename = (
            payload.get("filename")
            or (payload.get("source") or {}).get("filename")
            or doc_id
        )
        title = payload.get("title") or filename
        now = datetime.now(timezone.utc)

        raw_year = payload.get("year") or payload.get("publication_year")
        year: int | None = None
        if raw_year is not None:
            try:
                year = int(raw_year)
            except (TypeError, ValueError):
                year = None

        return LibraryDocumentDetail(
            id=str(payload.get("document_id") or doc_id),
            filename=str(filename),
            title=str(title) if title else None,
            law_name=payload.get("law_name"),
            document_type=payload.get("document_type"),
            year=year,
            text=text,
            text_preview=text[:500],
            character_count=len(text),
            chunk_count=chunk_count,
            source_type="legal",
            created_at=now,
            updated_at=now,
        )

    async def _scroll_by_document_id(self, document_id: str):
        """Scroll all points for a document (Qdrant scroll is paginated)."""
        collected = []
        next_offset = None
        while True:
            points, next_offset = await qdrant_service.client.scroll(
                collection_name=settings.LEGAL_QDRANT_COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            collected.extend(points or [])
            if next_offset is None:
                break
        return collected

    async def _scroll_by_filename(self, filename: str):
        collected = []
        next_offset = None
        while True:
            points, next_offset = await qdrant_service.client.scroll(
                collection_name=settings.LEGAL_QDRANT_COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="filename",
                            match=MatchValue(value=filename),
                        )
                    ]
                ),
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            collected.extend(points or [])
            if next_offset is None:
                break
        return collected

    @staticmethod
    def _merge_points(points) -> tuple[str, int]:
        """
        Full file = every unique chunk merged with overlap collapsed.

        Do **not** drop children that share the same ``chunk_index`` (common on
        some server indexes) — that previously kept only the longest window
        and looked like “half the file”.
        """
        children: list[tuple[int, str]] = []
        parents: list[str] = []
        seen_bodies: set[str] = set()

        for point in points:
            payload = point.payload or {}
            body = _chunk_body(payload)
            if not body:
                continue
            # Prefer chunk_id identity when present so equal indexes stay distinct.
            chunk_id = str(payload.get("chunk_id") or "").strip()
            dedupe_key = chunk_id or body
            if dedupe_key in seen_bodies:
                continue
            seen_bodies.add(dedupe_key)

            index = int(payload.get("chunk_index") or 0)
            is_parent = payload.get("is_parent") is True
            role = str(payload.get("chunk_role") or "").lower()
            if is_parent or role == "parent":
                parents.append(body)
            else:
                children.append((index, body))

        # Keep every child; sort by index then by length (stable coverage order).
        ordered_children = [
            body
            for _index, body in sorted(children, key=lambda item: (item[0], -len(item[1])))
        ]

        candidates: list[tuple[str, int]] = []

        if ordered_children:
            stitched = stitch_chunk_texts(ordered_children)
            if stitched:
                candidates.append((stitched, len(ordered_children)))

        all_bodies = [*parents, *ordered_children]
        if len(all_bodies) > 1:
            stitched_all = stitch_chunk_texts(all_bodies)
            if stitched_all:
                candidates.append((stitched_all, len(all_bodies)))

        for parent in parents:
            candidates.append((parent, 1))

        if not candidates:
            return "", 0

        def result_score(item: tuple[str, int]) -> tuple[int, int, int]:
            text, _count = item
            # Longest clean-start reconstruction wins (never prefer a short parent).
            return (
                1 if _looks_like_doc_start(text) else 0,
                0 if text[:1].islower() else 1,
                len(text),
            )

        best_text, best_count = max(candidates, key=result_score)
        logger.info(
            "library_document_merge points=%s children=%s parents=%s chars=%s",
            len(points),
            len(ordered_children),
            len(parents),
            len(best_text),
        )
        return best_text.strip(), best_count
