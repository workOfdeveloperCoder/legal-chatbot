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


def _append_with_overlap(left: str, right: str, *, min_overlap: int = 40) -> str:
    """
    Stitch two chunks that share a sliding-window overlap.

    Child chunks often cut mid-word, e.g. left ends with ``...poor person of…``
    and right starts with ``erson of…``.
    """
    if not left:
        return right
    if not right:
        return left
    if right in left:
        return left
    if left in right:
        return right

    max_check = min(len(left), len(right), 4000)

    # Exact suffix/prefix overlap.
    for size in range(max_check, min_overlap - 1, -1):
        if left[-size:] == right[:size]:
            return left + right[size:]

    # Mid-word / soft overlap: longest prefix of ``right`` found in the tail of ``left``.
    tail = left[-max_check:]
    best_idx = -1
    best_size = 0
    # Cap prefix search; still long enough for reliable overlaps.
    for size in range(min(len(right), max_check), min_overlap - 1, -1):
        prefix = right[:size]
        idx = tail.rfind(prefix)
        if idx >= 0:
            best_idx = idx
            best_size = size
            break

    if best_idx >= 0 and best_size >= min_overlap:
        return left[: len(left) - len(tail) + best_idx] + right

    return left + "\n\n" + right


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

        payload = points[0].payload or {}
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
        points, _ = await qdrant_service.client.scroll(
            collection_name=settings.LEGAL_QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        return points

    async def _scroll_by_filename(self, filename: str):
        points, _ = await qdrant_service.client.scroll(
            collection_name=settings.LEGAL_QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=filename),
                    )
                ]
            ),
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        return points

    @staticmethod
    def _merge_points(points) -> tuple[str, int]:
        """
        Rebuild document text from chunks.

        Prefer a parent chunk when it already holds the full document.
        Otherwise stitch child chunks in order, collapsing sliding-window overlaps.
        """
        children: list[tuple[int, str]] = []
        parents: list[tuple[int, str]] = []

        for point in points:
            payload = point.payload or {}
            body = _chunk_body(payload)
            if not body:
                continue
            index = int(payload.get("chunk_index") or 0)
            is_parent = payload.get("is_parent") is True
            role = str(payload.get("chunk_role") or "").lower()
            if is_parent or role == "parent":
                parents.append((index, body))
            else:
                children.append((index, body))

        if parents:
            longest_parent = max(parents, key=lambda item: len(item[1]))
            longest_child_len = max((len(body) for _i, body in children), default=0)
            # Parent usually stores the full article; children are overlapping windows.
            if len(longest_parent[1]) >= longest_child_len:
                return longest_parent[1], 1

        ordered = sorted(children or parents, key=lambda item: item[0])
        if not ordered:
            return "", 0

        merged = ordered[0][1]
        for _index, body in ordered[1:]:
            merged = _append_with_overlap(merged, body)

        return merged.strip(), len(ordered)
