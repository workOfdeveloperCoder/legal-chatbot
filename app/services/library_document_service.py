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
        Prefer child chunks in index order (parents often duplicate them).
        Fall back to parent text when no children exist.
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

        ordered = sorted(children or parents, key=lambda item: item[0])
        parts: list[str] = []
        for _index, body in ordered:
            if parts and (body in parts[-1] or parts[-1] in body):
                if len(body) > len(parts[-1]):
                    parts[-1] = body
                continue
            parts.append(body)

        return "\n\n".join(parts).strip(), len(ordered)
