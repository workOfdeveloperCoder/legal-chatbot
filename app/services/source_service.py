from __future__ import annotations

from collections import OrderedDict
from typing import Any

from app.rag.models import RetrievedChunk


class SourceService:
    """
    Builds, validates and formats legal sources.

    Responsibilities
    ----------------
    - Deduplicate sources
    - Sort by relevance
    - Build API response
    - Build LLM citations
    - Build readable citations
    """

    def build(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[dict[str, Any]]:

        sources: OrderedDict[str, dict[str, Any]] = OrderedDict()

        for chunk in sorted(
            chunks,
            key=lambda x: x.score,
            reverse=True,
        ):

            key = self._key(chunk)

            if key in sources:
                continue

            sources[key] = {
                "id": chunk.id,
                "document_id": chunk.document_id,

                "filename": chunk.filename,
                "title": chunk.title,
                "heading": chunk.heading,

                "law_name": chunk.law_name,
                "document_type": chunk.document_type,

                "category": chunk.category,
                "sub_category": chunk.sub_category,
                "practice_area": chunk.practice_area,

                "court": chunk.court,
                "year": chunk.year,
                "jurisdiction": chunk.jurisdiction,

                "sections": chunk.sections,
                "keywords": chunk.keywords,
                "summary": chunk.summary,

                "excerpt": (chunk.text or "").strip()[:800] or None,

                "score": round(chunk.score, 4),
            }

        return list(sources.values())


    def build_inline_citations(
        self,
        chunks: list[RetrievedChunk],
    ) -> str:

        citations: list[str] = []

        seen: set[str] = set()

        for chunk in chunks:

            label = self._citation(chunk)

            if not label:
                continue

            if label in seen:
                continue

            seen.add(label)
            citations.append(label)

        return "; ".join(citations)


    def top_sources(
        self,
        chunks: list[RetrievedChunk],
        limit: int = 5,
    ) -> list[RetrievedChunk]:

        return sorted(
            chunks,
            key=lambda c: c.score,
            reverse=True,
        )[:limit]


    @staticmethod
    def _citation(
        chunk: RetrievedChunk,
    ) -> str:

        parts: list[str] = []

        if chunk.title:
            parts.append(chunk.title)

        elif chunk.law_name:
            parts.append(chunk.law_name)

        elif chunk.filename:
            parts.append(chunk.filename)


        if chunk.sections:
            parts.append(
                f"Sections: {', '.join(chunk.sections)}"
            )


        if chunk.court:
            parts.append(
                chunk.court
            )


        if chunk.year:
            parts.append(
                str(chunk.year)
            )


        return " | ".join(parts)


    @staticmethod
    def _key(
        chunk: RetrievedChunk,
    ) -> str:

        return "|".join(
            [
                chunk.filename or "",
                chunk.title or "",
                chunk.law_name or "",
                ",".join(chunk.sections),
            ]
        )