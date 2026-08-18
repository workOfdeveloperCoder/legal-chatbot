from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.rag.models import RetrievedChunk, SourceType


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        raise NotImplementedError


class NoOpReranker(BaseReranker):
    async def rerank(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        return chunks


_DOCUMENT_TASKS = {"summarization", "document_qa"}


class HybridReranker(BaseReranker):
    """
    Lightweight hybrid reranker combining vector score, keyword overlap,
    metadata matches, and source authority tier boosts.

    Designed as a drop-in replacement for NoOpReranker without external deps.
    """

    TIER_BOOST = {
        SourceType.LEGAL.value: 0.20,
        SourceType.CONVERSATION.value: 0.08,
        SourceType.MATTER.value: 0.12,
    }

    DOCUMENT_TASK_TIER_BOOST = {
        SourceType.LEGAL.value: 0.0,
        SourceType.CONVERSATION.value: 0.15,
        SourceType.MATTER.value: 0.18,
    }

    async def rerank(
        self,
        *,
        query: str,
        chunks: list[RetrievedChunk],
        filters: dict[str, str] | None = None,
        document_task: bool = False,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        query_terms = _tokenize(query)
        tier_boosts = (
            self.DOCUMENT_TASK_TIER_BOOST
            if document_task
            else self.TIER_BOOST
        )

        scored: list[RetrievedChunk] = []
        for chunk in chunks:
            relevance = _score_chunk(
                chunk=chunk,
                query_terms=query_terms,
                filters=filters or {},
                tier_boosts=tier_boosts,
            )
            scored.append(
                chunk.model_copy(update={"relevance_score": relevance})
            )

        scored.sort(
            key=lambda item: item.relevance_score or 0.0,
            reverse=True,
        )
        return scored


def _score_chunk(
    *,
    chunk: RetrievedChunk,
    query_terms: set[str],
    filters: dict[str, str],
    tier_boosts: dict[str, float],
) -> float:
    base = float(chunk.score or 0.0)
    text = (chunk.text or "").lower()

    text_terms = _tokenize(text)
    overlap = len(query_terms & text_terms)
    keyword_boost = min(overlap * 0.03, 0.18)

    section_boost = 0.0
    section_filter = filters.get("section")
    if section_filter:
        section_filter = section_filter.lower()
        chunk_sections = [s.lower() for s in (chunk.sections or [])]
        if section_filter in chunk_sections or section_filter in text:
            section_boost = 0.15

    court_boost = 0.0
    court_filter = filters.get("court")
    if court_filter and chunk.court:
        if court_filter.lower() in chunk.court.lower():
            court_boost = 0.12

    year_boost = 0.0
    year_filter = filters.get("year")
    if year_filter and chunk.year is not None:
        if str(chunk.year) == str(year_filter):
            year_boost = 0.08

    exact_phrase_boost = 0.0
    for term in query_terms:
        if len(term) >= 5 and term in text:
            exact_phrase_boost = 0.05
            break

    source_type = chunk.source_type or SourceType.LEGAL.value
    tier_boost = tier_boosts.get(source_type, 0.0)

    return round(
        base
        + keyword_boost
        + section_boost
        + court_boost
        + year_boost
        + exact_phrase_boost
        + tier_boost,
        6,
    )


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3
    }
