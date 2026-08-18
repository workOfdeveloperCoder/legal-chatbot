"""Near-duplicate overlapping passage deduplication."""

from __future__ import annotations

from app.rag.evidence import (
    dedupe_near_duplicate_chunks,
    passages_are_near_duplicates,
)
from app.rag.models import RetrievedChunk, SourceType
from app.services.response_formatter import ResponseFormatter

DOC = "11111111-1111-1111-1111-111111111111"

OPENING = (
    "CLOG ON DISCRETION By M. Qasim Khan Khattak, Advocate, Karak, "
    "(N.-W.F.P.) Discretion makes its own place not necessarily in a "
    "given situation, is always obedient to a special statute or in "
    "other words a special statutory law can bar or act as clog on the "
    "judicial exercise of discretion. Taking example of section 54-C of "
    "Electricity Act, 1910, an Act got much familiarity in the present reign."
)


def test_passages_are_near_duplicates_for_overlap_windows():
    left = OPENING
    right = OPENING + " The consumers have been put by it to wholesale bills."
    assert passages_are_near_duplicates(left, right)


def test_dedupe_near_duplicate_chunks_keeps_highest_score():
    chunks = [
        RetrievedChunk(
            id=f"{DOC}:0",
            chunk_id=f"{DOC}:0",
            document_id=DOC,
            chunk_index=0,
            score=0.91,
            relevance_score=0.91,
            text=OPENING,
            filename="2000J8.txt",
            source_type=SourceType.MATTER.value,
            start_offset=0,
            end_offset=500,
        ),
        RetrievedChunk(
            id=f"{DOC}:1",
            chunk_id=f"{DOC}:1",
            document_id=DOC,
            chunk_index=1,
            score=0.89,
            relevance_score=0.89,
            text=OPENING + " Extra overlapping tail about WAPDA bills.",
            filename="2000J8.txt",
            source_type=SourceType.MATTER.value,
            start_offset=120,
            end_offset=620,
        ),
    ]

    deduped = dedupe_near_duplicate_chunks(chunks)
    assert len(deduped) == 1
    assert deduped[0].chunk_index == 0


def test_resource_evidence_drops_near_duplicate_passages():
    chunks = [
        RetrievedChunk(
            id=f"{DOC}:0",
            chunk_id=f"{DOC}:0",
            document_id=DOC,
            chunk_index=0,
            score=0.91,
            relevance_score=0.91,
            text=OPENING,
            filename="2000J8.txt",
            display_name="CLOG ON DISCRETION",
            source_type=SourceType.MATTER.value,
            start_offset=0,
            end_offset=500,
        ),
        RetrievedChunk(
            id=f"{DOC}:1",
            chunk_id=f"{DOC}:1",
            document_id=DOC,
            chunk_index=1,
            score=0.88,
            relevance_score=0.88,
            text=OPENING + " Continuing the same opening discussion.",
            filename="2000J8.txt",
            display_name="CLOG ON DISCRETION",
            source_type=SourceType.MATTER.value,
            start_offset=100,
            end_offset=600,
        ),
    ]

    result = ResponseFormatter().format(
        answer="According to the article [Source 1] [Source 2]",
        chunks=chunks,
        sources_used=[1, 2],
        build_resources=True,
    )

    assert len(result["resources"]) == 1
    assert len(result["resources"][0].evidence) == 1
