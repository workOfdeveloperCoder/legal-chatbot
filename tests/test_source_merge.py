"""Tests for merging duplicate sources by document."""

from app.rag.models import RetrievedChunk, SourceType
from app.services.response_formatter import ResponseFormatter

DOC_A = "doc-a"
DOC_B = "doc-b"


def _chunk(document_id: str, filename: str, index: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{document_id}:{index}",
        chunk_id=f"{document_id}:{index}",
        score=score,
        relevance_score=score,
        text=f"Chunk {index} from {filename}",
        filename=filename,
        document_id=document_id,
        chunk_index=index,
        source_type=SourceType.MATTER.value,
        start_offset=index * 100,
        end_offset=(index + 1) * 100,
    )


def test_resources_merge_by_document_id():
    chunks = [
        _chunk(DOC_A, "2000J8.txt", 0, 0.88),
        _chunk(DOC_A, "2000J8.txt", 1, 0.86),
        _chunk(DOC_B, "2002J18.txt", 0, 0.80),
    ]
    formatter = ResponseFormatter()
    result = formatter.format(
        answer="Answer [Source 1] [Source 2] [Source 3]",
        chunks=chunks,
        sources_used=[1, 2, 3],
        build_resources=True,
    )

    assert len(result["resources"]) == 2
    assert len(result["sources"]) == 3


def test_resource_keeps_all_evidence_highlights():
    chunks = [
        _chunk(DOC_A, "2000J8.txt", 0, 0.70),
        _chunk(DOC_A, "2000J8.txt", 1, 0.91),
    ]
    formatter = ResponseFormatter()
    result = formatter.format(
        answer="Answer [Source 1] [Source 2]",
        chunks=chunks,
        sources_used=[1, 2],
        build_resources=True,
    )

    resource = result["resources"][0]
    assert resource.relevance == 0.91
    assert len(resource.evidence) == 2
    assert resource.evidence[0].chunk_index in (0, 1)


def test_display_sources_not_merged_into_one_chunk():
    chunks = [
        _chunk(DOC_A, "2000J8.txt", 0, 0.88),
        _chunk(DOC_A, "2000J8.txt", 1, 0.86),
        _chunk(DOC_B, "2002J18.txt", 0, 0.80),
    ]
    formatter = ResponseFormatter()
    raw = formatter._build_sources(chunks)
    displayed = formatter._filter_sources_for_display(raw, sources_used=[1, 2, 3])
    resources = formatter._build_resources(displayed)

    assert len(displayed) == 3
    assert len(resources) == 2
