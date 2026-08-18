"""Tests for document and chunk identity normalization."""

import uuid

from app.rag.document_identity import (
    chunk_identity_key,
    normalize_document_id,
    recover_document_id,
    resource_identity_key,
)
from app.rag.evidence import dedupe_chunks
from app.rag.models import RetrievedChunk, SourceType
from app.services.response_formatter import ResponseFormatter


DOC = "11111111-1111-1111-1111-111111111111"


def test_normalize_document_id_uuid_formats():
    raw = "11111111-1111-1111-1111-111111111111"
    assert normalize_document_id(raw) == raw.lower()
    assert normalize_document_id(raw.upper()) == raw.lower()
    assert (
        normalize_document_id("11111111111111111111111111111111")
        == raw.lower()
    )


def test_resource_identity_uses_document_id_only():
    assert resource_identity_key(
        document_id=DOC,
        chunk_id="doc:0",
    ) == f"doc:{DOC.lower()}"
    assert resource_identity_key(
        document_id=DOC,
        chunk_id="doc:1",
    ) == resource_identity_key(document_id=DOC, chunk_id="doc:9")


def test_resource_identity_keeps_same_filename_separate_by_document_id():
    key_a = resource_identity_key(
        document_id=DOC,
        filename="2000J8.txt",
        source_type="matter",
        chunk_id="a",
    )
    key_b = resource_identity_key(
        document_id="22222222-2222-2222-2222-222222222222",
        filename="2000J8.TXT",
        source_type="conversation",
        chunk_id="b",
    )
    assert key_a != key_b
    assert key_a.startswith("doc:")
    assert key_b.startswith("doc:")


def test_dedupe_chunks_same_document_and_chunk_id():
    chunk_a = RetrievedChunk(
        id=f"{DOC}:0",
        chunk_id=f"{DOC}:0",
        document_id=DOC,
        chunk_index=0,
        score=0.7,
        text="Same text",
        source_type=SourceType.MATTER.value,
    )
    chunk_b = RetrievedChunk(
        id=f"{DOC}:0-dup",
        chunk_id=f"{DOC}:0",
        document_id=str(uuid.UUID(DOC)).upper(),
        chunk_index=0,
        score=0.9,
        text="Same text",
        source_type=SourceType.MATTER.value,
    )

    deduped = dedupe_chunks([chunk_a, chunk_b])
    assert len(deduped) == 1
    assert deduped[0].score == 0.9


def test_formatter_dedupes_duplicate_chunk_in_sources_and_resources():
    chunks = [
        RetrievedChunk(
            id=f"{DOC}:0",
            chunk_id=f"{DOC}:0",
            document_id=DOC,
            chunk_index=0,
            score=0.91,
            relevance_score=0.91,
            text="Section 54-C discussion",
            filename="2000J8.txt",
            source_type=SourceType.MATTER.value,
        ),
        RetrievedChunk(
            id=f"{DOC}:0-dup",
            chunk_id=f"{DOC}:0",
            document_id=str(uuid.UUID(DOC)).upper(),
            chunk_index=0,
            score=0.90,
            relevance_score=0.90,
            text="Section 54-C discussion",
            filename="2000J8.txt",
            source_type=SourceType.MATTER.value,
        ),
    ]

    formatter = ResponseFormatter()
    result = formatter.format(
        answer="Answer [Source 1] [Source 2]",
        chunks=chunks,
        sources_used=[1, 2],
        build_resources=True,
    )

    assert len(result["resources"]) == 1
    assert len(result["sources"]) == 1
    assert len(result["resources"][0].evidence) == 1


def test_recover_document_id_from_chunk_prefix():
    assert recover_document_id(chunk_id=f"{DOC}:3") == DOC.lower()
    assert (
        recover_document_id(
            document_id=None,
            chunk_id=f"{DOC.upper()}:0",
        )
        == DOC.lower()
    )


def test_chunk_identity_key_prefers_document_and_chunk_id():
    key = chunk_identity_key(
        document_id=DOC,
        chunk_id=f"{DOC}:2",
        chunk_index=2,
    )
    assert key == f"doc:{DOC.lower()}:chunk:{DOC.lower()}:2"
