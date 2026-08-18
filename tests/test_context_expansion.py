"""Tests for parent context expansion."""

import pytest

from app.rag.context_expansion import expand_parent_context
from app.rag.models import RetrievedChunk


@pytest.mark.asyncio
async def test_expand_parent_context_noop_without_parent(monkeypatch):
    chunk = RetrievedChunk(
        id="c1",
        score=0.9,
        text="Standalone chunk text.",
        parent_chunk_id=None,
    )
    result = await expand_parent_context([chunk])
    assert len(result) == 1
    assert result[0].text == "Standalone chunk text."


@pytest.mark.asyncio
async def test_expand_short_child_with_parent(monkeypatch):
    parent_id = "parent-uuid-1"

    class FakeRecord:
        id = parent_id
        payload = {"chunk_text": "PRINCIPLE OF REDRESS full context paragraph."}

    class FakeClient:
        async def retrieve(self, **kwargs):
            return [FakeRecord()]

    class FakeService:
        legal_collection = "legal_documents"
        client = FakeClient()

    monkeypatch.setattr(
        "app.rag.context_expansion.qdrant_service",
        FakeService(),
    )

    chunk = RetrievedChunk(
        id="c1",
        score=0.9,
        text="SOCIAL",
        parent_chunk_id=parent_id,
        validation_status="orphan_heading",
    )
    result = await expand_parent_context([chunk])
    assert "PRINCIPLE OF REDRESS" in result[0].text
    assert "SOCIAL" in result[0].text
