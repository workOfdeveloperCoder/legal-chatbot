"""Tests for conversation-scoped document detail API."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.models.document import DocumentScope
from tests.test_matter_document_delete import _build_service


CONVERSATION_ID = uuid.uuid4()
DOCUMENT_ID = uuid.uuid4()
USER_ID = uuid.uuid4()


def _conversation():
    return SimpleNamespace(
        id=CONVERSATION_ID,
        user_id=USER_ID,
    )


def _document():
    return SimpleNamespace(
        id=DOCUMENT_ID,
        owner_id=USER_ID,
        filename="notes.txt",
        scope=DocumentScope.CONVERSATION,
        conversation_id=CONVERSATION_ID,
        matter_id=None,
        mime_type="text/plain",
        extracted_text="Conversation-only document text.",
        processed=True,
        vectorized=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        storage_path=None,
    )


@pytest.mark.asyncio
async def test_get_conversation_document_returns_text():
    service = _build_service()
    service.conversations.get = AsyncMock(return_value=_conversation())
    service.repo.get_by_id = AsyncMock(return_value=_document())

    detail = await service.get_conversation_document(
        user_id=USER_ID,
        conversation_id=CONVERSATION_ID,
        document_id=DOCUMENT_ID,
    )

    assert detail.filename == "notes.txt"
    assert detail.conversation_id == CONVERSATION_ID
    assert detail.text == "Conversation-only document text."


@pytest.mark.asyncio
async def test_get_conversation_document_rejects_matter_scope():
    service = _build_service()
    service.conversations.get = AsyncMock(return_value=_conversation())
    wrong_scope = _document()
    wrong_scope.scope = DocumentScope.MATTER
    service.repo.get_by_id = AsyncMock(return_value=wrong_scope)

    with pytest.raises(HTTPException) as exc:
        await service.get_conversation_document(
            user_id=USER_ID,
            conversation_id=CONVERSATION_ID,
            document_id=DOCUMENT_ID,
        )

    assert exc.value.status_code == 404
