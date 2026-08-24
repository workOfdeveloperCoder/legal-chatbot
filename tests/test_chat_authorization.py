from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.schemas.chat import ChatRequest
from app.services.chat_service import ChatService


def _service(conversation_service) -> ChatService:
    return ChatService(
        rag_service=MagicMock(),
        conversation_service=conversation_service,
        memory_service=MagicMock(),
        prompt_context_service=MagicMock(),
        query_router=MagicMock(),
        document_repository=MagicMock(),
        activity_log_service=MagicMock(),
    )


@pytest.mark.asyncio
async def test_require_matter_access_rejects_other_owner():
    user = SimpleNamespace(id=uuid4())
    matter_id = uuid4()
    conversation_service = MagicMock()
    conversation_service.matters.get = AsyncMock(
        return_value=SimpleNamespace(id=matter_id, owner_id=uuid4()),
    )
    service = _service(conversation_service)
    with pytest.raises(HTTPException) as exc:
        await service._require_matter_access(user=user, matter_id=matter_id)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_require_matter_access_allows_owner():
    user = SimpleNamespace(id=uuid4())
    matter_id = uuid4()
    conversation_service = MagicMock()
    conversation_service.matters.get = AsyncMock(
        return_value=SimpleNamespace(id=matter_id, owner_id=user.id),
    )
    service = _service(conversation_service)
    await service._require_matter_access(user=user, matter_id=matter_id)


@pytest.mark.asyncio
async def test_chat_does_not_attach_foreign_matter_to_existing_conversation():
    user = SimpleNamespace(id=uuid4())
    foreign_matter = uuid4()
    conversation = SimpleNamespace(id=uuid4(), matter_id=None)
    conversation_service = MagicMock()
    conversation_service.get_or_create = AsyncMock(return_value=conversation)
    conversation_service.matters.get = AsyncMock(
        return_value=SimpleNamespace(id=foreign_matter, owner_id=uuid4()),
    )
    conversation_service.db = MagicMock()
    conversation_service.db.commit = AsyncMock()
    conversation_service.db.refresh = AsyncMock()

    service = _service(conversation_service)
    payload = ChatRequest(
        message="What is section 302 PPC?",
        conversation_id=conversation.id,
        matter_id=foreign_matter,
    )
    with pytest.raises(HTTPException) as exc:
        await service.chat(user=user, payload=payload)
    assert exc.value.status_code == 403
    conversation_service.db.commit.assert_not_awaited()
    assert conversation.matter_id is None
