from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest

from app.services.prompt_context_service import PromptContextService


@pytest.mark.asyncio
async def test_build_continues_when_memory_embeddings_fail():
    repo = SimpleNamespace(
        list_by_conversation=AsyncMock(return_value=[]),
    )
    memory = SimpleNamespace(
        get_memories=AsyncMock(side_effect=ConnectionError("ollama down")),
    )
    service = PromptContextService(
        message_repository=repo,
        memory_service=memory,
    )
    ctx = await service.build(
        user=SimpleNamespace(id=uuid4()),
        conversation=SimpleNamespace(id=uuid4(), matter_id=None),
        query="What is section 54-C?",
    )
    assert ctx.history == []
    assert ctx.memories == []
