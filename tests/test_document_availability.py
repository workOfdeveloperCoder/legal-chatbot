from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.document import DocumentScope
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import DocumentResponse
from app.vector.filters import QdrantFilterBuilder


def _document_payload(
    *,
    processed: bool = False,
    vectorized: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        filename="test.pdf",
        mime_type="application/pdf",
        storage_path="/tmp/test.pdf",
        scope=DocumentScope.MATTER.value,
        matter_id=uuid.uuid4(),
        conversation_id=None,
        processed=processed,
        vectorized=vectorized,
        created_at=datetime.now(timezone.utc),
    )


class TestDocumentResponse:
    def test_ready_for_qa_when_fully_indexed(self):
        doc = _document_payload(processed=True, vectorized=True)
        response = DocumentResponse.model_validate(doc)
        assert response.ready_for_qa is True

    def test_ready_for_qa_false_when_unprocessed(self):
        doc = _document_payload(processed=False, vectorized=False)
        response = DocumentResponse.model_validate(doc)
        assert response.ready_for_qa is False

    def test_ready_for_qa_false_when_processed_but_not_vectorized(self):
        doc = _document_payload(processed=True, vectorized=False)
        response = DocumentResponse.model_validate(doc)
        assert response.ready_for_qa is False


class TestHasForScope:
    @pytest.mark.asyncio
    async def test_returns_true_only_for_ready_documents(self):
        db = AsyncMock()
        db.execute = AsyncMock(
            return_value=MagicMock(
                scalar_one_or_none=MagicMock(return_value=uuid.uuid4())
            )
        )
        repo = DocumentRepository(db)

        matter_id = uuid.uuid4()
        conversation_id = uuid.uuid4()
        owner_id = uuid.uuid4()

        assert await repo.has_for_scope(
            owner_id=owner_id,
            matter_id=matter_id,
            conversation_id=conversation_id,
        ) is True

        db.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_no_ready_documents(self):
        db = AsyncMock()
        db.execute = AsyncMock(
            return_value=MagicMock(
                scalar_one_or_none=MagicMock(return_value=None)
            )
        )
        repo = DocumentRepository(db)

        assert await repo.has_for_scope(
            owner_id=uuid.uuid4(),
            matter_id=uuid.uuid4(),
        ) is False

    @pytest.mark.asyncio
    async def test_returns_false_without_scope(self):
        db = AsyncMock()
        repo = DocumentRepository(db)

        assert await repo.has_for_scope(owner_id=uuid.uuid4()) is False
        db.execute.assert_not_called()


class TestQdrantDocumentVisibility:
    def test_matter_document_visible_to_conversations_in_same_matter(self):
        user_id = str(uuid.uuid4())
        matter_a = str(uuid.uuid4())
        conversation_1 = str(uuid.uuid4())

        filt = QdrantFilterBuilder.document_visibility(
            user_id=user_id,
            matter_id=matter_a,
            conversation_id=conversation_1,
        )

        payload = {
            "user_id": user_id,
            "scope": "matter",
            "matter_id": matter_a,
            "conversation_id": None,
        }

        assert self._payload_matches_filter(payload, filt) is True

    def test_matter_document_not_visible_to_other_matter(self):
        user_id = str(uuid.uuid4())
        matter_a = str(uuid.uuid4())
        matter_b = str(uuid.uuid4())
        conversation_3 = str(uuid.uuid4())

        filt = QdrantFilterBuilder.document_visibility(
            user_id=user_id,
            matter_id=matter_b,
            conversation_id=conversation_3,
        )

        payload = {
            "user_id": user_id,
            "scope": "matter",
            "matter_id": matter_a,
            "conversation_id": None,
        }

        assert self._payload_matches_filter(payload, filt) is False

    def test_conversation_document_isolated_between_conversations(self):
        user_id = str(uuid.uuid4())
        matter_a = str(uuid.uuid4())
        conversation_1 = str(uuid.uuid4())
        conversation_2 = str(uuid.uuid4())

        filt = QdrantFilterBuilder.document_visibility(
            user_id=user_id,
            matter_id=matter_a,
            conversation_id=conversation_2,
        )

        payload = {
            "user_id": user_id,
            "scope": "conversation",
            "matter_id": None,
            "conversation_id": conversation_1,
        }

        assert self._payload_matches_filter(payload, filt) is False

    @staticmethod
    def _payload_matches_filter(payload: dict, filt) -> bool:
        """
        Minimal structural check: matter branch requires scope+matter_id,
        conversation branch requires scope+conversation_id.
        """
        user_ok = payload.get("user_id") is not None

        matter_ok = (
            payload.get("scope") == "matter"
            and payload.get("matter_id")
            in {
                cond.match.value
                for branch in filt.min_should.conditions
                for cond in branch.must
                if cond.key == "matter_id"
            }
        )

        conversation_ok = (
            payload.get("scope") == "conversation"
            and payload.get("conversation_id")
            in {
                cond.match.value
                for branch in filt.min_should.conditions
                for cond in branch.must
                if cond.key == "conversation_id"
            }
        )

        return user_ok and (matter_ok or conversation_ok)
