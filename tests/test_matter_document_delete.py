from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.models.document import DocumentScope
from app.services.document_service import DocumentService


def _matter(*, owner_id: uuid.UUID) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        owner_id=owner_id,
    )


def _document(
    *,
    owner_id: uuid.UUID,
    matter_id: uuid.UUID,
    scope: DocumentScope = DocumentScope.MATTER,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        filename="2002J18.txt",
        storage_path="/tmp/2002J18.txt",
        owner_id=owner_id,
        matter_id=matter_id,
        scope=scope,
        extracted_text="Sample text",
        vectorized=True,
        processed=True,
        mime_type="text/plain",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _build_service(**overrides) -> DocumentService:
    db = AsyncMock()
    db.commit = AsyncMock()
    repo = AsyncMock()
    matters = AsyncMock()
    vector_service = AsyncMock()
    vector_service.delete_document = AsyncMock()

    service = DocumentService(
        db=db,
        repository=repo,
        matter_repository=matters,
        conversation_repository=AsyncMock(),
        extractor=MagicMock(),
        vector_service=vector_service,
        activity_log_service=AsyncMock(),
    )

    for key, value in overrides.items():
        setattr(service, key, value)

    return service


class TestDeleteMatterDocument:
    @pytest.mark.asyncio
    async def test_deletes_from_qdrant_postgres_and_storage(self, tmp_path):
        user_id = uuid.uuid4()
        matter_id = uuid.uuid4()
        document = _document(owner_id=user_id, matter_id=matter_id)

        storage_file = tmp_path / "2002J18.txt"
        storage_file.write_text("content")
        document.storage_path = str(storage_file)

        service = _build_service()
        service.matters.get = AsyncMock(
            return_value=_matter(owner_id=user_id)
        )
        service.repo.get_by_id = AsyncMock(return_value=document)
        service.repo.delete = AsyncMock()

        await service.delete_matter_document(
            user_id=user_id,
            matter_id=matter_id,
            document_id=document.id,
        )

        service.vector_service.delete_document.assert_awaited_once_with(
            document_id=str(document.id),
            owner_id=str(user_id),
        )
        service.repo.delete.assert_awaited_once_with(document)
        service.db.commit.assert_awaited_once()
        assert not storage_file.exists()

    @pytest.mark.asyncio
    async def test_qdrant_failure_still_deletes_postgres(self):
        user_id = uuid.uuid4()
        matter_id = uuid.uuid4()
        document = _document(owner_id=user_id, matter_id=matter_id)

        service = _build_service()
        service.matters.get = AsyncMock(
            return_value=_matter(owner_id=user_id)
        )
        service.repo.get_by_id = AsyncMock(return_value=document)
        service.repo.delete = AsyncMock()
        service.vector_service.delete_document = AsyncMock(
            side_effect=RuntimeError("qdrant unavailable")
        )

        with patch.object(
            DocumentService,
            "_delete_storage_file",
            return_value=None,
        ):
            await service.delete_matter_document(
                user_id=user_id,
                matter_id=matter_id,
                document_id=document.id,
            )

        service.repo.delete.assert_awaited_once_with(document)
        service.db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rejects_non_matter_scoped_document(self):
        user_id = uuid.uuid4()
        matter_id = uuid.uuid4()
        document = _document(
            owner_id=user_id,
            matter_id=matter_id,
            scope=DocumentScope.CONVERSATION,
        )

        service = _build_service()
        service.matters.get = AsyncMock(
            return_value=_matter(owner_id=user_id)
        )
        service.repo.get_by_id = AsyncMock(return_value=document)

        with pytest.raises(HTTPException) as exc_info:
            await service.delete_matter_document(
                user_id=user_id,
                matter_id=matter_id,
                document_id=document.id,
            )

        assert exc_info.value.status_code == 404
        service.vector_service.delete_document.assert_not_called()
        service.repo.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_foreign_matter(self):
        user_id = uuid.uuid4()
        matter_id = uuid.uuid4()
        other_user = uuid.uuid4()

        service = _build_service()
        service.matters.get = AsyncMock(
            return_value=_matter(owner_id=other_user)
        )

        with pytest.raises(HTTPException) as exc_info:
            await service.delete_matter_document(
                user_id=user_id,
                matter_id=matter_id,
                document_id=uuid.uuid4(),
            )

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_missing_storage_file_is_ignored(self):
        user_id = uuid.uuid4()
        matter_id = uuid.uuid4()
        document = _document(owner_id=user_id, matter_id=matter_id)
        document.storage_path = str(Path("/tmp/does-not-exist.txt"))

        service = _build_service()
        service.matters.get = AsyncMock(
            return_value=_matter(owner_id=user_id)
        )
        service.repo.get_by_id = AsyncMock(return_value=document)
        service.repo.delete = AsyncMock()

        await service.delete_matter_document(
            user_id=user_id,
            matter_id=matter_id,
            document_id=document.id,
        )

        service.repo.delete.assert_awaited_once_with(document)
