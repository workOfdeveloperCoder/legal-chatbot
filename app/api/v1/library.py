from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies.auth import get_current_active_user
from app.models.user import User
from app.schemas.library import LibraryDocumentDetail
from app.services.library_document_service import LibraryDocumentService

router = APIRouter(
    prefix="/library",
    tags=["Library"],
)


@router.get(
    "/documents/{document_id}",
    response_model=LibraryDocumentDetail,
)
async def get_library_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Full legal-corpus document text rebuilt from Qdrant chunks.

    ``document_id`` is the corpus hash (not a Postgres UUID), e.g. from
    chat Resources cards.
    """
    del current_user  # auth required; corpus is shared read-only
    service = LibraryDocumentService()
    return await service.get_document(document_id=document_id)
