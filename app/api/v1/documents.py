from uuid import UUID

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
)

from app.api.dependencies.auth import get_current_active_user
from app.api.dependencies.services import get_document_service
from app.models.user import User
from app.services.document_service import DocumentService
from app.schemas.document import DocumentResponse, MatterDocumentDetail


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post(
    "/conversations/{conversation_id}/upload",
    response_model=DocumentResponse,
)
async def upload_conversation_document(
    conversation_id: UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    service: DocumentService = Depends(get_document_service),
):
    """Upload a document scoped to one conversation."""
    return await service.upload(
        user_id=current_user.id,
        conversation_id=conversation_id,
        file=file,
    )


@router.get(
    "/conversations/{conversation_id}/document/{document_id}",
    response_model=MatterDocumentDetail,
)
async def get_conversation_document(
    conversation_id: UUID,
    document_id: UUID,
    current_user: User = Depends(get_current_active_user),
    service: DocumentService = Depends(get_document_service),
):
    """Get one conversation-scoped document with full merged text content."""
    return await service.get_conversation_document(
        user_id=current_user.id,
        conversation_id=conversation_id,
        document_id=document_id,
    )


@router.post(
    "/matters/{matter_id}/upload",
    response_model=DocumentResponse,
)
async def upload_matter_document(
    matter_id: UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    service: DocumentService = Depends(get_document_service),
):
    """Upload a document shared across a matter."""
    return await service.upload(
        user_id=current_user.id,
        matter_id=matter_id,
        file=file,
    )


@router.post(
    "/{matter_id}/upload",
    response_model=DocumentResponse,
    deprecated=True,
)
async def upload_document_legacy(
    matter_id: UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    service: DocumentService = Depends(get_document_service),
):
    """Legacy matter upload path. Prefer /documents/matters/{matter_id}/upload."""
    return await service.upload(
        user_id=current_user.id,
        matter_id=matter_id,
        file=file,
    )
