from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies.auth import get_current_active_user
from app.api.dependencies.services import (
    get_conversation_service,
    get_document_service,
    get_matter_service,
)
from app.api.v1.conversations import _to_list_item
from app.models.user import User
from app.schemas.conversation import ConversationListResponse
from app.schemas.document import MatterDocumentDetail, MatterDocumentListResponse
from app.schemas.matter import (
    CreateMatterRequest,
    MatterListResponse,
    MatterResponse,
    UpdateMatterRequest,
)
from app.services.conversation_service import ConversationService
from app.services.document_service import DocumentService
from app.services.matter_service import MatterService

router = APIRouter(
    prefix="/matters",
    tags=["Matters"],
)


@router.post(
    "",
    response_model=MatterResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_matter(
    payload: CreateMatterRequest,
    current_user: User = Depends(get_current_active_user),
    service: MatterService = Depends(get_matter_service),
):
    matter = await service.create(
        current_user=current_user,
        payload=payload,
    )

    return MatterResponse.model_validate(matter)


@router.get(
    "",
    response_model=MatterListResponse,
)
async def list_matters(
    current_user: User = Depends(get_current_active_user),
    service: MatterService = Depends(get_matter_service),
):
    matters = await service.list(current_user)

    return MatterListResponse(
        items=[
            MatterResponse.model_validate(m)
            for m in matters
        ]
    )


@router.get(
    "/{matter_id}/conversations",
    response_model=ConversationListResponse,
)
async def list_matter_conversations(
    matter_id: UUID,
    current_user: User = Depends(get_current_active_user),
    matter_service: MatterService = Depends(get_matter_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
):
    await matter_service.get(
        current_user=current_user,
        matter_id=matter_id,
    )

    conversations = await conversation_service.list_by_matter(
        user=current_user,
        matter_id=matter_id,
    )

    items = []
    for conversation in conversations:
        last = await conversation_service.messages.get_last_message(
            conversation.id,
        )
        items.append(
            _to_list_item(
                conversation,
                last_message=last.content if last else None,
            )
        )

    return ConversationListResponse(items=items)


@router.get(
    "/{matter_id}/documents",
    response_model=MatterDocumentListResponse,
)
async def list_matter_documents(
    matter_id: UUID,
    current_user: User = Depends(get_current_active_user),
    document_service: DocumentService = Depends(get_document_service),
):
    """List matter-scoped documents (summary only; no full text)."""
    items = await document_service.list_matter_documents(
        user_id=current_user.id,
        matter_id=matter_id,
    )

    return MatterDocumentListResponse(
        matter_id=matter_id,
        items=items,
        total=len(items),
    )


@router.get(
    "/{matter_id}/document/{document_id}",
    response_model=MatterDocumentDetail,
)
async def get_matter_document(
    matter_id: UUID,
    document_id: UUID,
    current_user: User = Depends(get_current_active_user),
    document_service: DocumentService = Depends(get_document_service),
):
    """Get one matter document with full merged text content."""
    return await document_service.get_matter_document(
        user_id=current_user.id,
        matter_id=matter_id,
        document_id=document_id,
    )


@router.delete(
    "/{matter_id}/document/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_matter_document(
    matter_id: UUID,
    document_id: UUID,
    current_user: User = Depends(get_current_active_user),
    document_service: DocumentService = Depends(get_document_service),
):
    """Delete a matter-scoped document from PostgreSQL and Qdrant (if present)."""
    await document_service.delete_matter_document(
        user_id=current_user.id,
        matter_id=matter_id,
        document_id=document_id,
    )


@router.get(
    "/{matter_id}",
    response_model=MatterResponse,
)
async def get_matter(
    matter_id: UUID,
    current_user: User = Depends(get_current_active_user),
    service: MatterService = Depends(get_matter_service),
):
    matter = await service.get(
        current_user=current_user,
        matter_id=matter_id,
    )

    return MatterResponse.model_validate(matter)


@router.patch(
    "/{matter_id}",
    response_model=MatterResponse,
)
async def update_matter(
    matter_id: UUID,
    payload: UpdateMatterRequest,
    current_user: User = Depends(get_current_active_user),
    service: MatterService = Depends(get_matter_service),
):
    matter = await service.update(
        current_user=current_user,
        matter_id=matter_id,
        payload=payload,
    )

    return MatterResponse.model_validate(matter)


@router.delete(
    "/{matter_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_matter(
    matter_id: UUID,
    current_user: User = Depends(get_current_active_user),
    service: MatterService = Depends(get_matter_service),
):
    await service.delete(
        current_user=current_user,
        matter_id=matter_id,
    )

    return Response(status_code=status.HTTP_204_NO_CONTENT)