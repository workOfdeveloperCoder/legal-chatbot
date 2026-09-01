from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, status

from app.api.dependencies.auth import get_current_active_user
from app.api.dependencies.services import get_conversation_service
from app.models.user import User
from app.schemas.conversation import (
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
    CreateConversationRequest,
    UpdateConversationRequest,
)
from app.schemas.message import MessageResponse
from app.services.conversation_service import ConversationService

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)


def _to_list_item(
    conversation,
    *,
    last_message: str | None = None,
) -> ConversationResponse:
    matter_title = None
    if conversation.matter is not None:
        matter_title = conversation.matter.title

    return ConversationResponse(
        id=conversation.id,
        matter_id=conversation.matter_id,
        title=conversation.title,
        is_pinned=conversation.is_pinned,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        last_message_at=conversation.last_message_at,
        last_message=last_message,
        matter_title=matter_title,
    )


@router.get(
    "",
    response_model=ConversationListResponse,
)
async def list_conversations(
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service),
):
    conversations = await service.list_by_user(user=current_user)

    items: list[ConversationResponse] = []
    for conversation in conversations:
        last = await service.messages.get_last_message(conversation.id)
        items.append(
            _to_list_item(
                conversation,
                last_message=last.content if last else None,
            )
        )

    return ConversationListResponse(items=items)


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    payload: CreateConversationRequest,
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service),
):
    conversation = await service.create(
        user=current_user,
        title=payload.title,
        matter_id=payload.matter_id,
    )
    return _to_list_item(conversation)


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
)
async def get_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service),
):
    conversation = await service.get_for_user(
        user=current_user,
        conversation_id=conversation_id,
    )

    await service.record_view(
        user=current_user,
        conversation=conversation,
    )

    messages = await service.get_history(
        user=current_user,
        conversation_id=conversation_id,
    )

    matter_title = None
    if conversation.matter is not None:
        matter_title = conversation.matter.title

    return ConversationDetailResponse(
        id=conversation.id,
        matter_id=conversation.matter_id,
        title=conversation.title,
        is_pinned=conversation.is_pinned,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        last_message_at=conversation.last_message_at,
        matter_title=matter_title,
        messages=[
            MessageResponse.model_validate(message)
            for message in messages
        ],
    )


@router.patch(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
async def update_conversation(
    conversation_id: UUID,
    payload: UpdateConversationRequest,
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service),
):
    conversation = await service.rename(
        user=current_user,
        conversation_id=conversation_id,
        title=payload.title,
    )
    if payload.is_pinned is not None:
        conversation.is_pinned = payload.is_pinned
        await service.db.commit()
        await service.db.refresh(conversation)

    last = await service.messages.get_last_message(conversation.id)
    return _to_list_item(
        conversation,
        last_message=last.content if last else None,
    )


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_active_user),
    service: ConversationService = Depends(get_conversation_service),
):
    await service.delete(
        user=current_user,
        conversation_id=conversation_id,
    )
