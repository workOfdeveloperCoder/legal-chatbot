from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.message import MessageResponse


class CreateConversationRequest(BaseModel):
    title: str = "New Conversation"
    matter_id: UUID | None = None


class UpdateConversationRequest(BaseModel):
    title: str
    is_pinned: bool | None = None


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    matter_id: UUID | None = None

    title: str

    is_pinned: bool

    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None = None

    # Optional list enrichment for sidebar / UI
    last_message: str | None = None
    matter_title: str | None = None


class ConversationListResponse(BaseModel):
    items: list[ConversationResponse] = Field(default_factory=list)


class ConversationDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    matter_id: UUID | None = None

    title: str

    is_pinned: bool

    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None = None

    matter_title: str | None = None

    messages: list[MessageResponse] = Field(default_factory=list)
