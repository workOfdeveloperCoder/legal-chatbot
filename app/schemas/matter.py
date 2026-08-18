from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.matter import MatterStatus


class CreateMatterRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str | None = None
    color: str = "#2563EB"
    icon: str = "briefcase"


class UpdateMatterRequest(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    description: str | None = None
    color: str | None = None
    icon: str | None = None
    is_pinned: bool | None = None
    status: MatterStatus | None = None


class MatterResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    description: str | None

    color: str
    icon: str

    status: MatterStatus
    is_pinned: bool

    created_at: datetime
    updated_at: datetime
    last_opened_at: datetime | None


class MatterListResponse(BaseModel):
    items: list[MatterResponse]