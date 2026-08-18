from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ActivityLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    user_id: UUID | None = None
    action: str
    entity_type: str | None = None
    entity_id: UUID | None = None
    summary: str
    metadata: dict | None = Field(default=None, validation_alias="metadata_")
    ip_address: str | None = None
    user_agent: str | None = None
    request_id: UUID | None = None
    created_at: datetime


class ActivityLogListResponse(BaseModel):
    items: list[ActivityLogResponse] = Field(default_factory=list)
    total: int
    limit: int
    offset: int


class RequestLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    request_id: UUID
    user_id: UUID | None = None
    method: str
    path: str
    query_params: dict | None = None
    path_params: dict | None = None
    request_headers: dict | None = None
    request_body: dict | list | str | None = None
    response_status: int
    response_body: dict | list | str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    device_type: str | None = None
    device_os: str | None = None
    device_browser: str | None = None
    duration_ms: int | None = None
    created_at: datetime


class RequestLogListResponse(BaseModel):
    items: list[RequestLogResponse] = Field(default_factory=list)
    total: int
    limit: int
    offset: int


class ErrorLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    request_id: UUID | None = None
    user_id: UUID | None = None
    error_type: str
    error_message: str
    stack_trace: str | None = None
    method: str | None = None
    path: str | None = None
    query_params: dict | None = None
    request_body: dict | list | str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    status_code: int | None = None
    created_at: datetime


class ErrorLogListResponse(BaseModel):
    items: list[ErrorLogResponse] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
