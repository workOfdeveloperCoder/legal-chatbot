from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies.auth import get_current_active_user, require_roles
from app.api.dependencies.services import (
    get_activity_log_service,
    get_error_log_service,
    get_request_log_service,
)
from app.models.user import User, UserRole
from app.schemas.logs import (
    ActivityLogListResponse,
    ActivityLogResponse,
    ErrorLogListResponse,
    ErrorLogResponse,
    RequestLogListResponse,
    RequestLogResponse,
)
from app.services.activity_log_service import ActivityLogService
from app.services.error_log_service import ErrorLogService
from app.services.request_log_service import RequestLogService

router = APIRouter(
    prefix="/logs",
    tags=["Logs"],
)


@router.get(
    "/activities",
    response_model=ActivityLogListResponse,
)
async def list_my_activities(
    action: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_active_user),
    service: ActivityLogService = Depends(get_activity_log_service),
):
    items, total = await service.list_for_user(
        user_id=current_user.id,
        action=action,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )

    return ActivityLogListResponse(
        items=[ActivityLogResponse.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/activities/{activity_id}",
    response_model=ActivityLogResponse,
)
async def get_my_activity(
    activity_id: UUID,
    current_user: User = Depends(get_current_active_user),
    service: ActivityLogService = Depends(get_activity_log_service),
):
    entry = await service.get_for_user(
        user_id=current_user.id,
        activity_id=activity_id,
    )

    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Activity log not found.",
        )

    return ActivityLogResponse.model_validate(entry)


@router.get(
    "/requests",
    response_model=RequestLogListResponse,
)
async def list_request_logs(
    user_id: UUID | None = None,
    status_code: int | None = None,
    path_prefix: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _: User = Depends(require_roles(UserRole.ADMIN)),
    service: RequestLogService = Depends(get_request_log_service),
):
    items, total = await service.list(
        user_id=user_id,
        status_code=status_code,
        path_prefix=path_prefix,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )

    return RequestLogListResponse(
        items=[RequestLogResponse.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/requests/{request_log_id}",
    response_model=RequestLogResponse,
)
async def get_request_log(
    request_log_id: UUID,
    _: User = Depends(require_roles(UserRole.ADMIN)),
    service: RequestLogService = Depends(get_request_log_service),
):
    entry = await service.get(request_log_id)

    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Request log not found.",
        )

    return RequestLogResponse.model_validate(entry)


@router.get(
    "/errors",
    response_model=ErrorLogListResponse,
)
async def list_error_logs(
    user_id: UUID | None = None,
    error_type: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _: User = Depends(require_roles(UserRole.ADMIN)),
    service: ErrorLogService = Depends(get_error_log_service),
):
    items, total = await service.list(
        user_id=user_id,
        error_type=error_type,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )

    return ErrorLogListResponse(
        items=[ErrorLogResponse.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/errors/{error_log_id}",
    response_model=ErrorLogResponse,
)
async def get_error_log(
    error_log_id: UUID,
    _: User = Depends(require_roles(UserRole.ADMIN)),
    service: ErrorLogService = Depends(get_error_log_service),
):
    entry = await service.get(error_log_id)

    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Error log not found.",
        )

    return ErrorLogResponse.model_validate(entry)
