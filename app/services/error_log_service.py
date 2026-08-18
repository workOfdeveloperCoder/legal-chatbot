from __future__ import annotations

import traceback
from datetime import datetime
from uuid import UUID

from fastapi import HTTPException, Request

from app.models.error_log import ErrorLog
from app.observability.context import get_request_context
from app.observability.sanitize import parse_json_body, truncate_payload
from app.observability.writer import run_in_log_session
from app.repositories.error_log_repository import ErrorLogRepository
from app.core.config import settings


class ErrorLogService:
    async def record_exception(
        self,
        request: Request,
        exc: Exception,
        *,
        status_code: int | None = None,
    ) -> None:
        if not self._should_log_exception(exc, status_code=status_code):
            return

        context = get_request_context()
        body_bytes = getattr(request.state, "body_bytes", b"")

        entry = ErrorLog(
            request_id=context.request_id,
            user_id=context.user_id,
            error_type=type(exc).__name__,
            error_message=str(exc) or type(exc).__name__,
            stack_trace=traceback.format_exc(),
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            request_body=truncate_payload(
                parse_json_body(body_bytes),
                max_bytes=settings.LOG_BODY_MAX_BYTES,
            ),
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            status_code=status_code,
        )

        async def _persist(session) -> None:
            await ErrorLogRepository(session).create(entry)

        await run_in_log_session(_persist)

    @staticmethod
    def _should_log_exception(
        exc: Exception,
        *,
        status_code: int | None,
    ) -> bool:
        if not isinstance(exc, HTTPException):
            return True

        if status_code is None:
            status_code = exc.status_code

        if status_code >= 500:
            return True

        if status_code in {401, 403, 409, 422}:
            return True

        return False

    async def list(
        self,
        *,
        user_id: UUID | None = None,
        error_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ErrorLog], int]:
        async def _fetch(session):
            repo = ErrorLogRepository(session)
            items = await repo.list(
                user_id=user_id,
                error_type=error_type,
                since=since,
                until=until,
                limit=limit,
                offset=offset,
            )
            total = await repo.count(
                user_id=user_id,
                error_type=error_type,
                since=since,
                until=until,
            )
            return items, total

        result = await run_in_log_session(_fetch)
        return result or ([], 0)

    async def get(self, error_log_id: UUID) -> ErrorLog | None:
        async def _fetch(session):
            return await ErrorLogRepository(session).get(error_log_id)

        return await run_in_log_session(_fetch)
