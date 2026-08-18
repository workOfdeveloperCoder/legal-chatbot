from __future__ import annotations

from datetime import datetime
from uuid import UUID

from app.models.request_log import RequestLog
from app.observability.writer import run_in_log_session
from app.repositories.request_log_repository import RequestLogRepository


class RequestLogService:
    async def record(self, entry: RequestLog) -> None:
        async def _persist(session) -> None:
            await RequestLogRepository(session).create(entry)

        await run_in_log_session(_persist)

    async def list(
        self,
        *,
        user_id: UUID | None = None,
        status_code: int | None = None,
        path_prefix: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[RequestLog], int]:
        async def _fetch(session):
            repo = RequestLogRepository(session)
            items = await repo.list(
                user_id=user_id,
                status_code=status_code,
                path_prefix=path_prefix,
                since=since,
                until=until,
                limit=limit,
                offset=offset,
            )
            total = await repo.count(
                user_id=user_id,
                status_code=status_code,
                path_prefix=path_prefix,
                since=since,
                until=until,
            )
            return items, total

        result = await run_in_log_session(_fetch)
        return result or ([], 0)

    async def get(self, request_log_id: UUID) -> RequestLog | None:
        async def _fetch(session):
            return await RequestLogRepository(session).get(request_log_id)

        return await run_in_log_session(_fetch)
