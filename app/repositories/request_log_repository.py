from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import Select, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.request_log import RequestLog


class RequestLogRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, entry: RequestLog) -> RequestLog:
        self.db.add(entry)
        await self.db.flush()
        await self.db.refresh(entry)
        return entry

    def _base_query(
        self,
        *,
        user_id: UUID | None = None,
        status_code: int | None = None,
        path_prefix: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> Select[tuple[RequestLog]]:
        query = select(RequestLog)

        if user_id is not None:
            query = query.where(RequestLog.user_id == user_id)

        if status_code is not None:
            query = query.where(RequestLog.response_status == status_code)

        if path_prefix is not None:
            query = query.where(RequestLog.path.like(f"{path_prefix}%"))

        if since is not None:
            query = query.where(RequestLog.created_at >= since)

        if until is not None:
            query = query.where(RequestLog.created_at <= until)

        return query.order_by(desc(RequestLog.created_at))

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
    ) -> list[RequestLog]:
        query = self._base_query(
            user_id=user_id,
            status_code=status_code,
            path_prefix=path_prefix,
            since=since,
            until=until,
        ).limit(limit).offset(offset)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def count(
        self,
        *,
        user_id: UUID | None = None,
        status_code: int | None = None,
        path_prefix: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> int:
        from sqlalchemy import func

        query = select(func.count()).select_from(RequestLog)

        if user_id is not None:
            query = query.where(RequestLog.user_id == user_id)

        if status_code is not None:
            query = query.where(RequestLog.response_status == status_code)

        if path_prefix is not None:
            query = query.where(RequestLog.path.like(f"{path_prefix}%"))

        if since is not None:
            query = query.where(RequestLog.created_at >= since)

        if until is not None:
            query = query.where(RequestLog.created_at <= until)

        result = await self.db.execute(query)
        return int(result.scalar_one())

    async def get(self, request_log_id: UUID) -> RequestLog | None:
        result = await self.db.execute(
            select(RequestLog).where(RequestLog.id == request_log_id)
        )
        return result.scalar_one_or_none()

    async def get_by_request_id(
        self,
        request_id: UUID,
    ) -> RequestLog | None:
        result = await self.db.execute(
            select(RequestLog).where(RequestLog.request_id == request_id)
        )
        return result.scalar_one_or_none()
