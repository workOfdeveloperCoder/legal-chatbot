from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import Select, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.error_log import ErrorLog


class ErrorLogRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, entry: ErrorLog) -> ErrorLog:
        self.db.add(entry)
        await self.db.flush()
        await self.db.refresh(entry)
        return entry

    def _base_query(
        self,
        *,
        user_id: UUID | None = None,
        error_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> Select[tuple[ErrorLog]]:
        query = select(ErrorLog)

        if user_id is not None:
            query = query.where(ErrorLog.user_id == user_id)

        if error_type is not None:
            query = query.where(ErrorLog.error_type == error_type)

        if since is not None:
            query = query.where(ErrorLog.created_at >= since)

        if until is not None:
            query = query.where(ErrorLog.created_at <= until)

        return query.order_by(desc(ErrorLog.created_at))

    async def list(
        self,
        *,
        user_id: UUID | None = None,
        error_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ErrorLog]:
        query = self._base_query(
            user_id=user_id,
            error_type=error_type,
            since=since,
            until=until,
        ).limit(limit).offset(offset)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def count(
        self,
        *,
        user_id: UUID | None = None,
        error_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> int:
        from sqlalchemy import func

        query = select(func.count()).select_from(ErrorLog)

        if user_id is not None:
            query = query.where(ErrorLog.user_id == user_id)

        if error_type is not None:
            query = query.where(ErrorLog.error_type == error_type)

        if since is not None:
            query = query.where(ErrorLog.created_at >= since)

        if until is not None:
            query = query.where(ErrorLog.created_at <= until)

        result = await self.db.execute(query)
        return int(result.scalar_one())

    async def get(self, error_log_id: UUID) -> ErrorLog | None:
        result = await self.db.execute(
            select(ErrorLog).where(ErrorLog.id == error_log_id)
        )
        return result.scalar_one_or_none()
