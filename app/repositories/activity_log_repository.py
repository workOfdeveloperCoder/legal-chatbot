from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import Select, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.activity_log import ActivityLog


class ActivityLogRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(self, entry: ActivityLog) -> ActivityLog:
        self.db.add(entry)
        await self.db.flush()
        await self.db.refresh(entry)
        return entry

    def _base_query(
        self,
        *,
        user_id: UUID | None = None,
        action: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> Select[tuple[ActivityLog]]:
        query = select(ActivityLog)

        if user_id is not None:
            query = query.where(ActivityLog.user_id == user_id)

        if action is not None:
            query = query.where(ActivityLog.action == action)

        if since is not None:
            query = query.where(ActivityLog.created_at >= since)

        if until is not None:
            query = query.where(ActivityLog.created_at <= until)

        return query.order_by(desc(ActivityLog.created_at))

    async def list(
        self,
        *,
        user_id: UUID | None = None,
        action: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ActivityLog]:
        query = self._base_query(
            user_id=user_id,
            action=action,
            since=since,
            until=until,
        ).limit(limit).offset(offset)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def count(
        self,
        *,
        user_id: UUID | None = None,
        action: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> int:
        from sqlalchemy import func

        query = select(func.count()).select_from(ActivityLog)

        if user_id is not None:
            query = query.where(ActivityLog.user_id == user_id)

        if action is not None:
            query = query.where(ActivityLog.action == action)

        if since is not None:
            query = query.where(ActivityLog.created_at >= since)

        if until is not None:
            query = query.where(ActivityLog.created_at <= until)

        result = await self.db.execute(query)
        return int(result.scalar_one())

    async def get(self, activity_id: UUID) -> ActivityLog | None:
        result = await self.db.execute(
            select(ActivityLog).where(ActivityLog.id == activity_id)
        )
        return result.scalar_one_or_none()
