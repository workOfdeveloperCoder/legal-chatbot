from __future__ import annotations

from datetime import datetime
from uuid import UUID

from app.models.activity_log import ActivityAction, ActivityLog
from app.observability.context import get_request_context
from app.observability.writer import run_in_log_session
from app.repositories.activity_log_repository import ActivityLogRepository


class ActivityLogService:
    async def record(
        self,
        *,
        action: ActivityAction | str,
        summary: str,
        user_id: UUID | None = None,
        entity_type: str | None = None,
        entity_id: UUID | None = None,
        metadata: dict | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        request_id: UUID | None = None,
    ) -> None:
        context = get_request_context()
        action_value = action.value if isinstance(action, ActivityAction) else action

        entry = ActivityLog(
            user_id=user_id or context.user_id,
            action=action_value,
            entity_type=entity_type,
            entity_id=entity_id,
            summary=summary,
            metadata_=metadata,
            ip_address=ip_address or context.ip_address,
            user_agent=user_agent or context.user_agent,
            request_id=request_id or context.request_id,
        )

        async def _persist(session) -> None:
            await ActivityLogRepository(session).create(entry)

        await run_in_log_session(_persist)

    async def list_for_user(
        self,
        *,
        user_id: UUID,
        action: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ActivityLog], int]:
        async def _fetch(session):
            repo = ActivityLogRepository(session)
            items = await repo.list(
                user_id=user_id,
                action=action,
                since=since,
                until=until,
                limit=limit,
                offset=offset,
            )
            total = await repo.count(
                user_id=user_id,
                action=action,
                since=since,
                until=until,
            )
            return items, total

        result = await run_in_log_session(_fetch)
        return result or ([], 0)

    async def get_for_user(
        self,
        *,
        user_id: UUID,
        activity_id: UUID,
    ) -> ActivityLog | None:
        async def _fetch(session):
            entry = await ActivityLogRepository(session).get(activity_id)

            if entry is None or entry.user_id != user_id:
                return None

            return entry

        return await run_in_log_session(_fetch)
