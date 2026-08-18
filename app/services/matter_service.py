from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.matter import Matter
from app.models.user import User
from app.repositories.matter_repository import MatterRepository
from app.schemas.matter import (
    CreateMatterRequest,
    UpdateMatterRequest,
)
from app.models.activity_log import ActivityAction
from app.services.activity_log_service import ActivityLogService


class MatterService:
    def __init__(
        self,
        db: AsyncSession,
        repository: MatterRepository,
        activity_log_service: ActivityLogService,
    ):
        self.db = db
        self.repository = repository
        self.activity_logs = activity_log_service

    async def create(
        self,
        current_user: User,
        payload: CreateMatterRequest,
    ) -> Matter:

        exists = await self.repository.exists_by_title(
            current_user.id,
            payload.title.strip(),
        )

        if exists:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A matter with this title already exists.",
            )

        # Session already has an open transaction (auth + exists check).
        # Do not call db.begin() again — commit the existing transaction.
        matter = Matter(
            owner_id=current_user.id,
            title=payload.title.strip(),
            description=payload.description,
        )

        matter = await self.repository.create(matter)
        await self.db.commit()
        await self.db.refresh(matter)

        await self.activity_logs.record(
            action=ActivityAction.MATTER_CREATED,
            summary=f'Created matter "{matter.title}"',
            user_id=current_user.id,
            entity_type="matter",
            entity_id=matter.id,
            metadata={"title": matter.title},
        )

        return matter

    async def get(
        self,
        current_user: User,
        matter_id: UUID,
    ) -> Matter:

        matter = await self.repository.get(matter_id)

        if matter is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Matter not found.",
            )

        if matter.owner_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this matter.",
            )

        return matter

    async def list(
        self,
        current_user: User,
    ) -> list[Matter]:

        return await self.repository.list_by_user(
            current_user.id,
        )

    async def update(
        self,
        current_user: User,
        matter_id: UUID,
        payload: UpdateMatterRequest,
    ) -> Matter:

        matter = await self.get(
            current_user,
            matter_id,
        )

        if payload.title is not None:
            matter.title = payload.title.strip()

        if payload.description is not None:
            matter.description = payload.description

        await self.repository.update(matter)
        await self.db.commit()
        await self.db.refresh(matter)

        await self.activity_logs.record(
            action=ActivityAction.MATTER_UPDATED,
            summary=f'Updated matter "{matter.title}"',
            user_id=current_user.id,
            entity_type="matter",
            entity_id=matter.id,
            metadata={"title": matter.title},
        )

        return matter

    async def delete(
        self,
        current_user: User,
        matter_id: UUID,
    ) -> None:

        matter = await self.get(
            current_user,
            matter_id,
        )

        title = matter.title

        await self.repository.delete(matter)
        await self.db.commit()

        await self.activity_logs.record(
            action=ActivityAction.MATTER_DELETED,
            summary=f'Deleted matter "{title}"',
            user_id=current_user.id,
            entity_type="matter",
            entity_id=matter_id,
            metadata={"title": title},
        )
