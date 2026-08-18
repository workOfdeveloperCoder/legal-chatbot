from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.matter import Matter


class MatterRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        matter: Matter,
    ) -> Matter:
        self.db.add(matter)
        await self.db.flush()
        await self.db.refresh(matter)
        return matter

    async def get(
        self,
        matter_id: UUID,
    ) -> Matter | None:
        result = await self.db.execute(
            select(Matter).where(
                Matter.id == matter_id
            )
        )

        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: UUID,
    ) -> list[Matter]:
        result = await self.db.execute(
            select(Matter)
            .where(Matter.owner_id == user_id)
            .order_by(Matter.updated_at.desc())
        )

        return list(result.scalars().all())

    async def update(
        self,
        matter: Matter,
    ) -> Matter:

        await self.db.flush()
        await self.db.refresh(matter)

        return matter

    async def delete(
        self,
        matter: Matter,
    ):
        await self.db.delete(matter)

    async def exists_by_title(
        self,
        owner_id: UUID,
        title: str,
    ) -> bool:
        result = await self.db.execute(
            select(Matter).where(
                Matter.owner_id == owner_id,
                Matter.title == title,
            )
        )

        return result.scalar_one_or_none() is not None