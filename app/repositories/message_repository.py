from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message import Message


class MessageRepository:

    def __init__(
        self,
        db: AsyncSession,
    ):
        self.db = db

    async def create(
        self,
        message: Message,
    ) -> Message:

        self.db.add(message)

        await self.db.flush()

        await self.db.refresh(message)

        return message

    async def list_by_conversation(
        self,
        conversation_id: UUID,
    ) -> list[Message]:

        result = await self.db.execute(
            select(Message)
            .where(
                Message.conversation_id == conversation_id
            )
            .order_by(
                Message.created_at.asc()
            )
        )

        return list(result.scalars().all())

    async def get_last_message(
        self,
        conversation_id: UUID,
    ) -> Message | None:

        result = await self.db.execute(
            select(Message)
            .where(
                Message.conversation_id == conversation_id
            )
            .order_by(
                Message.created_at.desc()
            )
            .limit(1)
        )

        return result.scalar_one_or_none()
