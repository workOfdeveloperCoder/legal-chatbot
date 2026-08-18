from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
from sqlalchemy import update

from app.models.refresh_token import RefreshToken


class RefreshTokenRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        token: RefreshToken,
    ) -> RefreshToken:
        self.db.add(token)
        await self.db.flush()
        await self.db.refresh(token)
        return token

    async def update(
        self,
        token: RefreshToken,
    ) -> RefreshToken:
        await self.db.flush()
        await self.db.refresh(token)
        return token

    async def get_by_hash(
        self,
        token_hash: str,
    ) -> RefreshToken | None:
        result = await self.db.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == token_hash
            )
        )

        return result.scalar_one_or_none()

    async def get_active_by_hash(
        self,
        token_hash: str,
    ) -> RefreshToken | None:
        result = await self.db.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == token_hash,
                RefreshToken.revoked_at.is_(None),
            )
        )

        return result.scalar_one_or_none()

    async def delete(
        self,
        token: RefreshToken,
    ) -> None:
        await self.db.delete(token)

    async def revoke(
        self,
        token: RefreshToken,
    ) -> None:
        token.revoked_at = datetime.now(timezone.utc)
        await self.db.flush()

    async def revoke_all_for_user(
        self,
        user_id,
    ) -> None:
        await self.db.execute(
            update(RefreshToken)
            .where(
                RefreshToken.user_id == user_id,
                RefreshToken.revoked_at.is_(None),
            )
            .values(
                revoked_at=datetime.now(timezone.utc)
            )
        )