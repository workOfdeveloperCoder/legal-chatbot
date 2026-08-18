from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import AsyncSessionLocal

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def run_in_log_session(
    operation: Callable[[AsyncSession], Awaitable[T]],
) -> T | None:
    """
    Persist log records in an isolated DB session so logging never
    interferes with the request transaction.
    """

    async with AsyncSessionLocal() as session:
        try:
            result = await operation(session)
            await session.commit()
            return result
        except Exception:
            await session.rollback()
            logger.exception("Failed to persist observability record")
            return None
