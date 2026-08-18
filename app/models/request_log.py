from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, UUIDMixin


class RequestLog(UUIDMixin, Base):
    __tablename__ = "request_logs"

    __table_args__ = (
        Index("idx_request_logs_created", "created_at"),
        Index("idx_request_logs_user_created", "user_id", "created_at"),
        Index("idx_request_logs_status_created", "response_status", "created_at"),
    )

    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        unique=True,
        nullable=False,
        index=True,
    )

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    method: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
    )

    path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    query_params: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    path_params: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    request_headers: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    request_body: Mapped[dict | list | str | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    response_status: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    response_body: Mapped[dict | list | str | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    ip_address: Mapped[str | None] = mapped_column(
        String(45),
        nullable=True,
    )

    user_agent: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    device_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    device_os: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    device_browser: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
