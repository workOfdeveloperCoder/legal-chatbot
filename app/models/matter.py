from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, DateTime, Enum as SQLEnum, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class MatterStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class Matter(Base):
    __tablename__ = "matters"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    color: Mapped[str] = mapped_column(
        String(20),
        default="#2563EB",
        nullable=False,
    )

    icon: Mapped[str] = mapped_column(
        String(50),
        default="briefcase",
        nullable=False,
    )

    status: Mapped[MatterStatus] = mapped_column(
        SQLEnum(MatterStatus),
        default=MatterStatus.ACTIVE,
        nullable=False,
    )

    is_pinned: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    last_opened_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    owner = relationship(
        "User",
        back_populates="matters",
        lazy="selectin",
    )

    conversations = relationship(
        "Conversation",
        back_populates="matter",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    
    documents = relationship(
        "Document",
        back_populates="matter",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
