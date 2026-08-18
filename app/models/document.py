from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import (
    String,
    DateTime,
    ForeignKey,
    Text,
    Boolean,
    Enum as SQLEnum,
    BigInteger,
    Index,
    func,
)

from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


class DocumentScope(str, Enum):
    """
    Controls document visibility.
    """

    MATTER = "matter"

    CONVERSATION = "conversation"



class Document(Base):

    __tablename__ = "documents"


    __table_args__ = (
        Index(
            "idx_documents_user_scope",
            "owner_id",
            "scope",
        ),

        Index(
            "idx_documents_matter",
            "owner_id",
            "matter_id",
        ),

        Index(
            "idx_documents_conversation",
            "owner_id",
            "conversation_id",
        ),
    )


    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )


    # ==================================================
    # SECURITY
    # ==================================================

    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )


    # ==================================================
    # DOCUMENT VISIBILITY
    #
    # MATTER:
    #   Available to all conversations of matter
    #
    # CONVERSATION:
    #   Only available to one conversation
    # ==================================================

    scope: Mapped[DocumentScope] = mapped_column(
        SQLEnum(DocumentScope),
        nullable=False,
        default=DocumentScope.MATTER,
        index=True,
    )


    matter_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "matters.id",
            ondelete="CASCADE",
        ),
        nullable=True,
    )


    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "conversations.id",
            ondelete="CASCADE",
        ),
        nullable=True,
    )


    # ==================================================
    # FILE DATA
    # ==================================================

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )


    mime_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )


    file_size: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )


    storage_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )


    # ==================================================
    # EXTRACTION
    # ==================================================

    extracted_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )


    processed: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )


    # ==================================================
    # QDRANT VECTOR STATUS
    # ==================================================

    vectorized: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )


    qdrant_collection: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )


    qdrant_point_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )


    # ==================================================
    # TIMESTAMPS
    # ==================================================

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


    # ==================================================
    # RELATIONSHIPS
    # ==================================================

    owner = relationship(
        "User",
        back_populates="documents",
    )


    matter = relationship(
        "Matter",
        back_populates="documents",
    )


    conversation = relationship(
        "Conversation",
        back_populates="documents",
    )