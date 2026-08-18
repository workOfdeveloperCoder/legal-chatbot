"""create documents table

Revision ID: b7e2c91f4a10
Revises: af04c4c6a495
Create Date: 2026-08-11 19:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "b7e2c91f4a10"
down_revision: Union[str, Sequence[str], None] = "af04c4c6a495"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_type
                WHERE typname = 'documentscope'
            ) THEN
                CREATE TYPE documentscope AS ENUM ('MATTER', 'CONVERSATION');
            END IF;
        END
        $$;
        """
    )

    documentscope = postgresql.ENUM(
        "MATTER",
        "CONVERSATION",
        name="documentscope",
        create_type=False,
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("owner_id", sa.UUID(), nullable=False),
        sa.Column("scope", documentscope, nullable=False),
        sa.Column("matter_id", sa.UUID(), nullable=True),
        sa.Column("conversation_id", sa.UUID(), nullable=True),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("mime_type", sa.String(length=100), nullable=True),
        sa.Column("file_size", sa.BigInteger(), nullable=True),
        sa.Column("storage_path", sa.String(length=500), nullable=False),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column("processed", sa.Boolean(), nullable=False),
        sa.Column("vectorized", sa.Boolean(), nullable=False),
        sa.Column("qdrant_collection", sa.String(length=100), nullable=True),
        sa.Column("qdrant_point_id", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["matter_id"],
            ["matters.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        op.f("ix_documents_owner_id"),
        "documents",
        ["owner_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_documents_scope"),
        "documents",
        ["scope"],
        unique=False,
    )
    op.create_index(
        op.f("ix_documents_processed"),
        "documents",
        ["processed"],
        unique=False,
    )
    op.create_index(
        op.f("ix_documents_vectorized"),
        "documents",
        ["vectorized"],
        unique=False,
    )
    op.create_index(
        op.f("ix_documents_qdrant_point_id"),
        "documents",
        ["qdrant_point_id"],
        unique=False,
    )
    op.create_index(
        "idx_documents_user_scope",
        "documents",
        ["owner_id", "scope"],
        unique=False,
    )
    op.create_index(
        "idx_documents_matter",
        "documents",
        ["owner_id", "matter_id"],
        unique=False,
    )
    op.create_index(
        "idx_documents_conversation",
        "documents",
        ["owner_id", "conversation_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_documents_conversation", table_name="documents")
    op.drop_index("idx_documents_matter", table_name="documents")
    op.drop_index("idx_documents_user_scope", table_name="documents")
    op.drop_index(op.f("ix_documents_qdrant_point_id"), table_name="documents")
    op.drop_index(op.f("ix_documents_vectorized"), table_name="documents")
    op.drop_index(op.f("ix_documents_processed"), table_name="documents")
    op.drop_index(op.f("ix_documents_scope"), table_name="documents")
    op.drop_index(op.f("ix_documents_owner_id"), table_name="documents")
    op.drop_table("documents")
    op.execute("DROP TYPE IF EXISTS documentscope")
