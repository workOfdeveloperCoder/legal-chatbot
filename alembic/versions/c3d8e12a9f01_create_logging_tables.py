"""create logging tables

Revision ID: c3d8e12a9f01
Revises: b7e2c91f4a10
Create Date: 2026-08-12 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "c3d8e12a9f01"
down_revision: Union[str, Sequence[str], None] = "b7e2c91f4a10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "activity_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("entity_type", sa.String(length=50), nullable=True),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_activity_logs_user_created", "activity_logs", ["user_id", "created_at"])
    op.create_index("idx_activity_logs_action_created", "activity_logs", ["action", "created_at"])
    op.create_index("idx_activity_logs_request_id", "activity_logs", ["request_id"])

    op.create_table(
        "request_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("method", sa.String(length=10), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("query_params", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("path_params", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("request_headers", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("request_body", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("response_status", sa.Integer(), nullable=False),
        sa.Column("response_body", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("device_type", sa.String(length=50), nullable=True),
        sa.Column("device_os", sa.String(length=100), nullable=True),
        sa.Column("device_browser", sa.String(length=100), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("request_id"),
    )
    op.create_index("idx_request_logs_created", "request_logs", ["created_at"])
    op.create_index("idx_request_logs_user_created", "request_logs", ["user_id", "created_at"])
    op.create_index(
        "idx_request_logs_status_created",
        "request_logs",
        ["response_status", "created_at"],
    )
    op.create_index(op.f("ix_request_logs_request_id"), "request_logs", ["request_id"], unique=True)
    op.create_index(op.f("ix_request_logs_user_id"), "request_logs", ["user_id"], unique=False)

    op.create_table(
        "error_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("error_type", sa.String(length=255), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=False),
        sa.Column("stack_trace", sa.Text(), nullable=True),
        sa.Column("method", sa.String(length=10), nullable=True),
        sa.Column("path", sa.Text(), nullable=True),
        sa.Column("query_params", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("request_body", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_error_logs_created", "error_logs", ["created_at"])
    op.create_index("idx_error_logs_user_created", "error_logs", ["user_id", "created_at"])
    op.create_index("idx_error_logs_type_created", "error_logs", ["error_type", "created_at"])
    op.create_index(op.f("ix_error_logs_request_id"), "error_logs", ["request_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_error_logs_request_id"), table_name="error_logs")
    op.drop_index("idx_error_logs_type_created", table_name="error_logs")
    op.drop_index("idx_error_logs_user_created", table_name="error_logs")
    op.drop_index("idx_error_logs_created", table_name="error_logs")
    op.drop_table("error_logs")

    op.drop_index(op.f("ix_request_logs_user_id"), table_name="request_logs")
    op.drop_index(op.f("ix_request_logs_request_id"), table_name="request_logs")
    op.drop_index("idx_request_logs_status_created", table_name="request_logs")
    op.drop_index("idx_request_logs_user_created", table_name="request_logs")
    op.drop_index("idx_request_logs_created", table_name="request_logs")
    op.drop_table("request_logs")

    op.drop_index("idx_activity_logs_request_id", table_name="activity_logs")
    op.drop_index("idx_activity_logs_action_created", table_name="activity_logs")
    op.drop_index("idx_activity_logs_user_created", table_name="activity_logs")
    op.drop_table("activity_logs")
