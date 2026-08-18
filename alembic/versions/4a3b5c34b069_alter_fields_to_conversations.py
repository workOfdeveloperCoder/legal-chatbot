"""alter conversations

Revision ID: 32f1d59ae725
Revises: 22f1d59ae725
Create Date: 2026-07-24

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers
revision = "32f1d59ae725"
down_revision = "22f1d59ae725"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --------------------------------------------------
    # Add conversation owner
    # --------------------------------------------------

    op.add_column(
        "conversations",
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )

    op.create_index(
        "ix_conversations_user_id",
        "conversations",
        ["user_id"],
    )

    op.create_foreign_key(
        "fk_conversations_user_id",
        "conversations",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # --------------------------------------------------
    # Make matter optional
    # --------------------------------------------------

    op.drop_constraint(
        "conversations_matter_id_fkey",
        "conversations",
        type_="foreignkey",
    )

    op.alter_column(
        "conversations",
        "matter_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=True,
    )

    op.create_foreign_key(
        "fk_conversations_matter_id",
        "conversations",
        "matters",
        ["matter_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    # --------------------------------------------------
    # Restore matter constraint
    # --------------------------------------------------

    op.drop_constraint(
        "fk_conversations_matter_id",
        "conversations",
        type_="foreignkey",
    )

    op.alter_column(
        "conversations",
        "matter_id",
        existing_type=postgresql.UUID(as_uuid=True),
        nullable=False,
    )

    op.create_foreign_key(
        "conversations_matter_id_fkey",
        "conversations",
        "matters",
        ["matter_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # --------------------------------------------------
    # Remove user_id
    # --------------------------------------------------

    op.drop_constraint(
        "fk_conversations_user_id",
        "conversations",
        type_="foreignkey",
    )

    op.drop_index(
        "ix_conversations_user_id",
        table_name="conversations",
    )

    op.drop_column(
        "conversations",
        "user_id",
    )