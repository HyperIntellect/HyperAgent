"""Add sandbox_snapshots table

Revision ID: add_sandbox_snapshots
Revises: add_agent_run_ledger
Create Date: 2026-03-01 10:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect as sa_inspect

# revision identifiers, used by Alembic.
revision: str = "add_sandbox_snapshots"
down_revision: Union[str, Sequence[str], None] = "add_agent_run_ledger"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    """Check if a table already exists (handles dev auto-create)."""
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    """Upgrade schema."""
    if not _table_exists("sandbox_snapshots"):
        op.create_table(
            "sandbox_snapshots",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("user_id", sa.String(length=255), nullable=False),
            sa.Column("task_id", sa.String(length=36), nullable=False),
            sa.Column("sandbox_type", sa.String(length=20), nullable=False),
            sa.Column("storage_key", sa.String(length=500), nullable=False),
            sa.Column("paths_included", sa.JSON(), nullable=False),
            sa.Column("size_bytes", sa.Integer(), nullable=False, server_default=sa.text("0")),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("storage_key"),
        )

    # Indexes
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_sandbox_snapshots_user_id "
        "ON sandbox_snapshots (user_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_sandbox_snapshots_task_id "
        "ON sandbox_snapshots (task_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_sandbox_snapshots_user_task_type "
        "ON sandbox_snapshots (user_id, task_id, sandbox_type)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_sandbox_snapshots_user_task_type", table_name="sandbox_snapshots")
    op.drop_index("ix_sandbox_snapshots_task_id", table_name="sandbox_snapshots")
    op.drop_index("ix_sandbox_snapshots_user_id", table_name="sandbox_snapshots")
    op.drop_table("sandbox_snapshots")
