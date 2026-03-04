"""Add outcome columns to agent_runs

Revision ID: add_run_outcome_columns
Revises: add_sandbox_snapshots
Create Date: 2026-03-04 16:30:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect as sa_inspect

# revision identifiers, used by Alembic.
revision: str = "add_run_outcome_columns"
down_revision: Union[str, Sequence[str], None] = "add_sandbox_snapshots"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    cols = {c["name"] for c in inspector.get_columns(table_name)}
    return column_name in cols


def upgrade() -> None:
    if not _column_exists("agent_runs", "outcome_label"):
        op.add_column("agent_runs", sa.Column("outcome_label", sa.String(length=32), nullable=True))
    if not _column_exists("agent_runs", "outcome_reason_code"):
        op.add_column("agent_runs", sa.Column("outcome_reason_code", sa.String(length=128), nullable=True))
    if not _column_exists("agent_runs", "quality_score"):
        op.add_column("agent_runs", sa.Column("quality_score", sa.Float(), nullable=True))

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_agent_runs_outcome_label "
        "ON agent_runs (outcome_label)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_agent_runs_outcome_label")
    if _column_exists("agent_runs", "quality_score"):
        op.drop_column("agent_runs", "quality_score")
    if _column_exists("agent_runs", "outcome_reason_code"):
        op.drop_column("agent_runs", "outcome_reason_code")
    if _column_exists("agent_runs", "outcome_label"):
        op.drop_column("agent_runs", "outcome_label")

