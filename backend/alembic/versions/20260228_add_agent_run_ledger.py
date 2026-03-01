"""Add durable run ledger tables

Revision ID: add_agent_run_ledger
Revises: add_memories
Create Date: 2026-02-28 17:20:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect as sa_inspect

# revision identifiers, used by Alembic.
revision: str = "add_agent_run_ledger"
down_revision: Union[str, Sequence[str], None] = "add_memories"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa_inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if not _table_exists("agent_runs"):
        op.create_table(
            "agent_runs",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("task_id", sa.String(length=36), nullable=True),
            sa.Column("conversation_id", sa.String(length=36), nullable=True),
            sa.Column("user_id", sa.String(length=36), nullable=True),
            sa.Column("mode", sa.String(length=20), nullable=False),
            sa.Column("objective", sa.Text(), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("execution_mode", sa.String(length=20), nullable=False),
            sa.Column("run_labels", sa.JSON(), nullable=True),
            sa.Column("budget_json", sa.JSON(), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id"),
        )
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_runs_task_id ON agent_runs (task_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_runs_conversation_id ON agent_runs (conversation_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_runs_user_id ON agent_runs (user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_runs_user_created ON agent_runs (user_id, created_at)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_runs_status ON agent_runs (status)")

    if not _table_exists("agent_steps"):
        op.create_table(
            "agent_steps",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("run_id", sa.String(length=36), nullable=False),
            sa.Column("step_id", sa.String(length=64), nullable=False),
            sa.Column("step_type", sa.String(length=32), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("title", sa.String(length=255), nullable=True),
            sa.Column("details", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
            sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_steps_run_id ON agent_steps (run_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_steps_step_id ON agent_steps (step_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_steps_run_created ON agent_steps (run_id, created_at)")

    if not _table_exists("agent_events"):
        op.create_table(
            "agent_events",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("run_id", sa.String(length=36), nullable=False),
            sa.Column("event_type", sa.String(length=50), nullable=False),
            sa.Column("step_id", sa.String(length=64), nullable=True),
            sa.Column("dedup_key", sa.String(length=255), nullable=True),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("payload_hash", sa.String(length=64), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
            sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_events_run_id ON agent_events (run_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_events_event_type ON agent_events (event_type)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_events_step_id ON agent_events (step_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_events_run_created ON agent_events (run_id, created_at)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_agent_events_dedup_key ON agent_events (dedup_key)")

    if not _table_exists("tool_executions"):
        op.create_table(
            "tool_executions",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("run_id", sa.String(length=36), nullable=False),
            sa.Column("step_id", sa.String(length=64), nullable=True),
            sa.Column("tool_call_id", sa.String(length=128), nullable=False),
            sa.Column("tool_name", sa.String(length=100), nullable=False),
            sa.Column("tool_args", sa.JSON(), nullable=True),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("policy_decision", sa.String(length=32), nullable=True),
            sa.Column("policy_reason", sa.String(length=100), nullable=True),
            sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("result_summary", sa.Text(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
            sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
    op.execute("CREATE INDEX IF NOT EXISTS ix_tool_executions_run_id ON tool_executions (run_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_tool_executions_step_id ON tool_executions (step_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_tool_executions_tool_call_id ON tool_executions (tool_call_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_tool_executions_tool_name ON tool_executions (tool_name)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_tool_executions_run_created "
        "ON tool_executions (run_id, created_at)"
    )


def downgrade() -> None:
    op.drop_table("tool_executions")
    op.drop_table("agent_events")
    op.drop_table("agent_steps")
    op.drop_table("agent_runs")

