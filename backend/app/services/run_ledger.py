"""Durable run ledger service for replay, audit, and resumability."""

from __future__ import annotations

import functools
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.exc import ProgrammingError

from app.core.logging import get_logger
from app.db.base import get_db_session
from app.db.models import AgentEvent, AgentRun, AgentStep, ToolExecution

logger = get_logger(__name__)

_TABLE_MISSING_WARNED = False


def _safe(func):
    """Decorator that catches missing-table errors so the ledger never crashes the request."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        global _TABLE_MISSING_WARNED
        try:
            return await func(*args, **kwargs)
        except (ProgrammingError, Exception) as exc:
            if "UndefinedTableError" in str(type(exc).__name__) or "UndefinedTable" in str(exc):
                if not _TABLE_MISSING_WARNED:
                    logger.warning(
                        "run_ledger tables missing — run 'make migrate' to create them. "
                        "Ledger writes disabled until then."
                    )
                    _TABLE_MISSING_WARNED = True
                return None
            raise

    return wrapper


class RunLedgerService:
    @_safe
    async def create_run(
        self,
        *,
        run_id: str,
        user_id: str | None,
        mode: str,
        objective: str,
        task_id: str | None,
        conversation_id: str | None,
        execution_mode: str = "auto",
        budget: dict | None = None,
        run_labels: dict | None = None,
    ) -> None:
        async with get_db_session() as db:
            existing = await db.get(AgentRun, run_id)
            if existing:
                return
            db.add(
                AgentRun(
                    id=run_id,
                    user_id=user_id,
                    mode=mode,
                    objective=objective,
                    task_id=task_id,
                    conversation_id=conversation_id,
                    execution_mode=execution_mode,
                    budget_json=budget or {},
                    run_labels=run_labels or {},
                    status="running",
                )
            )
            await db.commit()

    @_safe
    async def mark_run_status(
        self,
        run_id: str,
        status: str,
        *,
        last_error: str | None = None,
        outcome_label: str | None = None,
        outcome_reason_code: str | None = None,
        quality_score: float | None = None,
    ) -> None:
        async with get_db_session() as db:
            run = await db.get(AgentRun, run_id)
            if not run:
                return
            run.status = status
            run.last_error = last_error
            if outcome_label is not None:
                run.outcome_label = outcome_label
            if outcome_reason_code is not None:
                run.outcome_reason_code = outcome_reason_code
            if quality_score is not None:
                run.quality_score = quality_score
            if status in {"completed", "failed", "cancelled"}:
                run.completed_at = datetime.now(timezone.utc)
            await db.commit()

    @_safe
    async def record_event(
        self,
        *,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
        step_id: str | None = None,
        dedup_key: str | None = None,
    ) -> None:
        payload_json = json.dumps(payload, sort_keys=True, default=str)
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        async with get_db_session() as db:
            db.add(
                AgentEvent(
                    id=str(uuid.uuid4()),
                    run_id=run_id,
                    event_type=event_type,
                    step_id=step_id,
                    dedup_key=dedup_key,
                    payload=payload,
                    payload_hash=payload_hash,
                )
            )
            await db.commit()

    @_safe
    async def record_step(
        self,
        *,
        run_id: str,
        step_id: str,
        step_type: str,
        status: str,
        title: str | None = None,
        details: dict | None = None,
    ) -> None:
        async with get_db_session() as db:
            db.add(
                AgentStep(
                    id=str(uuid.uuid4()),
                    run_id=run_id,
                    step_id=step_id,
                    step_type=step_type,
                    status=status,
                    title=title,
                    details=details or {},
                )
            )
            await db.commit()

    @_safe
    async def record_tool_execution_start(
        self,
        *,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict,
        step_id: str | None = None,
        policy_decision: str | None = None,
        policy_reason: str | None = None,
    ) -> None:
        async with get_db_session() as db:
            existing = await db.execute(
                select(ToolExecution).where(
                    ToolExecution.run_id == run_id,
                    ToolExecution.tool_call_id == tool_call_id,
                )
            )
            row = existing.scalar_one_or_none()
            if row:
                return
            db.add(
                ToolExecution(
                    id=str(uuid.uuid4()),
                    run_id=run_id,
                    step_id=step_id,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    status="running",
                    policy_decision=policy_decision,
                    policy_reason=policy_reason,
                    retry_count=0,
                )
            )
            await db.commit()

    @_safe
    async def record_tool_execution_end(
        self,
        *,
        run_id: str,
        tool_call_id: str,
        status: str,
        result_summary: str | None = None,
        error: str | None = None,
    ) -> None:
        async with get_db_session() as db:
            result = await db.execute(
                select(ToolExecution).where(
                    ToolExecution.run_id == run_id,
                    ToolExecution.tool_call_id == tool_call_id,
                )
            )
            execution = result.scalar_one_or_none()
            if not execution:
                return
            execution.status = status
            execution.result_summary = result_summary
            execution.error = error
            await db.commit()

    @_safe
    async def get_run(self, run_id: str) -> dict | None:
        async with get_db_session() as db:
            run = await db.get(AgentRun, run_id)
            if not run:
                return None
            return run.to_dict()

    @_safe
    async def get_timeline(self, run_id: str) -> list[dict]:
        async with get_db_session() as db:
            result = await db.execute(
                select(AgentEvent)
                .where(AgentEvent.run_id == run_id)
                .order_by(AgentEvent.created_at.asc())
            )
            events = result.scalars().all()
            return [
                {
                    "id": e.id,
                    "run_id": e.run_id,
                    "type": e.event_type,
                    "step_id": e.step_id,
                    "payload": e.payload,
                    "dedup_key": e.dedup_key,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in events
            ]

    async def cancel_run(self, run_id: str) -> bool:
        await self.mark_run_status(run_id, "cancelled")
        return True

    async def resume_run(self, run_id: str) -> bool:
        await self.mark_run_status(run_id, "running", last_error=None)
        return True

    async def delete_run_data(self, run_id: str) -> None:
        """Helper for tests."""
        async with get_db_session() as db:
            await db.execute(delete(AgentEvent).where(AgentEvent.run_id == run_id))
            await db.execute(delete(AgentStep).where(AgentStep.run_id == run_id))
            await db.execute(delete(ToolExecution).where(ToolExecution.run_id == run_id))
            await db.execute(delete(AgentRun).where(AgentRun.id == run_id))
            await db.commit()


run_ledger_service = RunLedgerService()
