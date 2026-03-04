"""Reliability metric helpers shared by scripts and tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class ReliabilityMetrics:
    total_runs: int
    eligible_runs: int
    success_runs: int
    blocked_runs: int
    cancelled_runs: int
    duplicate_event_rate: float
    missing_complete_rate: float
    tsr: float

    def as_dict(self) -> dict:
        return {
            "total_runs": self.total_runs,
            "eligible_runs": self.eligible_runs,
            "success_runs": self.success_runs,
            "blocked_runs": self.blocked_runs,
            "cancelled_runs": self.cancelled_runs,
            "duplicate_event_rate": self.duplicate_event_rate,
            "missing_complete_rate": self.missing_complete_rate,
            "tsr": self.tsr,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


def compute_reliability_metrics(runs, events) -> ReliabilityMetrics:
    total_runs = len(runs)
    cancelled_runs = sum(1 for r in runs if (getattr(r, "outcome_label", "") or "") == "user_cancelled")
    blocked_runs = sum(1 for r in runs if (getattr(r, "outcome_label", "") or "") == "blocked")
    eligible_runs = max(total_runs - cancelled_runs, 0)
    success_runs = sum(1 for r in runs if (getattr(r, "outcome_label", "") or "") == "success")
    tsr = (success_runs / eligible_runs) if eligible_runs else 0.0

    by_run: dict[str, list] = {}
    for ev in events:
        by_run.setdefault(ev.run_id, []).append(ev)

    duplicate_count = 0
    considered_count = 0
    missing_complete = 0
    dedupe_types = {"token", "tool_call", "tool_result", "stage", "reasoning", "source"}
    for run in runs:
        run_id = getattr(run, "id")
        run_events = by_run.get(run_id, [])
        seen_ids: set[str] = set()
        has_complete = False
        for ev in run_events:
            if ev.event_type == "complete":
                has_complete = True
            if ev.event_type in dedupe_types:
                considered_count += 1
                event_id = str((ev.payload or {}).get("event_id") or "")
                if event_id:
                    if event_id in seen_ids:
                        duplicate_count += 1
                    else:
                        seen_ids.add(event_id)
        if (getattr(run, "outcome_label", "") or "") != "user_cancelled" and not has_complete:
            missing_complete += 1

    duplicate_event_rate = (duplicate_count / considered_count) if considered_count else 0.0
    missing_complete_rate = (missing_complete / eligible_runs) if eligible_runs else 0.0

    return ReliabilityMetrics(
        total_runs=total_runs,
        eligible_runs=eligible_runs,
        success_runs=success_runs,
        blocked_runs=blocked_runs,
        cancelled_runs=cancelled_runs,
        duplicate_event_rate=duplicate_event_rate,
        missing_complete_rate=missing_complete_rate,
        tsr=tsr,
    )

