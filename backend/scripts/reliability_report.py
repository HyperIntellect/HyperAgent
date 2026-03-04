"""Generate reliability metrics and optionally enforce quality gates.

Usage:
  uv run python scripts/reliability_report.py
  uv run python scripts/reliability_report.py --days 7 --json-out /tmp/reliability.json
  uv run python scripts/reliability_report.py --gate --min-tsr 0.60 --max-duplicate-rate 0.003
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import timedelta, timezone

from sqlalchemy import select

from app.db.base import get_db_session
from app.db.models import AgentEvent, AgentRun
from app.services.reliability_metrics import ReliabilityMetrics, compute_reliability_metrics


async def _load_runs(days: int) -> list[AgentRun]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    async with get_db_session() as db:
        result = await db.execute(
            select(AgentRun).where(AgentRun.created_at >= since).order_by(AgentRun.created_at.desc())
        )
        return list(result.scalars().all())


async def _load_events_for_runs(run_ids: list[str]) -> list[AgentEvent]:
    if not run_ids:
        return []
    async with get_db_session() as db:
        result = await db.execute(
            select(AgentEvent)
            .where(AgentEvent.run_id.in_(run_ids))
            .order_by(AgentEvent.created_at.asc())
        )
        return list(result.scalars().all())


def _print_markdown(metrics: ReliabilityMetrics, days: int) -> str:
    data = metrics.as_dict()
    return (
        f"# Reliability Report ({days}d)\n\n"
        f"- Total runs: {data['total_runs']}\n"
        f"- Eligible runs: {data['eligible_runs']}\n"
        f"- Success runs: {data['success_runs']}\n"
        f"- TSR: {data['tsr']:.3f}\n"
        f"- Duplicate event rate: {data['duplicate_event_rate']:.4f}\n"
        f"- Missing complete rate: {data['missing_complete_rate']:.4f}\n"
        f"- Generated at: {data['generated_at']}\n"
    )


def _enforce_gate(metrics: ReliabilityMetrics, min_tsr: float, max_dup: float, max_missing_complete: float) -> int:
    errors: list[str] = []
    if metrics.tsr < min_tsr:
        errors.append(f"TSR {metrics.tsr:.3f} < min {min_tsr:.3f}")
    if metrics.duplicate_event_rate > max_dup:
        errors.append(
            f"duplicate_event_rate {metrics.duplicate_event_rate:.4f} > max {max_dup:.4f}"
        )
    if metrics.missing_complete_rate > max_missing_complete:
        errors.append(
            f"missing_complete_rate {metrics.missing_complete_rate:.4f} > max {max_missing_complete:.4f}"
        )
    if errors:
        print("RELIABILITY GATE FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Reliability gate passed.")
    return 0


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Generate reliability metrics from run ledger.")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days.")
    parser.add_argument("--json-out", type=str, default="", help="Optional path to write JSON report.")
    parser.add_argument("--markdown-out", type=str, default="", help="Optional path to write markdown report.")
    parser.add_argument("--gate", action="store_true", help="Enforce metric thresholds and exit non-zero on breach.")
    parser.add_argument("--min-tsr", type=float, default=0.0, help="Minimum task success rate.")
    parser.add_argument(
        "--max-duplicate-rate",
        type=float,
        default=1.0,
        help="Maximum duplicate event rate.",
    )
    parser.add_argument(
        "--max-missing-complete-rate",
        type=float,
        default=1.0,
        help="Maximum runs missing complete event ratio.",
    )
    args = parser.parse_args()

    runs = await _load_runs(args.days)
    events = await _load_events_for_runs([r.id for r in runs])
    metrics = compute_reliability_metrics(runs, events)

    report_json = json.dumps(metrics.as_dict(), indent=2)
    print(report_json)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(report_json + "\n")

    markdown = _print_markdown(metrics, args.days)
    if args.markdown_out:
        with open(args.markdown_out, "w", encoding="utf-8") as f:
            f.write(markdown)

    if args.gate:
        return _enforce_gate(
            metrics,
            min_tsr=args.min_tsr,
            max_dup=args.max_duplicate_rate,
            max_missing_complete=args.max_missing_complete_rate,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
