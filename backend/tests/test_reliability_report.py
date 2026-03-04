"""Unit tests for reliability report metric computation."""

from types import SimpleNamespace

from app.services.reliability_metrics import compute_reliability_metrics


def _run(id_: str, outcome: str):
    return SimpleNamespace(id=id_, outcome_label=outcome)


def _event(run_id: str, event_type: str, payload: dict):
    return SimpleNamespace(run_id=run_id, event_type=event_type, payload=payload)


def test_compute_metrics_basic():
    runs = [
        _run("r1", "success"),
        _run("r2", "partial"),
        _run("r3", "user_cancelled"),
    ]
    events = [
        _event("r1", "token", {"event_id": "token:a"}),
        _event("r1", "complete", {"event_id": "complete:1"}),
        _event("r2", "stage", {"event_id": "stage:s:running"}),
        _event("r2", "stage", {"event_id": "stage:s:running"}),  # duplicate
    ]
    metrics = compute_reliability_metrics(runs, events)
    assert metrics.total_runs == 3
    assert metrics.eligible_runs == 2
    assert metrics.success_runs == 1
    assert abs(metrics.tsr - 0.5) < 1e-9
    assert metrics.duplicate_event_rate > 0
    assert metrics.missing_complete_rate > 0
