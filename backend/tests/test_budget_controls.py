"""Tests for budget pressure controls in supervisor."""

from app.agents.supervisor import (
    _apply_budget_pressure_defaults,
    _derive_budget_pressure_state,
)
from app.models.schemas import ResearchDepth


def test_derives_budget_pressure_ratio():
    budget_state = _derive_budget_pressure_state(
        budget={"max_tokens": 1000, "max_tool_calls": 10},
        usage_totals={"total_tokens": 500, "cost_usd": 0.0},
        tool_calls_count=2,
        elapsed_seconds=5,
    )
    assert budget_state["pressure_ratio"] == 0.5
    assert budget_state["exhausted"] is False


def test_marks_budget_exhausted():
    budget_state = _derive_budget_pressure_state(
        budget={"max_cost_usd": 0.1},
        usage_totals={"total_tokens": 0, "cost_usd": 0.11},
        tool_calls_count=0,
        elapsed_seconds=0,
    )
    assert budget_state["exhausted"] is True


def test_applies_strict_budget_downgrade_to_flash():
    tier, depth, adjustment = _apply_budget_pressure_defaults(
        budget={"max_cost_usd": 0.03},
        mode="task",
        tier="max",
        depth=ResearchDepth.DEEP,
    )
    assert tier == "flash"
    assert depth == ResearchDepth.DEEP
    assert adjustment is not None
    assert adjustment["reason"] == "strict_budget"


def test_applies_research_depth_downgrade_on_strict_budget():
    tier, depth, adjustment = _apply_budget_pressure_defaults(
        budget={"max_tokens": 3000},
        mode="research",
        tier="max",
        depth=ResearchDepth.DEEP,
    )
    assert tier == "flash"
    assert depth == ResearchDepth.FAST
    assert adjustment is not None


def test_applies_moderate_downgrade_max_to_pro():
    tier, depth, adjustment = _apply_budget_pressure_defaults(
        budget={"max_tokens": 15000},
        mode="task",
        tier="max",
        depth=ResearchDepth.FAST,
    )
    assert tier == "pro"
    assert depth == ResearchDepth.FAST
    assert adjustment is not None
