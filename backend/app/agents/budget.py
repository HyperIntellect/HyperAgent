"""Budget pressure utilities for controlling agent execution costs.

Extracted from supervisor.py so both orchestrator and supervisor can share
the canonical implementations without circular dependencies.
"""

from typing import Any


def derive_budget_pressure_state(
    budget: dict[str, Any],
    usage_totals: dict[str, Any],
    tool_calls_count: int,
    elapsed_seconds: int,
) -> dict[str, Any]:
    """Compute normalized budget pressure signals for runtime reporting."""
    max_tokens = int(budget.get("max_tokens", 0) or 0)
    max_cost = float(budget.get("max_cost_usd", 0.0) or 0.0)
    max_tool_calls = int(budget.get("max_tool_calls", 0) or 0)
    max_wall_clock = int(budget.get("max_wall_clock_seconds", 0) or 0)

    token_ratio = (usage_totals.get("total_tokens", 0) / max_tokens) if max_tokens else 0.0
    cost_ratio = (usage_totals.get("cost_usd", 0.0) / max_cost) if max_cost else 0.0
    tool_ratio = (tool_calls_count / max_tool_calls) if max_tool_calls else 0.0
    wall_ratio = (elapsed_seconds / max_wall_clock) if max_wall_clock else 0.0
    pressure_ratio = max(token_ratio, cost_ratio, tool_ratio, wall_ratio)

    return {
        "exhausted": pressure_ratio >= 1.0,
        "pressure_ratio": round(pressure_ratio, 4),
        "max_tokens": max_tokens,
        "max_cost_usd": max_cost,
        "max_tool_calls": max_tool_calls,
        "max_wall_clock_seconds": max_wall_clock,
        "total_tokens": usage_totals.get("total_tokens", 0),
        "cost_usd": usage_totals.get("cost_usd", 0.0),
        "tool_calls": tool_calls_count,
        "elapsed_seconds": elapsed_seconds,
    }


def apply_budget_pressure_defaults(
    budget: dict[str, Any],
    mode: str | None,
    tier: Any,
    depth: Any,
) -> tuple[Any, Any, dict[str, Any] | None]:
    """Downgrade tier/depth at run start when budgets are very constrained."""
    from app.models.schemas import ResearchDepth

    if not budget:
        return tier, depth, None

    normalized_tier = str(tier.value if hasattr(tier, "value") else (tier or "pro")).lower()
    max_tokens = int(budget.get("max_tokens", 0) or 0)
    max_cost = float(budget.get("max_cost_usd", 0.0) or 0.0)
    max_tool_calls = int(budget.get("max_tool_calls", 0) or 0)

    target_tier = normalized_tier
    reason = None
    # Conservative thresholds to avoid unexpected quality drops.
    if (
        (max_cost and max_cost <= 0.05)
        or (max_tokens and max_tokens <= 5_000)
        or (max_tool_calls and max_tool_calls <= 6)
    ):
        if normalized_tier in {"max", "pro"}:
            target_tier = "lite"
            reason = "strict_budget"
    elif (
        (max_cost and max_cost <= 0.25)
        or (max_tokens and max_tokens <= 20_000)
        or (max_tool_calls and max_tool_calls <= 16)
    ):
        if normalized_tier == "max":
            target_tier = "pro"
            reason = "budget_pressure"

    target_depth = depth
    if str(mode or "").lower() == "research":
        if reason == "strict_budget":
            target_depth = ResearchDepth.FAST

    if target_tier != normalized_tier or target_depth != depth:
        return (
            target_tier,
            target_depth,
            {
                "reason": reason or "budget_pressure",
                "applied_tier": target_tier,
                "applied_depth": str(
                    target_depth.value if hasattr(target_depth, "value") else target_depth
                ),
                "original_tier": normalized_tier,
                "original_depth": (
                    str(depth.value if hasattr(depth, "value") else depth) if depth else None
                ),
            },
        )

    # Normalize tier to string for consistent return type
    return normalized_tier, depth, None
