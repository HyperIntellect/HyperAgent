"""Base types for evaluation results."""

from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationResult:
    """Result of an evaluation."""

    key: str
    score: float
    comment: str | None = None
    metadata: dict[str, Any] | None = None


def compute_batch_stats(
    evaluations: list[EvaluationResult],
    categories: list[str],
) -> dict[str, Any]:
    """Compute accuracy and per-category breakdown for a batch of evaluations.

    Args:
        evaluations: List of EvaluationResult objects
        categories: Parallel list of category strings (one per evaluation)

    Returns:
        Dict with accuracy, correct/total counts, per-category accuracy and stats
    """
    if not evaluations:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    category_stats: dict[str, dict[str, int]] = {}

    for evaluation, category in zip(evaluations, categories):
        if category not in category_stats:
            category_stats[category] = {"correct": 0, "total": 0}
        category_stats[category]["total"] += 1
        if evaluation.score == 1.0:
            category_stats[category]["correct"] += 1

    correct = sum(1 for e in evaluations if e.score == 1.0)
    total = len(evaluations)
    accuracy = correct / total if total > 0 else 0.0

    category_accuracy = {
        cat: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for cat, stats in category_stats.items()
    }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "category_accuracy": category_accuracy,
        "category_stats": category_stats,
        "evaluations": evaluations,
    }
