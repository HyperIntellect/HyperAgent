"""Routing accuracy evaluator for agent routing decisions."""

from typing import Any

from evals.evaluators.base import EvaluationResult, compute_batch_stats


class RoutingEvaluator:
    """Evaluator for routing accuracy.

    Measures whether the supervisor routes queries to the correct agent.
    """

    def __init__(self):
        """Initialize the routing evaluator."""
        self.name = "routing_accuracy"

    def evaluate(
        self,
        expected_agent: str,
        actual_agent: str,
        query: str | None = None,
    ) -> EvaluationResult:
        """Evaluate routing accuracy for a single case.

        Args:
            expected_agent: The expected agent to route to
            actual_agent: The actual agent that was selected
            query: Optional query for context in the comment

        Returns:
            EvaluationResult with score 1.0 if correct, 0.0 otherwise
        """
        is_correct = actual_agent.lower() == expected_agent.lower()
        score = 1.0 if is_correct else 0.0

        comment = f"Expected: {expected_agent}, Got: {actual_agent}"
        if query:
            truncated = query[:50] + "..." if len(query) > 50 else query
            comment = f"{comment} (Query: {truncated})"

        return EvaluationResult(
            key=self.name,
            score=score,
            comment=comment,
            metadata={
                "expected": expected_agent,
                "actual": actual_agent,
                "correct": is_correct,
            },
        )

    def evaluate_batch(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate a batch of routing results.

        Args:
            results: List of dicts with 'expected_agent', 'actual_agent', and optional 'query'

        Returns:
            Summary statistics including accuracy, correct/total counts, and per-category breakdown
        """
        if not results:
            return {"accuracy": 0.0, "correct": 0, "total": 0}

        evaluations = []
        categories = []

        for result in results:
            eval_result = self.evaluate(
                expected_agent=result["expected_agent"],
                actual_agent=result["actual_agent"],
                query=result.get("query"),
            )
            evaluations.append(eval_result)
            categories.append(result.get("category", "unknown"))

        return compute_batch_stats(evaluations, categories)


def routing_accuracy_evaluator(
    run: Any,
    example: Any,
) -> EvaluationResult:
    """LangSmith-compatible evaluator function for routing accuracy.

    Args:
        run: LangSmith run object with outputs
        example: LangSmith example with expected outputs

    Returns:
        EvaluationResult for LangSmith
    """
    expected = example.outputs.get("expected_agent", "task")
    actual = run.outputs.get("routed_agent", run.outputs.get("selected_agent", "unknown"))

    evaluator = RoutingEvaluator()
    return evaluator.evaluate(
        expected_agent=expected,
        actual_agent=actual,
        query=example.inputs.get("query", ""),
    )
