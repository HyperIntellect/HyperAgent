"""Tool and skill selection evaluator for agent tool choices."""

from typing import Any

from evals.evaluators.base import EvaluationResult, compute_batch_stats


class ToolSelectionEvaluator:
    """Evaluator for tool and skill selection accuracy.

    Measures whether agents select the appropriate tools or skills for tasks.
    """

    def __init__(self):
        """Initialize the tool selection evaluator."""
        self.name = "tool_selection"

    def evaluate(
        self,
        expected_skill: str | None,
        expected_tool: str | None,
        tool_calls: list[dict[str, Any]],
        query: str | None = None,
    ) -> EvaluationResult:
        """Evaluate tool/skill selection for a single case.

        Args:
            expected_skill: Expected skill ID (e.g., 'image_generation')
            expected_tool: Expected tool name (e.g., 'web_search')
            tool_calls: List of actual tool calls made
            query: Optional query for context

        Returns:
            EvaluationResult with score based on correct selection
        """
        # Extract selected tools and skills from tool calls
        selected_tools = [tc.get("name", "") for tc in tool_calls]
        selected_skills = []

        for tc in tool_calls:
            if tc.get("name") == "invoke_skill":
                args = tc.get("args", {})
                skill_id = args.get("skill_id", "")
                if skill_id:
                    selected_skills.append(skill_id)

        # Determine correctness based on expectations
        if expected_skill:
            # Should have invoked the expected skill
            is_correct = expected_skill in selected_skills
            actual = f"skills: {selected_skills}" if selected_skills else "no skills invoked"
            expected = f"skill: {expected_skill}"
        elif expected_tool:
            # Should have used the expected tool
            is_correct = expected_tool in selected_tools
            actual = f"tools: {selected_tools}" if selected_tools else "no tools used"
            expected = f"tool: {expected_tool}"
        else:
            # Should not have used any tools
            is_correct = len(tool_calls) == 0
            actual = f"tools: {selected_tools}" if tool_calls else "no tools"
            expected = "no tools"

        score = 1.0 if is_correct else 0.0
        comment = f"Expected: {expected}, Got: {actual}"

        return EvaluationResult(
            key=self.name,
            score=score,
            comment=comment,
            metadata={
                "expected_skill": expected_skill,
                "expected_tool": expected_tool,
                "selected_tools": selected_tools,
                "selected_skills": selected_skills,
                "correct": is_correct,
            },
        )

    def evaluate_batch(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate a batch of tool selection results.

        Args:
            results: List of dicts with expectations and actual tool calls

        Returns:
            Summary statistics including accuracy and breakdowns
        """
        if not results:
            return {"accuracy": 0.0, "correct": 0, "total": 0}

        evaluations = []
        categories = []

        for result in results:
            eval_result = self.evaluate(
                expected_skill=result.get("expected_skill"),
                expected_tool=result.get("expected_tool"),
                tool_calls=result.get("tool_calls", []),
                query=result.get("query"),
            )
            evaluations.append(eval_result)
            categories.append(result.get("category", "unknown"))

        return compute_batch_stats(evaluations, categories)


def tool_selection_evaluator(
    run: Any,
    example: Any,
) -> EvaluationResult:
    """LangSmith-compatible evaluator function for tool selection.

    Args:
        run: LangSmith run object with outputs
        example: LangSmith example with expected outputs

    Returns:
        EvaluationResult for LangSmith
    """
    expected_skill = example.outputs.get("expected_skill")
    expected_tool = example.outputs.get("expected_tool")
    tool_calls = run.outputs.get("tool_calls", [])

    evaluator = ToolSelectionEvaluator()
    return evaluator.evaluate(
        expected_skill=expected_skill,
        expected_tool=expected_tool,
        tool_calls=tool_calls,
        query=example.inputs.get("query", ""),
    )
