"""Tool and skill selection evaluation tests."""

import pytest

from evals.conftest import TOOL_SELECTION_ACCURACY_THRESHOLD
from evals.evaluators.tool_selection_evaluator import ToolSelectionEvaluator


class TestToolSelection:
    """Test suite for tool and skill selection evaluation."""

    @pytest.mark.asyncio
    async def test_tool_selection_accuracy(self, tool_selection_cases, mock_chat_agent):
        """Test tool selection accuracy using mock chat agent.

        Args:
            tool_selection_cases: Test cases from tool_selection.json
            mock_chat_agent: Mock chat agent fixture
        """
        evaluator = ToolSelectionEvaluator()
        results = []

        for case in tool_selection_cases:
            agent_result = await mock_chat_agent.process(case["input"])

            eval_result = evaluator.evaluate(
                expected_skill=case.get("expected_skill"),
                expected_tool=case.get("expected_tool"),
                tool_calls=agent_result["tool_calls"],
                query=case["input"],
            )

            results.append(
                {
                    "id": case["id"],
                    "input": case["input"],
                    "expected_skill": case.get("expected_skill"),
                    "expected_tool": case.get("expected_tool"),
                    "actual_tools": [tc["name"] for tc in agent_result["tool_calls"]],
                    "passed": eval_result.score == 1.0,
                    "category": case.get("category", "unknown"),
                }
            )

        # Calculate accuracy
        accuracy = sum(r["passed"] for r in results) / len(results)

        # Print detailed results
        passed_count = sum(r["passed"] for r in results)
        print(f"\n{'=' * 60}")
        print(f"Tool Selection Accuracy: {accuracy:.1%} ({passed_count}/{len(results)})")
        print(f"{'=' * 60}")

        # Show failures
        failures = [r for r in results if not r["passed"]]
        if failures:
            print("\nFailures:")
            for f in failures:
                expected = f["expected_skill"] or f["expected_tool"] or "no tools"
                print(f"  - {f['id']}: expected '{expected}', got {f['actual_tools']}")
                print(f"    Input: {f['input'][:60]}...")

        # Assert minimum accuracy threshold
        assert accuracy >= TOOL_SELECTION_ACCURACY_THRESHOLD, (
            f"Tool selection accuracy {accuracy:.1%} "
            f"< {TOOL_SELECTION_ACCURACY_THRESHOLD:.0%} threshold"
        )

    @pytest.mark.asyncio
    async def test_tool_selection_by_category(self, tool_selection_cases, mock_chat_agent):
        """Test tool selection accuracy broken down by category.

        Args:
            tool_selection_cases: Test cases from tool_selection.json
            mock_chat_agent: Mock chat agent fixture
        """
        evaluator = ToolSelectionEvaluator()
        category_results: dict[str, list[bool]] = {}

        for case in tool_selection_cases:
            agent_result = await mock_chat_agent.process(case["input"])

            eval_result = evaluator.evaluate(
                expected_skill=case.get("expected_skill"),
                expected_tool=case.get("expected_tool"),
                tool_calls=agent_result["tool_calls"],
            )

            category = case.get("category", "unknown")
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(eval_result.score == 1.0)

        print(f"\n{'=' * 60}")
        print("Tool Selection Accuracy by Category:")
        print(f"{'=' * 60}")

        for category, results in sorted(category_results.items()):
            accuracy = sum(results) / len(results)
            print(f"  {category}: {accuracy:.1%} ({sum(results)}/{len(results)})")

    def test_tool_selection_evaluator_skill_correct(self):
        """Test ToolSelectionEvaluator with correct skill selection."""
        evaluator = ToolSelectionEvaluator()
        result = evaluator.evaluate(
            expected_skill="image_generation",
            expected_tool=None,
            tool_calls=[
                {
                    "name": "invoke_skill",
                    "args": {"skill_id": "image_generation", "params": {}},
                    "id": "call_1",
                }
            ],
        )

        assert result.score == 1.0
        assert result.metadata["correct"] is True

    def test_tool_selection_evaluator_skill_incorrect(self):
        """Test ToolSelectionEvaluator with incorrect skill selection."""
        evaluator = ToolSelectionEvaluator()
        result = evaluator.evaluate(
            expected_skill="image_generation",
            expected_tool=None,
            tool_calls=[
                {
                    "name": "invoke_skill",
                    "args": {"skill_id": "code_generation", "params": {}},
                    "id": "call_1",
                }
            ],
        )

        assert result.score == 0.0
        assert result.metadata["correct"] is False

    def test_tool_selection_evaluator_tool_correct(self):
        """Test ToolSelectionEvaluator with correct tool selection."""
        evaluator = ToolSelectionEvaluator()
        result = evaluator.evaluate(
            expected_skill=None,
            expected_tool="web_search",
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"query": "test"},
                    "id": "call_1",
                }
            ],
        )

        assert result.score == 1.0

    def test_tool_selection_evaluator_no_tools_correct(self):
        """Test ToolSelectionEvaluator when no tools expected."""
        evaluator = ToolSelectionEvaluator()
        result = evaluator.evaluate(
            expected_skill=None,
            expected_tool=None,
            tool_calls=[],
        )

        assert result.score == 1.0

    def test_tool_selection_evaluator_no_tools_incorrect(self):
        """Test ToolSelectionEvaluator when no tools expected but tools used."""
        evaluator = ToolSelectionEvaluator()
        result = evaluator.evaluate(
            expected_skill=None,
            expected_tool=None,
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {},
                    "id": "call_1",
                }
            ],
        )

        assert result.score == 0.0

    def test_tool_selection_evaluator_batch(self, tool_selection_cases):
        """Test batch evaluation functionality."""
        evaluator = ToolSelectionEvaluator()

        # Create mock results with correct selections
        mock_results = []
        for case in tool_selection_cases[:5]:
            tool_calls = []
            if case.get("expected_skill"):
                tool_calls = [
                    {
                        "name": "invoke_skill",
                        "args": {"skill_id": case["expected_skill"]},
                        "id": "call_1",
                    }
                ]
            elif case.get("expected_tool"):
                tool_calls = [
                    {
                        "name": case["expected_tool"],
                        "args": {},
                        "id": "call_1",
                    }
                ]

            mock_results.append(
                {
                    "expected_skill": case.get("expected_skill"),
                    "expected_tool": case.get("expected_tool"),
                    "tool_calls": tool_calls,
                    "category": case.get("category", "unknown"),
                }
            )

        summary = evaluator.evaluate_batch(mock_results)

        assert summary["accuracy"] == 1.0
        assert summary["correct"] == 5


class TestSkillSelection:
    """Test skill selection specifically."""

    @pytest.mark.asyncio
    async def test_image_generation_skill(self, mock_chat_agent):
        """Test image generation skill selection."""
        result = await mock_chat_agent.process("Generate a beautiful image of a mountain landscape")

        skill_ids = [
            tc["args"].get("skill_id")
            for tc in result["tool_calls"]
            if tc["name"] == "invoke_skill"
        ]

        assert "image_generation" in skill_ids

    @pytest.mark.asyncio
    async def test_code_generation_skill(self, mock_chat_agent):
        """Test code generation skill selection."""
        result = await mock_chat_agent.process("Write a Python function to calculate factorial")

        skill_ids = [
            tc["args"].get("skill_id")
            for tc in result["tool_calls"]
            if tc["name"] == "invoke_skill"
        ]

        assert "code_generation" in skill_ids


class TestToolSelectionEdgeCases:
    """Test edge cases for tool selection."""

    @pytest.mark.asyncio
    async def test_no_tools_for_greeting(self, mock_chat_agent):
        """Greeting should not invoke any tools."""
        result = await mock_chat_agent.process("Hello, how are you?")
        assert len(result["tool_calls"]) == 0

    @pytest.mark.asyncio
    async def test_no_tools_for_simple_question(self, mock_chat_agent):
        """Simple knowledge question should not invoke tools."""
        result = await mock_chat_agent.process("What is 2 + 2?")
        assert len(result["tool_calls"]) == 0

    @pytest.mark.asyncio
    async def test_multiple_tools_possible(self, mock_chat_agent):
        """Query that could use multiple tools should pick at least one relevant tool."""
        result = await mock_chat_agent.process(
            "Create a chart showing the data from web search results"
        )
        # Should pick at least one relevant tool (Issue 7: was >= 0 which is always true)
        tool_names = [tc["name"] for tc in result["tool_calls"]]
        assert len(tool_names) >= 1, "Expected at least one tool call for a multi-tool query"


class TestToolSelectionLangSmith:
    """LangSmith integration tests for tool selection (Issue 2)."""

    @pytest.mark.langsmith_integration
    def test_tool_selection_evaluator_function(self, langsmith_client):
        """Test the LangSmith-compatible tool selection evaluator function."""
        from evals.evaluators import tool_selection_evaluator

        class MockRun:
            def __init__(self, outputs):
                self.outputs = outputs

        class MockExample:
            def __init__(self, inputs, outputs):
                self.inputs = inputs
                self.outputs = outputs

        run = MockRun(
            outputs={
                "tool_calls": [
                    {
                        "name": "invoke_skill",
                        "args": {"skill_id": "image_generation"},
                        "id": "call_1",
                    }
                ]
            }
        )
        example = MockExample(
            inputs={"query": "Generate an image of a sunset"},
            outputs={"expected_skill": "image_generation", "expected_tool": None},
        )

        result = tool_selection_evaluator(run, example)

        assert result.score == 1.0
        assert result.key == "tool_selection"

    @pytest.mark.langsmith_integration
    def test_tool_selection_evaluator_no_tools(self, langsmith_client):
        """Test tool selection evaluator when no tools expected."""
        from evals.evaluators import tool_selection_evaluator

        class MockRun:
            def __init__(self, outputs):
                self.outputs = outputs

        class MockExample:
            def __init__(self, inputs, outputs):
                self.inputs = inputs
                self.outputs = outputs

        run = MockRun(outputs={"tool_calls": []})
        example = MockExample(
            inputs={"query": "Hello"},
            outputs={"expected_skill": None, "expected_tool": None},
        )

        result = tool_selection_evaluator(run, example)

        assert result.score == 1.0
