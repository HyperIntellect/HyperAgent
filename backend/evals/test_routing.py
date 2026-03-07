"""Routing accuracy evaluation tests."""

import pytest

from evals.conftest import ROUTING_ACCURACY_THRESHOLD, ROUTING_CATEGORY_THRESHOLD
from evals.evaluators.routing_evaluator import RoutingEvaluator


class TestRoutingAccuracy:
    """Test suite for routing accuracy evaluation."""

    @pytest.mark.asyncio
    async def test_routing_accuracy_with_mock(self, routing_cases, mock_supervisor):
        """Test routing accuracy using mock supervisor.

        Args:
            routing_cases: Test cases from routing.json
            mock_supervisor: Mock supervisor fixture
        """
        results = []

        for case in routing_cases:
            result = await mock_supervisor.route(case["input"])
            passed = result["next_agent"].lower() == case["expected_agent"].lower()
            results.append(
                {
                    "id": case["id"],
                    "input": case["input"],
                    "expected_agent": case["expected_agent"],
                    "actual_agent": result["next_agent"],
                    "passed": passed,
                    "category": case.get("category", "unknown"),
                }
            )

        # Calculate accuracy
        accuracy = sum(r["passed"] for r in results) / len(results)

        # Print detailed results for debugging
        passed_count = sum(r["passed"] for r in results)
        print(f"\n{'=' * 60}")
        print(f"Routing Accuracy: {accuracy:.1%} ({passed_count}/{len(results)})")
        print(f"{'=' * 60}")

        # Show failures
        failures = [r for r in results if not r["passed"]]
        if failures:
            print("\nFailures:")
            for f in failures:
                print(f"  - {f['id']}: expected '{f['expected_agent']}', got '{f['actual_agent']}'")
                print(f"    Input: {f['input'][:60]}...")

        # Assert minimum accuracy threshold
        assert accuracy >= ROUTING_ACCURACY_THRESHOLD, (
            f"Routing accuracy {accuracy:.1%} < {ROUTING_ACCURACY_THRESHOLD:.0%} threshold"
        )

    @pytest.mark.asyncio
    async def test_routing_by_category(self, routing_cases, mock_supervisor):
        """Test routing accuracy broken down by category.

        Args:
            routing_cases: Test cases from routing.json
            mock_supervisor: Mock supervisor fixture
        """
        category_results: dict[str, list[bool]] = {}

        for case in routing_cases:
            result = await mock_supervisor.route(case["input"])
            passed = result["next_agent"].lower() == case["expected_agent"].lower()

            category = case.get("category", "unknown")
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(passed)

        print(f"\n{'=' * 60}")
        print("Routing Accuracy by Category:")
        print(f"{'=' * 60}")

        for category, results in sorted(category_results.items()):
            accuracy = sum(results) / len(results)
            print(f"  {category}: {accuracy:.1%} ({sum(results)}/{len(results)})")

            # Each category should have at least threshold accuracy
            assert accuracy >= ROUTING_CATEGORY_THRESHOLD, (
                f"Category '{category}' accuracy {accuracy:.1%} < {ROUTING_CATEGORY_THRESHOLD:.0%}"
            )

    def test_routing_evaluator_correct(self):
        """Test RoutingEvaluator with correct routing."""
        evaluator = RoutingEvaluator()
        result = evaluator.evaluate(
            expected_agent="task",
            actual_agent="task",
            query="Hello world",
        )

        assert result.score == 1.0
        assert result.metadata["correct"] is True

    def test_routing_evaluator_incorrect(self):
        """Test RoutingEvaluator with incorrect routing."""
        evaluator = RoutingEvaluator()
        result = evaluator.evaluate(
            expected_agent="research",
            actual_agent="task",
            query="Write a comprehensive research paper",
        )

        assert result.score == 0.0
        assert result.metadata["correct"] is False

    def test_routing_evaluator_case_insensitive(self):
        """Test RoutingEvaluator is case-insensitive."""
        evaluator = RoutingEvaluator()
        result = evaluator.evaluate(
            expected_agent="TASK",
            actual_agent="task",
            query="Hello",
        )

        assert result.score == 1.0

    def test_routing_evaluator_batch(self, routing_cases):
        """Test batch evaluation functionality."""
        evaluator = RoutingEvaluator()

        # Create mock results
        mock_results = [
            {
                "expected_agent": case["expected_agent"],
                "actual_agent": case["expected_agent"],  # All correct
                "query": case["input"],
                "category": case.get("category", "unknown"),
            }
            for case in routing_cases[:5]
        ]

        summary = evaluator.evaluate_batch(mock_results)

        assert summary["accuracy"] == 1.0
        assert summary["correct"] == 5
        assert summary["total"] == 5


class TestRoutingWithClassifier:
    """Test routing using the real classify_query() heuristic (Issue 4).

    This tests the actual production routing logic rather than mock patterns.
    classify_query returns 'simple' (→ task) or 'complex' (→ research/planner).
    """

    def test_classifier_on_routing_dataset(self, routing_cases):
        """Test classify_query against the routing dataset."""
        from app.agents.classifier import classify_query

        results = []

        for case in routing_cases:
            classification = classify_query(case["input"])
            # In the orchestrator: simple → ExecutorAgent (task), complex → PlannerAgent
            # Research cases are complex; everything else is simple
            expected_complexity = "complex" if case["expected_agent"] == "research" else "simple"
            passed = classification == expected_complexity
            results.append(
                {
                    "id": case["id"],
                    "input": case["input"],
                    "expected": expected_complexity,
                    "actual": classification,
                    "passed": passed,
                }
            )

        accuracy = sum(r["passed"] for r in results) / len(results)

        print(f"\n{'=' * 60}")
        print(f"Classifier Accuracy: {accuracy:.1%}")
        print(f"{'=' * 60}")

        failures = [r for r in results if not r["passed"]]
        if failures:
            print("\nClassifier failures:")
            for f in failures:
                print(f"  - {f['id']}: expected '{f['expected']}', got '{f['actual']}'")
                print(f"    Input: {f['input'][:60]}...")

        # The heuristic classifier may not achieve the same threshold as mock routing,
        # but should be reasonable
        assert accuracy >= ROUTING_CATEGORY_THRESHOLD, (
            f"Classifier accuracy {accuracy:.1%} < {ROUTING_CATEGORY_THRESHOLD:.0%}"
        )

    def test_classifier_simple_queries(self):
        """Simple queries should classify as 'simple'."""
        from app.agents.classifier import classify_query

        simple_queries = [
            "Hello, how are you?",
            "What is Python?",
            "Hi there",
        ]

        for query in simple_queries:
            result = classify_query(query)
            assert result == "simple", f"Expected 'simple' for: {query}"

    def test_classifier_complex_queries(self):
        """Complex/research queries should classify as 'complex'."""
        from app.agents.classifier import classify_query

        complex_queries = [
            "Write a comprehensive 20-page research paper on quantum computing with citations",
            "Build a complete dashboard application with authentication and real-time updates",
            "First research the topic, then create a detailed outline, and finally write the paper",
        ]

        for query in complex_queries:
            result = classify_query(query)
            assert result == "complex", f"Expected 'complex' for: {query}"

    def test_classifier_skill_mode_override(self):
        """Explicit skill mode should always return 'simple'."""
        from app.agents.classifier import classify_query

        # Even a complex-sounding query with explicit mode should be simple
        result = classify_query("Create a comprehensive multi-page presentation", mode="slide")
        assert result == "simple"


class TestRoutingEdgeCases:
    """Test edge cases for routing."""

    @pytest.mark.asyncio
    async def test_empty_query(self, mock_supervisor):
        """Empty query should default to task."""
        result = await mock_supervisor.route("")
        assert result["next_agent"] == "task"

    @pytest.mark.asyncio
    async def test_ambiguous_query(self, mock_supervisor):
        """Ambiguous query should route to task as default."""
        result = await mock_supervisor.route("help")
        assert result["next_agent"] == "task"

    @pytest.mark.asyncio
    async def test_mixed_signals_query(self, mock_supervisor):
        """Query with mixed signals should pick the dominant one."""
        # This query mentions research but is really about task
        result = await mock_supervisor.route("Can you quickly search for information about AI?")
        assert result["next_agent"] == "task"


class TestRoutingSpecificAgents:
    """Test routing to specific agents."""

    @pytest.mark.asyncio
    async def test_research_agent_routing(self, mock_supervisor):
        """Test queries that should route to research agent."""
        research_queries = [
            "Write a comprehensive 20-page research paper on quantum computing with citations",
            "Create an academic literature review with 30+ citations from peer-reviewed sources",
            "Conduct comprehensive market analysis with competitor research",
        ]

        for query in research_queries:
            result = await mock_supervisor.route(query)
            msg = f"Query should route to research: {query[:50]}..."
            assert result["next_agent"] == "research", msg

    @pytest.mark.asyncio
    async def test_data_queries_route_to_task(self, mock_supervisor):
        """Test queries about data analysis route to task agent (has data_analysis skill)."""
        data_queries = [
            "Analyze this CSV file and create visualizations",
            "Process this Excel spreadsheet and find trends",
            "Run statistical analysis on this dataset",
        ]

        for query in data_queries:
            result = await mock_supervisor.route(query)
            assert result["next_agent"] == "task", f"Query should route to task: {query[:50]}..."

    @pytest.mark.asyncio
    async def test_task_agent_routing(self, mock_supervisor):
        """Test queries that should route to task agent."""
        task_queries = [
            "Hello, how are you?",
            "Generate an image of a sunset",
            "Write a Python function for sorting",
            "What's the weather today?",
        ]

        for query in task_queries:
            result = await mock_supervisor.route(query)
            assert result["next_agent"] == "task", f"Query should route to task: {query[:50]}..."


class TestRoutingLangSmith:
    """LangSmith integration tests for routing (Issue 2).

    These tests exercise the LangSmith-compatible evaluator functions
    and are marked with @pytest.mark.langsmith_integration so they can be run
    separately via `make eval-langsmith`.
    """

    @pytest.mark.langsmith_integration
    def test_routing_accuracy_evaluator_function(self, langsmith_client):
        """Test the LangSmith-compatible routing evaluator function."""
        from evals.evaluators import routing_accuracy_evaluator

        class MockRun:
            def __init__(self, outputs):
                self.outputs = outputs

        class MockExample:
            def __init__(self, inputs, outputs):
                self.inputs = inputs
                self.outputs = outputs

        run = MockRun(outputs={"routed_agent": "task"})
        example = MockExample(
            inputs={"query": "Hello, how are you?"},
            outputs={"expected_agent": "task"},
        )

        result = routing_accuracy_evaluator(run, example)

        assert result.score == 1.0
        assert result.key == "routing_accuracy"

    @pytest.mark.langsmith_integration
    def test_routing_accuracy_evaluator_incorrect(self, langsmith_client):
        """Test routing evaluator function with incorrect routing."""
        from evals.evaluators import routing_accuracy_evaluator

        class MockRun:
            def __init__(self, outputs):
                self.outputs = outputs

        class MockExample:
            def __init__(self, inputs, outputs):
                self.inputs = inputs
                self.outputs = outputs

        run = MockRun(outputs={"routed_agent": "task"})
        example = MockExample(
            inputs={"query": "Write a comprehensive research paper"},
            outputs={"expected_agent": "research"},
        )

        result = routing_accuracy_evaluator(run, example)

        assert result.score == 0.0
