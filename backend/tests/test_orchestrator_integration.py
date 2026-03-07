"""Integration tests for the orchestrator graph (full flow)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.orchestrator import create_orchestrator_graph
from app.agents.state import OrchestratorState, PlanStep


def _base_state(**overrides) -> OrchestratorState:
    state: OrchestratorState = {
        "query": "What is Python?",
        "mode": None,
        "messages": [],
        "events": [],
        "provider": "anthropic",
        "tier": None,
        "locale": "en",
        "skills": [],
        "hitl_enabled": False,
    }
    state.update(overrides)
    return state


def _make_subgraph_mock(return_value=None, side_effect=None):
    """Create a MagicMock with async ainvoke that returns plain dicts."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=return_value, side_effect=side_effect)
    return mock


def _executor_result(result="Done", status="completed"):
    return {
        "result": result,
        "status": status,
        "events": [],
        "pending_handoff": None,
    }


def _planner_result(steps: list[PlanStep]):
    return {
        "plan_steps": steps,
        "complexity_assessment": "complex",
        "events": [],
    }


TWO_STEPS = [
    PlanStep(step_number=1, goal="Create backend API", tools_hint=["execute_code"], expected_output="API running"),
    PlanStep(step_number=2, goal="Build frontend", tools_hint=["app_write_file"], expected_output="UI rendered"),
]


@pytest.mark.asyncio
async def test_simple_query_bypasses_planning():
    """Simple query: classify -> dispatch_step -> finalize (no plan node)."""
    mock_executor = _make_subgraph_mock(return_value=_executor_result("Python is a language"))
    mock_planner = _make_subgraph_mock()

    with (
        patch("app.agents.executor.executor_subgraph", mock_executor),
        patch("app.agents.planner.planner_subgraph", mock_planner),
    ):
        graph = create_orchestrator_graph()
        config = {"configurable": {"thread_id": "test-simple"}, "recursion_limit": 50}
        result = await graph.ainvoke(_base_state(query="What is Python?"), config=config)

    assert result["query_complexity"] == "simple"
    assert "Python is a language" in result["response"]
    mock_executor.ainvoke.assert_called_once()
    mock_planner.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_complex_query_uses_planner_and_executor():
    """Complex query: classify -> plan -> dispatch(x2) -> verify -> finalize."""
    mock_executor = _make_subgraph_mock(side_effect=[
        _executor_result("Backend API created"),
        _executor_result("Frontend built"),
    ])
    mock_planner = _make_subgraph_mock(return_value=_planner_result(TWO_STEPS))

    mock_llm_svc = MagicMock()
    mock_llm_instance = AsyncMock()
    mock_llm_instance.ainvoke.return_value = MagicMock(content="PASS: All goals met")
    mock_llm_svc.get_llm_for_tier.return_value = mock_llm_instance

    with (
        patch("app.agents.executor.executor_subgraph", mock_executor),
        patch("app.agents.planner.planner_subgraph", mock_planner),
        patch("app.agents.orchestrator.llm_service", mock_llm_svc),
    ):
        graph = create_orchestrator_graph()
        config = {"configurable": {"thread_id": "test-complex"}, "recursion_limit": 50}
        result = await graph.ainvoke(
            _base_state(query="Build a weather app with Python backend and React frontend"),
            config=config,
        )

    assert result["query_complexity"] == "complex"
    assert mock_planner.ainvoke.call_count == 1
    assert mock_executor.ainvoke.call_count == 2
    assert result["verification_status"] == "passed"
    assert len(result["step_results"]) == 2


@pytest.mark.asyncio
async def test_dedicated_mode_bypasses_planning():
    """mode='app' always classifies as simple, even for complex-sounding queries."""
    mock_executor = _make_subgraph_mock(return_value=_executor_result("App built"))
    mock_planner = _make_subgraph_mock()

    with (
        patch("app.agents.executor.executor_subgraph", mock_executor),
        patch("app.agents.planner.planner_subgraph", mock_planner),
    ):
        graph = create_orchestrator_graph()
        config = {"configurable": {"thread_id": "test-mode"}, "recursion_limit": 50}
        result = await graph.ainvoke(
            _base_state(
                query="Build a complete weather dashboard with backend and frontend",
                mode="app",
            ),
            config=config,
        )

    assert result["query_complexity"] == "simple"
    mock_planner.ainvoke.assert_not_called()
    mock_executor.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_verification_failure_triggers_replan():
    """Verify FAIL -> re-plan -> verify FAIL again -> finalize (max 2 attempts)."""
    mock_executor = _make_subgraph_mock(return_value=_executor_result("Attempted"))
    mock_planner = _make_subgraph_mock(return_value=_planner_result([
        PlanStep(step_number=1, goal="Try something", tools_hint=[], expected_output="Result"),
    ]))

    mock_llm_svc = MagicMock()
    mock_llm_instance = AsyncMock()
    mock_llm_instance.ainvoke.return_value = MagicMock(content="FAIL: Goals not met")
    mock_llm_svc.get_llm_for_tier.return_value = mock_llm_instance

    with (
        patch("app.agents.executor.executor_subgraph", mock_executor),
        patch("app.agents.planner.planner_subgraph", mock_planner),
        patch("app.agents.orchestrator.llm_service", mock_llm_svc),
    ):
        graph = create_orchestrator_graph()
        config = {"configurable": {"thread_id": "test-replan"}, "recursion_limit": 50}
        result = await graph.ainvoke(
            _base_state(query="First create a backend, then build a frontend"),
            config=config,
        )

    assert result["query_complexity"] == "complex"
    # Planner called at least twice (initial + re-plan)
    assert mock_planner.ainvoke.call_count >= 2
    # Verification capped at 2 attempts before finalize
    assert result["verification_attempts"] >= 2
    # Should still produce a response
    assert result["response"]
