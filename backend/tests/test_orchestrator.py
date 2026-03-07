"""Tests for Orchestrator graph."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.agents.orchestrator import (
    create_orchestrator_graph,
    classify_node,
    step_check,
    finalize_node,
    should_plan_or_dispatch,
)
from app.agents.state import OrchestratorState, PlanStep, StepResult


def test_create_orchestrator_graph_compiles():
    graph = create_orchestrator_graph()
    assert graph is not None


@pytest.mark.asyncio
async def test_classify_node_simple():
    state: OrchestratorState = {
        "query": "Hello",
        "events": [],
    }
    result = await classify_node(state)
    assert result["query_complexity"] == "simple"


@pytest.mark.asyncio
async def test_classify_node_complex():
    state: OrchestratorState = {
        "query": "First create a backend API, then build a React frontend, and deploy it",
        "events": [],
    }
    result = await classify_node(state)
    assert result["query_complexity"] == "complex"


@pytest.mark.asyncio
async def test_classify_node_dedicated_mode_is_simple():
    state: OrchestratorState = {
        "query": "Build a complex multi-step app",
        "mode": "app",
        "events": [],
    }
    result = await classify_node(state)
    assert result["query_complexity"] == "simple"


def test_should_plan_or_dispatch_simple():
    state: OrchestratorState = {"query_complexity": "simple", "events": []}
    assert should_plan_or_dispatch(state) == "dispatch_step"


def test_should_plan_or_dispatch_complex():
    state: OrchestratorState = {"query_complexity": "complex", "events": []}
    assert should_plan_or_dispatch(state) == "plan"


def test_step_check_simple_done():
    state: OrchestratorState = {
        "query_complexity": "simple",
        "step_results": [
            StepResult(step_number=1, status="completed", output="done", events=[]),
        ],
        "current_step_index": 1,
        "execution_plan": [],
        "events": [],
    }
    result = step_check(state)
    assert result == "finalize"


def test_step_check_more_steps():
    state: OrchestratorState = {
        "query_complexity": "complex",
        "execution_plan": [
            PlanStep(step_number=1, goal="Step 1", tools_hint=[], expected_output=""),
            PlanStep(step_number=2, goal="Step 2", tools_hint=[], expected_output=""),
        ],
        "current_step_index": 1,
        "step_results": [
            StepResult(step_number=1, status="completed", output="done", events=[]),
        ],
        "events": [],
    }
    result = step_check(state)
    assert result == "dispatch_step"


def test_step_check_all_done():
    state: OrchestratorState = {
        "query_complexity": "complex",
        "execution_plan": [
            PlanStep(step_number=1, goal="Step 1", tools_hint=[], expected_output=""),
        ],
        "current_step_index": 1,
        "step_results": [
            StepResult(step_number=1, status="completed", output="done", events=[]),
        ],
        "events": [],
    }
    result = step_check(state)
    assert result == "verify"


@pytest.mark.asyncio
async def test_finalize_node_with_response():
    state: OrchestratorState = {
        "response": "Hello there!",
        "events": [],
    }
    result = await finalize_node(state)
    assert result["response"] == "Hello there!"


@pytest.mark.asyncio
async def test_finalize_node_from_step_results():
    state: OrchestratorState = {
        "step_results": [
            StepResult(step_number=1, status="completed", output="Created API", events=[]),
            StepResult(step_number=2, status="completed", output="Built frontend", events=[]),
        ],
        "events": [],
    }
    result = await finalize_node(state)
    assert "Created API" in result["response"]
    assert "Built frontend" in result["response"]


from app.agents.orchestrator import AgentOrchestrator, agent_orchestrator


def test_agent_orchestrator_instantiates():
    orchestrator = AgentOrchestrator()
    assert orchestrator is not None


def test_agent_orchestrator_global_instance():
    assert agent_orchestrator is not None
