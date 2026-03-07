"""Tests for new state type definitions."""
from app.agents.state import (
    ExecutorState,
    OrchestratorState,
    PlannerState,
    PlanStep,
    StepResult,
)


def test_plan_step_structure():
    step = PlanStep(
        step_number=1,
        goal="Create backend API",
        tools_hint=["execute_code"],
        expected_output="FastAPI server running",
    )
    assert step["step_number"] == 1
    assert step["goal"] == "Create backend API"
    assert step["tools_hint"] == ["execute_code"]


def test_step_result_structure():
    result = StepResult(
        step_number=1,
        status="completed",
        output="Created app.py",
        events=[],
    )
    assert result["status"] == "completed"


def test_orchestrator_state_has_plan_fields():
    state: OrchestratorState = {
        "query": "test",
        "query_complexity": "simple",
        "execution_plan": [],
        "current_step_index": 0,
        "step_results": [],
        "events": [],
    }
    assert state["query_complexity"] == "simple"


def test_planner_state_has_output_fields():
    state: PlannerState = {
        "query": "build an app",
        "plan_steps": [],
        "complexity_assessment": "complex",
        "events": [],
    }
    assert state["complexity_assessment"] == "complex"


def test_executor_state_has_step_goal():
    state: ExecutorState = {
        "step_goal": "Create the API endpoint",
        "step_number": 1,
        "lc_messages": [],
        "tool_iterations": 0,
        "consecutive_errors": 0,
        "events": [],
    }
    assert state["step_goal"] == "Create the API endpoint"
