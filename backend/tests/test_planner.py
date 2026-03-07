"""Tests for PlannerAgent subgraph."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.planner import (
    PLANNER_PROMPT,
    _parse_plan_steps,
    create_planner_graph,
    planner_node,
)
from app.agents.state import PlanStep, PlannerState


def test_create_planner_graph_compiles():
    graph = create_planner_graph()
    assert graph is not None


def test_planner_prompt_exists():
    assert "step" in PLANNER_PROMPT.lower() or "decompose" in PLANNER_PROMPT.lower()


def test_parse_plan_steps_json():
    content = '''[
        {"step_number": 1, "goal": "Create API", "tools_hint": ["execute_code"], "expected_output": "API running"},
        {"step_number": 2, "goal": "Build frontend", "tools_hint": ["app_write_file"], "expected_output": "UI rendered"}
    ]'''
    steps = _parse_plan_steps(content)
    assert len(steps) == 2
    assert steps[0]["goal"] == "Create API"
    assert steps[1]["tools_hint"] == ["app_write_file"]


def test_parse_plan_steps_markdown_wrapped():
    content = '''```json
[
    {"step_number": 1, "goal": "Do thing", "tools_hint": [], "expected_output": "Done"}
]
```'''
    steps = _parse_plan_steps(content)
    assert len(steps) == 1
    assert steps[0]["goal"] == "Do thing"


def test_parse_plan_steps_empty_raises():
    with pytest.raises(ValueError):
        _parse_plan_steps("[]")


@pytest.mark.asyncio
async def test_planner_node_returns_plan_steps():
    mock_response = MagicMock()
    mock_response.content = '''```json
[
    {"step_number": 1, "goal": "Create API", "tools_hint": ["execute_code"], "expected_output": "API running"},
    {"step_number": 2, "goal": "Build frontend", "tools_hint": ["app_write_file"], "expected_output": "UI rendered"}
]
```'''

    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    state: PlannerState = {
        "query": "Build a web app with API and frontend",
        "provider": "anthropic",
        "events": [],
    }

    with patch("app.agents.planner.llm_service") as mock_service:
        mock_service.choose_llm_for_task.return_value = mock_llm
        result = await planner_node(state)

    assert len(result["plan_steps"]) == 2
    assert result["plan_steps"][0]["goal"] == "Create API"
    assert result["plan_steps"][1]["goal"] == "Build frontend"


@pytest.mark.asyncio
async def test_planner_node_fallback_on_error():
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM failed"))

    state: PlannerState = {
        "query": "Build something",
        "provider": "anthropic",
        "events": [],
    }

    with patch("app.agents.planner.llm_service") as mock_service:
        mock_service.choose_llm_for_task.return_value = mock_llm
        result = await planner_node(state)

    # Should fallback to single step
    assert len(result["plan_steps"]) == 1
    assert result["plan_steps"][0]["goal"] == "Build something"
    assert result["complexity_assessment"] == "fallback_single_step"
