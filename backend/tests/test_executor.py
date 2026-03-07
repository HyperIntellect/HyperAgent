"""Tests for ExecutorAgent subgraph."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.executor import (
    create_executor_graph,
    executor_reason_node,
    executor_complete_node,
    executor_should_continue,
)
from app.agents.state import ExecutorState


def test_create_executor_graph_compiles():
    graph = create_executor_graph()
    assert graph is not None


def test_executor_should_continue_complete_on_no_tools():
    state: ExecutorState = {
        "step_goal": "Answer a question",
        "lc_messages": [
            HumanMessage(content="What is 2+2?"),
            AIMessage(content="4"),
        ],
        "tool_iterations": 0,
        "events": [],
    }
    result = executor_should_continue(state)
    assert result == "complete"


def test_executor_should_continue_act_on_tool_calls():
    state: ExecutorState = {
        "step_goal": "Search for something",
        "lc_messages": [
            HumanMessage(content="Search for Python"),
            AIMessage(
                content="",
                tool_calls=[{"name": "web_search", "args": {"query": "Python"}, "id": "tc1"}],
            ),
        ],
        "tool_iterations": 0,
        "tier": "pro",
        "events": [],
    }
    result = executor_should_continue(state)
    assert result == "act"


def test_executor_should_continue_complete_on_error():
    state: ExecutorState = {
        "step_goal": "Do something",
        "has_error": True,
        "lc_messages": [],
        "events": [],
    }
    result = executor_should_continue(state)
    assert result == "complete"


def test_executor_should_continue_complete_on_max_iterations():
    state: ExecutorState = {
        "step_goal": "Do something",
        "lc_messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "web_search", "args": {}, "id": "tc1"}],
            ),
        ],
        "tool_iterations": 100,
        "tier": "pro",
        "events": [],
    }
    result = executor_should_continue(state)
    assert result == "complete"
