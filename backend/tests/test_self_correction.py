"""Tests for the self-correction / verification loop in the task agent.

Covers:
- Consecutive error counting (increment on error, reset on success)
- Error message injection at threshold
- Verification node routing for planned execution
- Verification event emission
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents.events import VerificationEvent
from app.agents.state import TaskState
from app.agents.subagents.task import (
    act_node,
    should_adapt_or_finalize,
    should_continue,
    verify_node,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_plan_steps():
    """Sample plan steps for planned execution tests."""
    return [
        {
            "step_number": 1,
            "action": "Search for information",
            "tool_or_skill": "web_search",
            "depends_on": [],
        },
        {
            "step_number": 2,
            "action": "Analyze results",
            "tool_or_skill": "execute_code",
            "depends_on": [1],
        },
    ]


# ============================================================================
# Tests for consecutive error counting
# ============================================================================


class TestConsecutiveErrorTracking:
    """Tests for error detection and consecutive error counting in act_node."""

    @pytest.mark.asyncio
    async def test_increments_on_error(self):
        """consecutive_errors should increment when tool result contains error."""
        ai_msg = AIMessage(
            content="Running code",
            tool_calls=[{
                "name": "execute_code",
                "args": {"code": "bad()"},
                "id": "call_001",
            }],
        )

        error_result = ToolMessage(
            content="Error: NameError: name 'bad' is not defined",
            tool_call_id="call_001",
            name="execute_code",
        )

        state: TaskState = {
            "query": "run some code",
            "lc_messages": [
                SystemMessage(content="You are helpful."),
                HumanMessage(content="run some code"),
                ai_msg,
            ],
            "tool_iterations": 0,
            "events": [],
            "consecutive_errors": 0,
            "auto_approve_tools": [],
            "hitl_enabled": False,
        }

        with patch(
            "app.agents.subagents.task.execute_tools_batch",
            new_callable=AsyncMock,
            return_value=([error_result], [], 1, None),
        ), patch(
            "app.agents.subagents.task._get_cached_task_tools",
            return_value=[],
        ), patch(
            "app.agents.subagents.task._get_cached_react_config",
        ):
            result = await act_node(state)

        assert result["consecutive_errors"] == 1

    @pytest.mark.asyncio
    async def test_resets_on_success(self):
        """consecutive_errors should reset to 0 when tool succeeds."""
        ai_msg = AIMessage(
            content="Searching",
            tool_calls=[{
                "name": "web_search",
                "args": {"query": "test"},
                "id": "call_002",
            }],
        )

        success_result = ToolMessage(
            content="Found 10 results about testing",
            tool_call_id="call_002",
            name="web_search",
        )

        state: TaskState = {
            "query": "search for test",
            "lc_messages": [
                SystemMessage(content="You are helpful."),
                ai_msg,
            ],
            "tool_iterations": 1,
            "events": [],
            "consecutive_errors": 2,  # Previous errors
            "auto_approve_tools": [],
            "hitl_enabled": False,
        }

        with patch(
            "app.agents.subagents.task.execute_tools_batch",
            new_callable=AsyncMock,
            return_value=([success_result], [], 0, None),
        ), patch(
            "app.agents.subagents.task._get_cached_task_tools",
            return_value=[],
        ), patch(
            "app.agents.subagents.task._get_cached_react_config",
        ):
            result = await act_node(state)

        assert result["consecutive_errors"] == 0

    @pytest.mark.asyncio
    async def test_injects_strong_message_at_threshold(self):
        """Should inject strong system message at 3 consecutive errors."""
        ai_msg = AIMessage(
            content="Trying again",
            tool_calls=[{
                "name": "execute_code",
                "args": {"code": "fail()"},
                "id": "call_003",
            }],
        )

        error_result = ToolMessage(
            content="Error: function fail is not defined",
            tool_call_id="call_003",
            name="execute_code",
        )

        state: TaskState = {
            "query": "run code",
            "lc_messages": [
                SystemMessage(content="You are helpful."),
                ai_msg,
            ],
            "tool_iterations": 2,
            "events": [],
            "consecutive_errors": 2,  # Will become 3 after this error
            "auto_approve_tools": [],
            "hitl_enabled": False,
        }

        with patch(
            "app.agents.subagents.task.execute_tools_batch",
            new_callable=AsyncMock,
            return_value=([error_result], [], 1, None),
        ), patch(
            "app.agents.subagents.task._get_cached_task_tools",
            return_value=[],
        ), patch(
            "app.agents.subagents.task._get_cached_react_config",
        ):
            result = await act_node(state)

        assert result["consecutive_errors"] == 3

        # Check that a strong system message was injected
        system_msgs = [
            m for m in result["lc_messages"]
            if isinstance(m, SystemMessage) and "failed 3 consecutive" in m.content
        ]
        assert len(system_msgs) == 1

    @pytest.mark.asyncio
    async def test_injects_light_message_below_threshold(self):
        """Should inject lighter error message below 3 consecutive errors."""
        ai_msg = AIMessage(
            content="Running",
            tool_calls=[{
                "name": "execute_code",
                "args": {"code": "x"},
                "id": "call_004",
            }],
        )

        error_result = ToolMessage(
            content="Error: unexpected token",
            tool_call_id="call_004",
            name="execute_code",
        )

        state: TaskState = {
            "query": "run code",
            "lc_messages": [
                SystemMessage(content="You are helpful."),
                ai_msg,
            ],
            "tool_iterations": 0,
            "events": [],
            "consecutive_errors": 0,
            "auto_approve_tools": [],
            "hitl_enabled": False,
        }

        with patch(
            "app.agents.subagents.task.execute_tools_batch",
            new_callable=AsyncMock,
            return_value=([error_result], [], 1, None),
        ), patch(
            "app.agents.subagents.task._get_cached_task_tools",
            return_value=[],
        ), patch(
            "app.agents.subagents.task._get_cached_react_config",
        ):
            result = await act_node(state)

        assert result["consecutive_errors"] == 1

        # Check category-aware recovery message was injected (not the strong one)
        light_msgs = [
            m for m in result["lc_messages"]
            if isinstance(m, SystemMessage) and "Tool error (" in m.content
        ]
        assert len(light_msgs) == 1

        strong_msgs = [
            m for m in result["lc_messages"]
            if isinstance(m, SystemMessage) and "failed 3 consecutive" in m.content
        ]
        assert len(strong_msgs) == 0

    @pytest.mark.asyncio
    async def test_no_error_tracking_on_interrupt(self):
        """Should not track errors when there's a pending interrupt."""
        ai_msg = AIMessage(
            content="Asking user",
            tool_calls=[{
                "name": "ask_user",
                "args": {"question": "Which?"},
                "id": "call_005",
            }],
        )

        pending = {
            "interrupt_id": "int_001",
            "thread_id": "default",
            "tool_call_id": "call_005",
            "tool_name": "ask_user",
        }

        state: TaskState = {
            "query": "help",
            "lc_messages": [
                SystemMessage(content="You are helpful."),
                ai_msg,
            ],
            "tool_iterations": 0,
            "events": [],
            "consecutive_errors": 1,
            "auto_approve_tools": [],
            "hitl_enabled": True,
        }

        with patch(
            "app.agents.subagents.task.execute_tools_batch",
            new_callable=AsyncMock,
            return_value=([], [], 0, pending),
        ), patch(
            "app.agents.subagents.task._get_cached_task_tools",
            return_value=[],
        ), patch(
            "app.agents.subagents.task._get_cached_react_config",
        ):
            result = await act_node(state)

        # consecutive_errors should not change when there's a pending interrupt
        assert "consecutive_errors" not in result or result.get("consecutive_errors") is None


# ============================================================================
# Tests for verification node routing
# ============================================================================


class TestVerificationRouting:
    """Tests for should_continue routing to verify node."""

    def test_routes_to_verify_when_plan_complete(self, sample_plan_steps):
        """should_continue should route to 'verify' when all plan steps done and no tool calls."""
        state: TaskState = {
            "lc_messages": [
                AIMessage(content="All done, here is the summary."),
            ],
            "execution_plan": sample_plan_steps,
            "current_step_index": 2,  # Past the end (2 steps, 0-indexed)
        }

        result = should_continue(state)
        assert result == "verify"

    def test_routes_to_finalize_without_plan(self):
        """should_continue should route to 'finalize' when no plan and no tool calls."""
        state: TaskState = {
            "lc_messages": [
                AIMessage(content="Here is your answer."),
            ],
        }

        result = should_continue(state)
        assert result == "finalize"

    def test_routes_to_act_when_plan_active_and_tools_called(self, sample_plan_steps):
        """should_continue should route to 'act' when plan is active but tools are called."""
        state: TaskState = {
            "lc_messages": [
                AIMessage(
                    content="Searching",
                    tool_calls=[{"name": "web_search", "args": {}, "id": "1"}],
                ),
            ],
            "execution_plan": sample_plan_steps,
            "current_step_index": 0,
            "tool_iterations": 0,
        }

        with patch("app.agents.subagents.task._get_cached_react_config") as mock_config:
            mock_config.return_value = MagicMock(max_iterations=10)
            result = should_continue(state)

        assert result == "act"

    def test_routes_to_verify_at_max_iterations_with_plan(self, sample_plan_steps):
        """should_continue routes to 'verify' at max iterations if plan was active."""
        state: TaskState = {
            "lc_messages": [
                AIMessage(
                    content="Still going",
                    tool_calls=[{"name": "web_search", "args": {}, "id": "1"}],
                ),
            ],
            "execution_plan": sample_plan_steps,
            "current_step_index": 1,
            "tool_iterations": 10,
        }

        with patch("app.agents.subagents.task._get_cached_react_config") as mock_config:
            mock_config.return_value = MagicMock(max_iterations=10)
            result = should_continue(state)

        assert result == "verify"

    def test_routes_to_finalize_at_max_iterations_without_plan(self):
        """should_continue routes to 'finalize' at max iterations without plan."""
        state: TaskState = {
            "lc_messages": [
                AIMessage(
                    content="Still going",
                    tool_calls=[{"name": "web_search", "args": {}, "id": "1"}],
                ),
            ],
            "tool_iterations": 10,
        }

        with patch("app.agents.subagents.task._get_cached_react_config") as mock_config:
            mock_config.return_value = MagicMock(max_iterations=10)
            result = should_continue(state)

        assert result == "finalize"


# ============================================================================
# Tests for verify_node
# ============================================================================


class TestVerifyNode:
    """Tests for the verify_node function."""

    @pytest.mark.asyncio
    async def test_skips_without_plan(self):
        """verify_node should return empty events when no execution_plan."""
        state: TaskState = {
            "query": "simple question",
            "events": [],
        }

        result = await verify_node(state)
        assert result["events"] == []

    @pytest.mark.asyncio
    async def test_emits_passed_event(self, sample_plan_steps):
        """verify_node should emit verification passed event."""
        state: TaskState = {
            "query": "Search and analyze data",
            "execution_plan": sample_plan_steps,
            "completed_step_results": [
                {"step_number": 1, "action": "Search", "result_summary": "Found results"},
                {"step_number": 2, "action": "Analyze", "result_summary": "Analysis complete"},
            ],
            "events": [],
            "provider": "anthropic",
        }

        mock_response = MagicMock()
        mock_response.content = "PASS: All steps were completed successfully."
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch(
            "app.agents.subagents.task.llm_service"
        ) as mock_service:
            mock_service.get_llm_for_tier = MagicMock(return_value=mock_llm)
            result = await verify_node(state)

        verification_events = [e for e in result["events"] if e.get("type") == "verification"]
        assert len(verification_events) == 1
        assert verification_events[0]["status"] == "passed"
        assert "findings" in verification_events[0]

    @pytest.mark.asyncio
    async def test_emits_failed_event(self, sample_plan_steps):
        """verify_node should emit verification failed event."""
        state: TaskState = {
            "query": "Build a complete app",
            "execution_plan": sample_plan_steps,
            "completed_step_results": [
                {"step_number": 1, "action": "Search", "result_summary": "Found results"},
            ],
            "events": [],
            "provider": "anthropic",
        }

        mock_response = MagicMock()
        mock_response.content = "FAIL: Step 2 was not completed. Analysis is missing."
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch(
            "app.agents.subagents.task.llm_service"
        ) as mock_service:
            mock_service.get_llm_for_tier = MagicMock(return_value=mock_llm)
            result = await verify_node(state)

        verification_events = [e for e in result["events"] if e.get("type") == "verification"]
        assert len(verification_events) == 1
        assert verification_events[0]["status"] == "failed"
        assert "missing" in verification_events[0]["message"].lower()
        assert result["verification_status"] == "failed"
        assert result["verification_attempts"] == 1
        assert result["execution_plan"] == []

    @pytest.mark.asyncio
    async def test_handles_llm_error(self, sample_plan_steps):
        """verify_node should handle LLM errors gracefully."""
        state: TaskState = {
            "query": "do something",
            "execution_plan": sample_plan_steps,
            "completed_step_results": [],
            "events": [],
            "provider": "anthropic",
        }

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM unavailable"))

        with patch(
            "app.agents.subagents.task.llm_service"
        ) as mock_service:
            mock_service.get_llm_for_tier = MagicMock(return_value=mock_llm)
            result = await verify_node(state)

        verification_events = [e for e in result["events"] if e.get("type") == "verification"]
        assert len(verification_events) == 1
        assert verification_events[0]["status"] == "error"
        assert result["verification_status"] == "error"


class TestVerificationAdaptationRouting:
    """Tests for post-verification routing behavior."""

    def test_finalize_when_verification_passed(self):
        state: TaskState = {"verification_status": "passed", "verification_attempts": 0}
        assert should_adapt_or_finalize(state) == "finalize"

    def test_route_to_reason_on_first_failure(self):
        state: TaskState = {"verification_status": "failed", "verification_attempts": 1}
        assert should_adapt_or_finalize(state) == "reason"

    def test_finalize_after_attempt_limit_auto_mode(self):
        state: TaskState = {
            "verification_status": "failed",
            "verification_attempts": 2,
            "execution_mode": "auto",
        }
        assert should_adapt_or_finalize(state) == "finalize"

    def test_strict_mode_allows_only_single_attempt(self):
        state: TaskState = {
            "verification_status": "partial",
            "verification_attempts": 1,
            "execution_mode": "strict",
        }
        assert should_adapt_or_finalize(state) == "finalize"

    @pytest.mark.asyncio
    async def test_uses_flash_tier(self, sample_plan_steps):
        """verify_node should use FLASH tier LLM for cost efficiency."""
        from app.models.schemas import ModelTier

        state: TaskState = {
            "query": "test task",
            "execution_plan": sample_plan_steps,
            "completed_step_results": [],
            "events": [],
            "provider": "anthropic",
        }

        mock_response = MagicMock()
        mock_response.content = "PASS: looks good"
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch(
            "app.agents.subagents.task.llm_service"
        ) as mock_service:
            mock_service.get_llm_for_tier = MagicMock(return_value=mock_llm)
            await verify_node(state)

        # Verify FLASH tier was requested
        mock_service.get_llm_for_tier.assert_called_once_with(
            ModelTier.FLASH, provider="anthropic"
        )


# ============================================================================
# Tests for verification event model
# ============================================================================


class TestVerificationEvent:
    """Tests for the VerificationEvent model and factory."""

    def test_verification_event_model(self):
        """VerificationEvent should have correct fields."""
        event = VerificationEvent(status="passed", message="All good")
        assert event.type == "verification"
        assert event.status == "passed"
        assert event.message == "All good"
        assert event.step is None

    def test_verification_event_with_step(self):
        """VerificationEvent should accept step number."""
        event = VerificationEvent(status="failed", message="Step 2 failed", step=2)
        assert event.step == 2

    def test_verification_factory(self):
        """verification() factory should produce correct dict."""
        from app.agents.events import verification

        event = verification(status="passed", message="All good", step=1)
        assert event["type"] == "verification"
        assert event["status"] == "passed"
        assert event["message"] == "All good"
        assert event["step"] == 1
        assert "timestamp" in event


# ============================================================================
# Tests for TaskState new fields
# ============================================================================


class TestTaskStateNewFields:
    """Tests for consecutive_errors and completed_step_results in TaskState."""

    def test_state_accepts_consecutive_errors(self):
        """TaskState should accept consecutive_errors field."""
        state: TaskState = {
            "query": "test",
            "consecutive_errors": 2,
        }
        assert state["consecutive_errors"] == 2

    def test_state_accepts_completed_step_results(self):
        """TaskState should accept completed_step_results field."""
        state: TaskState = {
            "query": "test",
            "completed_step_results": [
                {"step_number": 1, "action": "test", "result_summary": "ok"},
            ],
        }
        assert len(state["completed_step_results"]) == 1

    def test_state_defaults_without_new_fields(self):
        """TaskState should work without new fields."""
        state: TaskState = {"query": "test"}
        assert state.get("consecutive_errors") is None
        assert state.get("completed_step_results") is None
