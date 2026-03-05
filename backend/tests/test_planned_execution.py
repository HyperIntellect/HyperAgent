"""Tests for the planned execution mode in the task agent.

Phase 2 of the hybrid planning approach: when the task_planning skill returns
a plan, the task agent parses it into state and tracks step-by-step progress.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents.events import PlanStepEvent
from app.agents.skills.builtin.task_planning_skill import PlanStep, TaskPlan
from app.agents.state import TaskState
from app.agents.subagents.task import (
    _extract_task_plan_from_messages,
    act_node,
    reason_node,
    should_continue,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_plan_steps():
    """Sample plan steps as they would appear in skill output."""
    return [
        {
            "step_number": 1,
            "action": "Search for Python web scraping libraries",
            "tool_or_skill": "web_search",
            "depends_on": [],
            "estimated_complexity": "low",
        },
        {
            "step_number": 2,
            "action": "Write scraping code using BeautifulSoup",
            "tool_or_skill": "execute_code",
            "depends_on": [1],
            "estimated_complexity": "medium",
        },
        {
            "step_number": 3,
            "action": "Test the scraper and handle errors",
            "tool_or_skill": "execute_code",
            "depends_on": [2],
            "estimated_complexity": "medium",
        },
    ]


@pytest.fixture
def task_planning_tool_message(sample_plan_steps):
    """A ToolMessage containing successful task_planning skill output."""
    return ToolMessage(
        content=json.dumps({
            "skill_id": "task_planning",
            "output": {
                "task_summary": "Build a web scraper",
                "complexity_assessment": "moderate",
                "steps": sample_plan_steps,
                "success_criteria": ["Scraper works"],
                "potential_challenges": [],
                "clarifying_questions": [],
            },
            "success": True,
            "events": [],
        }),
        tool_call_id="call_planning_123",
        name="invoke_skill",
    )


@pytest.fixture
def non_planning_tool_message():
    """A ToolMessage from a non-planning skill."""
    return ToolMessage(
        content=json.dumps({
            "skill_id": "web_research",
            "output": {"summary": "Some research results"},
            "success": True,
            "events": [],
        }),
        tool_call_id="call_research_456",
        name="invoke_skill",
    )


@pytest.fixture
def failed_planning_tool_message():
    """A ToolMessage from a failed task_planning skill."""
    return ToolMessage(
        content=json.dumps({
            "skill_id": "task_planning",
            "error": "LLM service unavailable",
            "events": [],
        }),
        tool_call_id="call_planning_fail",
        name="invoke_skill",
    )


@pytest.fixture
def regular_tool_message():
    """A ToolMessage from a regular (non-skill) tool."""
    return ToolMessage(
        content="Search results for Python scraping",
        tool_call_id="call_search_789",
        name="web_search",
    )


# ============================================================================
# Tests for _extract_task_plan_from_messages
# ============================================================================


class TestExtractTaskPlan:
    """Tests for plan extraction from tool messages."""

    def test_extracts_plan_from_successful_skill_output(
        self, task_planning_tool_message, sample_plan_steps
    ):
        """Should extract plan steps from task_planning skill output."""
        result = _extract_task_plan_from_messages([task_planning_tool_message])
        assert result is not None
        assert len(result) == 3
        assert result[0]["step_number"] == 1
        assert result[0]["action"] == "Search for Python web scraping libraries"
        assert result[0]["tool_or_skill"] == "web_search"

    def test_returns_none_for_non_planning_skill(self, non_planning_tool_message):
        """Should return None when no task_planning output exists."""
        result = _extract_task_plan_from_messages([non_planning_tool_message])
        assert result is None

    def test_returns_none_for_failed_planning(self, failed_planning_tool_message):
        """Should return None when task_planning failed."""
        result = _extract_task_plan_from_messages([failed_planning_tool_message])
        assert result is None

    def test_returns_none_for_regular_tool(self, regular_tool_message):
        """Should return None for non-invoke_skill tools."""
        result = _extract_task_plan_from_messages([regular_tool_message])
        assert result is None

    def test_returns_none_for_empty_list(self):
        """Should return None for empty message list."""
        result = _extract_task_plan_from_messages([])
        assert result is None

    def test_returns_none_for_empty_steps(self):
        """Should return None when plan has no steps."""
        msg = ToolMessage(
            content=json.dumps({
                "skill_id": "task_planning",
                "output": {"steps": [], "task_summary": "Empty"},
                "success": True,
            }),
            tool_call_id="call_empty",
            name="invoke_skill",
        )
        result = _extract_task_plan_from_messages([msg])
        assert result is None

    def test_handles_malformed_json(self):
        """Should handle malformed JSON gracefully."""
        msg = ToolMessage(
            content="not valid json",
            tool_call_id="call_bad",
            name="invoke_skill",
        )
        result = _extract_task_plan_from_messages([msg])
        assert result is None

    def test_finds_plan_among_multiple_messages(
        self, regular_tool_message, task_planning_tool_message, sample_plan_steps
    ):
        """Should find the plan even with other messages present."""
        result = _extract_task_plan_from_messages([
            regular_tool_message,
            task_planning_tool_message,
        ])
        assert result is not None
        assert len(result) == 3


# ============================================================================
# Tests for plan_step event factory
# ============================================================================


class TestPlanStepEvent:
    """Tests for the plan_step event factory."""

    def test_plan_step_event_running(self):
        """Should create a running plan_step event."""
        from app.agents.events import plan_step

        event = plan_step(step_number=1, total_steps=3, action="Do something")
        assert event["type"] == "plan_step"
        assert event["step_number"] == 1
        assert event["total_steps"] == 3
        assert event["action"] == "Do something"
        assert event["status"] == "running"
        assert "timestamp" in event

    def test_plan_step_event_completed(self):
        """Should create a completed plan_step event."""
        from app.agents.events import plan_step

        event = plan_step(
            step_number=2, total_steps=3, action="Done", status="completed"
        )
        assert event["status"] == "completed"

    def test_plan_step_event_model(self):
        """PlanStepEvent model should have correct fields."""
        event = PlanStepEvent(
            step_number=1, total_steps=5, action="Test step"
        )
        assert event.type == "plan_step"
        assert event.step_number == 1
        assert event.total_steps == 5
        assert event.status == "running"


# ============================================================================
# Tests for TaskState planned execution fields
# ============================================================================


class TestTaskStatePlannedFields:
    """Tests for execution_plan and current_step_index in TaskState."""

    def test_state_accepts_execution_plan(self):
        """TaskState should accept execution_plan field."""
        state: TaskState = {
            "query": "test",
            "execution_plan": [{"step_number": 1, "action": "test"}],
            "current_step_index": 0,
        }
        assert state["execution_plan"][0]["step_number"] == 1
        assert state["current_step_index"] == 0

    def test_state_defaults_without_plan(self):
        """TaskState should work without execution_plan."""
        state: TaskState = {"query": "test"}
        assert state.get("execution_plan") is None
        assert state.get("current_step_index") is None


# ============================================================================
# Tests for act_node plan parsing integration
# ============================================================================


class TestActNodePlanParsing:
    """Tests for act_node's plan parsing behavior."""

    @pytest.mark.asyncio
    async def test_act_node_parses_plan_into_state(self, sample_plan_steps):
        """act_node should parse task_planning output into execution_plan state."""
        # Create an AI message with an invoke_skill tool call
        ai_msg = AIMessage(
            content="Let me plan this task.",
            tool_calls=[{
                "name": "invoke_skill",
                "args": {
                    "skill_id": "task_planning",
                    "params": {"task_description": "Build a scraper"},
                },
                "id": "call_plan_001",
            }],
        )

        # Create the tool result
        plan_result = ToolMessage(
            content=json.dumps({
                "skill_id": "task_planning",
                "output": {
                    "task_summary": "Build a scraper",
                    "complexity_assessment": "moderate",
                    "steps": sample_plan_steps,
                    "success_criteria": ["Works"],
                    "potential_challenges": [],
                    "clarifying_questions": [],
                },
                "success": True,
                "events": [],
            }),
            tool_call_id="call_plan_001",
            name="invoke_skill",
        )

        state: TaskState = {
            "query": "Build a web scraper",
            "lc_messages": [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Build a web scraper"),
                ai_msg,
            ],
            "tool_iterations": 0,
            "events": [],
            "auto_approve_tools": [],
            "hitl_enabled": False,
        }

        # Mock execute_tools_batch to return our plan result
        with patch(
            "app.agents.subagents.task.execute_tools_batch",
            new_callable=AsyncMock,
            return_value=([plan_result], [], 0, None),
        ), patch(
            "app.agents.subagents.task._get_cached_task_tools",
            return_value=[],
        ), patch(
            "app.agents.subagents.task._get_react_config_for_state",
        ):
            result = await act_node(state)

        # Verify plan was parsed into state
        assert "execution_plan" in result
        assert len(result["execution_plan"]) == 3
        assert result["current_step_index"] == 0
        assert result["execution_plan"][0]["action"] == "Search for Python web scraping libraries"

        # Verify plan_step event was emitted for first step
        plan_events = [e for e in result["events"] if e.get("type") == "plan_step"]
        assert len(plan_events) == 1
        assert plan_events[0]["step_number"] == 1
        assert plan_events[0]["status"] == "running"

    @pytest.mark.asyncio
    async def test_act_node_advances_step_via_complete_step(self, sample_plan_steps):
        """act_node should advance current_step_index when complete_step is called."""
        ai_msg = AIMessage(
            content="Done searching.",
            tool_calls=[{
                "name": "complete_step",
                "args": {"step_number": 1, "result_summary": "Found BeautifulSoup"},
                "id": "call_complete_001",
            }],
        )

        complete_result = ToolMessage(
            content="Step 1 marked as completed. Summary: Found BeautifulSoup. The system will advance to the next step.",
            tool_call_id="call_complete_001",
            name="complete_step",
        )

        state: TaskState = {
            "query": "Build a web scraper",
            "lc_messages": [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Build a web scraper"),
                ai_msg,
            ],
            "execution_plan": sample_plan_steps,
            "current_step_index": 0,
            "tool_iterations": 1,
            "events": [],
            "auto_approve_tools": [],
            "hitl_enabled": False,
        }

        with patch(
            "app.agents.subagents.task.execute_tools_batch",
            new_callable=AsyncMock,
            return_value=([complete_result], [], 0, None),
        ), patch(
            "app.agents.subagents.task._get_cached_task_tools",
            return_value=[],
        ), patch(
            "app.agents.subagents.task._get_react_config_for_state",
        ):
            result = await act_node(state)

        # Step should advance from 0 to 1
        assert result["current_step_index"] == 1

        # Should have completed event for step 1 and running event for step 2
        plan_events = [e for e in result["events"] if e.get("type") == "plan_step"]
        assert len(plan_events) == 2
        assert plan_events[0]["status"] == "completed"
        assert plan_events[0]["step_number"] == 1
        assert plan_events[1]["status"] == "running"
        assert plan_events[1]["step_number"] == 2

    @pytest.mark.asyncio
    async def test_act_node_completes_last_step_via_complete_step(self, sample_plan_steps):
        """act_node should handle completing the final step of a plan via complete_step."""
        ai_msg = AIMessage(
            content="Done testing.",
            tool_calls=[{
                "name": "complete_step",
                "args": {"step_number": 3, "result_summary": "Tests pass"},
                "id": "call_complete_003",
            }],
        )

        complete_result = ToolMessage(
            content="Step 3 marked as completed.",
            tool_call_id="call_complete_003",
            name="complete_step",
        )

        state: TaskState = {
            "query": "Build a web scraper",
            "lc_messages": [
                SystemMessage(content="You are a helpful assistant."),
                ai_msg,
            ],
            "execution_plan": sample_plan_steps,
            "current_step_index": 2,  # Last step (0-indexed)
            "tool_iterations": 3,
            "events": [],
            "auto_approve_tools": [],
            "hitl_enabled": False,
        }

        with patch(
            "app.agents.subagents.task.execute_tools_batch",
            new_callable=AsyncMock,
            return_value=([complete_result], [], 0, None),
        ), patch(
            "app.agents.subagents.task._get_cached_task_tools",
            return_value=[],
        ), patch(
            "app.agents.subagents.task._get_react_config_for_state",
        ):
            result = await act_node(state)

        # Step should advance past the end
        assert result["current_step_index"] == 3

        # Should have completed event for last step, no running event
        plan_events = [e for e in result["events"] if e.get("type") == "plan_step"]
        assert len(plan_events) == 1
        assert plan_events[0]["status"] == "completed"
        assert plan_events[0]["step_number"] == 3

    @pytest.mark.asyncio
    async def test_act_node_no_advance_on_interrupt(self, sample_plan_steps):
        """act_node should NOT advance step when there's a pending interrupt."""
        ai_msg = AIMessage(
            content="Asking...",
            tool_calls=[{
                "name": "ask_user",
                "args": {"question": "Which library?"},
                "id": "call_ask_001",
            }],
        )

        state: TaskState = {
            "query": "Build a web scraper",
            "lc_messages": [
                SystemMessage(content="You are a helpful assistant."),
                ai_msg,
            ],
            "execution_plan": sample_plan_steps,
            "current_step_index": 0,
            "tool_iterations": 1,
            "events": [],
            "auto_approve_tools": [],
            "hitl_enabled": True,
        }

        pending = {
            "interrupt_id": "int_001",
            "thread_id": "default",
            "tool_call_id": "call_ask_001",
            "tool_name": "ask_user",
        }

        with patch(
            "app.agents.subagents.task.execute_tools_batch",
            new_callable=AsyncMock,
            return_value=([], [], 0, pending),
        ), patch(
            "app.agents.subagents.task._get_cached_task_tools",
            return_value=[],
        ), patch(
            "app.agents.subagents.task._get_react_config_for_state",
        ):
            result = await act_node(state)

        # Step should NOT advance since there's a pending interrupt
        assert result.get("current_step_index") is None or result.get("current_step_index", 0) == 0


# ============================================================================
# Tests for reason_node step injection
# ============================================================================


class TestReasonNodeStepInjection:
    """Tests for step context injection in reason_node."""

    @pytest.mark.asyncio
    async def test_injects_step_guidance_when_plan_active(self, sample_plan_steps):
        """reason_node should inject current step guidance for LLM call."""
        state: TaskState = {
            "query": "Build a web scraper",
            "lc_messages": [
                SystemMessage(content="You are helpful."),
                HumanMessage(content="Build a web scraper"),
            ],
            "execution_plan": sample_plan_steps,
            "current_step_index": 0,
            "events": [],
            "provider": "anthropic",
        }

        # Mock LLM to capture what messages it receives
        mock_ai_message = AIMessage(content="I'll search for scraping libraries.")
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm.ainvoke = AsyncMock(return_value=mock_ai_message)

        with patch(
            "app.agents.subagents.task.llm_service"
        ) as mock_service, patch(
            "app.agents.subagents.task._get_cached_task_tools",
            return_value=[],
        ), patch(
            "app.agents.subagents.task._get_react_config_for_state",
        ) as mock_config, patch(
            "app.agents.subagents.task.output_scanner"
        ) as mock_scanner:
            mock_service.choose_llm_for_task = MagicMock(return_value=mock_llm)
            mock_config.return_value = MagicMock(
                max_message_tokens=100000,
                preserve_recent_messages=10,
            )
            mock_scanner.scan = AsyncMock(return_value=MagicMock(
                blocked=False, sanitized_content=None
            ))

            await reason_node(state)

        # Check that LLM received step guidance in its messages
        call_args = mock_llm.ainvoke.call_args[0][0]
        # The last message before invocation should be the step guidance
        guidance_messages = [
            m for m in call_args
            if isinstance(m, SystemMessage) and "Plan Execution" in m.content
        ]
        assert len(guidance_messages) == 1
        assert "Step 1 of 3" in guidance_messages[0].content
        assert "Search for Python web scraping libraries" in guidance_messages[0].content

    @pytest.mark.asyncio
    async def test_injects_completion_hint_when_all_steps_done(self, sample_plan_steps):
        """reason_node should inject completion hint when all steps finished."""
        state: TaskState = {
            "query": "Build a web scraper",
            "lc_messages": [
                SystemMessage(content="You are helpful."),
                HumanMessage(content="Build a web scraper"),
            ],
            "execution_plan": sample_plan_steps,
            "current_step_index": 3,  # Past the end
            "events": [],
            "provider": "anthropic",
        }

        mock_ai_message = AIMessage(content="The scraper is complete.")
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm.ainvoke = AsyncMock(return_value=mock_ai_message)

        with patch(
            "app.agents.subagents.task.llm_service"
        ) as mock_service, patch(
            "app.agents.subagents.task._get_cached_task_tools",
            return_value=[],
        ), patch(
            "app.agents.subagents.task._get_react_config_for_state",
        ) as mock_config, patch(
            "app.agents.subagents.task.output_scanner"
        ) as mock_scanner:
            mock_service.choose_llm_for_task = MagicMock(return_value=mock_llm)
            mock_config.return_value = MagicMock(
                max_message_tokens=100000,
                preserve_recent_messages=10,
            )
            mock_scanner.scan = AsyncMock(return_value=MagicMock(
                blocked=False, sanitized_content=None
            ))

            await reason_node(state)

        call_args = mock_llm.ainvoke.call_args[0][0]
        completion_messages = [
            m for m in call_args
            if isinstance(m, SystemMessage) and "All steps completed" in m.content
        ]
        assert len(completion_messages) == 1

    @pytest.mark.asyncio
    async def test_no_injection_without_plan(self):
        """reason_node should NOT inject step guidance when no plan exists."""
        state: TaskState = {
            "query": "Simple question",
            "lc_messages": [
                SystemMessage(content="You are helpful."),
                HumanMessage(content="Simple question"),
            ],
            "events": [],
            "provider": "anthropic",
        }

        mock_ai_message = AIMessage(content="Simple answer.")
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm.ainvoke = AsyncMock(return_value=mock_ai_message)

        with patch(
            "app.agents.subagents.task.llm_service"
        ) as mock_service, patch(
            "app.agents.subagents.task._get_cached_task_tools",
            return_value=[],
        ), patch(
            "app.agents.subagents.task._get_react_config_for_state",
        ) as mock_config, patch(
            "app.agents.subagents.task.output_scanner"
        ) as mock_scanner:
            mock_service.choose_llm_for_task = MagicMock(return_value=mock_llm)
            mock_config.return_value = MagicMock(
                max_message_tokens=100000,
                preserve_recent_messages=10,
            )
            mock_scanner.scan = AsyncMock(return_value=MagicMock(
                blocked=False, sanitized_content=None
            ))

            await reason_node(state)

        call_args = mock_llm.ainvoke.call_args[0][0]
        guidance_messages = [
            m for m in call_args
            if isinstance(m, SystemMessage) and "Plan Execution" in m.content
        ]
        assert len(guidance_messages) == 0


# ============================================================================
# Tests for should_continue with plan tracking
# ============================================================================


class TestShouldContinueWithPlan:
    """Tests for should_continue plan progress logging."""

    def test_continues_act_with_plan(self, sample_plan_steps):
        """should_continue should return 'act' when tools are called during plan."""
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

        with patch("app.agents.subagents.task._get_react_config_for_state") as mock_config:
            mock_config.return_value = MagicMock(max_iterations=10)
            result = should_continue(state)

        assert result == "act"

    def test_routes_to_verify_when_plan_complete_no_tool_calls(self, sample_plan_steps):
        """should_continue should route to 'verify' when plan complete and no tool calls."""
        state: TaskState = {
            "lc_messages": [
                AIMessage(content="All done! Here is the summary."),
            ],
            "execution_plan": sample_plan_steps,
            "current_step_index": 3,
        }

        result = should_continue(state)
        assert result == "verify"

    def test_finalizes_on_error_with_plan(self, sample_plan_steps):
        """should_continue should finalize on error even with active plan."""
        state: TaskState = {
            "has_error": True,
            "lc_messages": [],
            "execution_plan": sample_plan_steps,
            "current_step_index": 1,
        }

        result = should_continue(state)
        assert result == "finalize"


# ============================================================================
# Tests for step guidance message not persisting in state
# ============================================================================


class TestStepGuidanceNotPersisted:
    """Verify that step guidance messages are temporary and don't pollute state."""

    @pytest.mark.asyncio
    async def test_guidance_not_in_returned_lc_messages(self, sample_plan_steps):
        """Step guidance should NOT appear in the returned lc_messages."""
        state: TaskState = {
            "query": "Build a web scraper",
            "lc_messages": [
                SystemMessage(content="You are helpful."),
                HumanMessage(content="Build a web scraper"),
            ],
            "execution_plan": sample_plan_steps,
            "current_step_index": 1,
            "events": [],
            "provider": "anthropic",
        }

        mock_ai_message = AIMessage(content="Writing code now.")
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm.ainvoke = AsyncMock(return_value=mock_ai_message)

        with patch(
            "app.agents.subagents.task.llm_service"
        ) as mock_service, patch(
            "app.agents.subagents.task._get_cached_task_tools",
            return_value=[],
        ), patch(
            "app.agents.subagents.task._get_react_config_for_state",
        ) as mock_config, patch(
            "app.agents.subagents.task.output_scanner"
        ) as mock_scanner:
            mock_service.choose_llm_for_task = MagicMock(return_value=mock_llm)
            mock_config.return_value = MagicMock(
                max_message_tokens=100000,
                preserve_recent_messages=10,
            )
            mock_scanner.scan = AsyncMock(return_value=MagicMock(
                blocked=False, sanitized_content=None
            ))

            result = await reason_node(state)

        # The returned lc_messages should contain the AI response but NOT the guidance
        returned_messages = result["lc_messages"]
        guidance_in_state = [
            m for m in returned_messages
            if isinstance(m, SystemMessage) and "Plan Execution" in m.content
        ]
        assert len(guidance_in_state) == 0

        # But should contain the AI response
        ai_messages = [m for m in returned_messages if isinstance(m, AIMessage)]
        assert len(ai_messages) == 1
        assert ai_messages[0].content == "Writing code now."
