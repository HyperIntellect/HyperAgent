"""Orchestrator: Main graph that combines Plan-and-Execute with ReAct.

Classifies queries, invokes PlannerAgent for complex tasks, dispatches
ExecutorAgent per step, verifies results, and produces final response.
"""

import asyncio
import time as _time
import uuid
from typing import Any, AsyncGenerator, Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from app.agents import events
from app.agents.classifier import classify_query
from app.agents.state import (
    ExecutorState,
    OrchestratorState,
    PlanStep,
    PlannerState,
    StepResult,
)
from app.ai.llm import extract_text_from_content, llm_service
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_STEP_TIMEOUT = 300


async def classify_node(state: OrchestratorState) -> dict:
    """Classify query complexity and set up execution path."""
    query = state.get("query", "").strip()
    mode = state.get("mode")
    skills = state.get("skills") or []

    complexity = classify_query(query, mode=mode, skills=skills)

    event_list = [
        events.reasoning(
            thinking=f"Query classified as {complexity}",
            confidence=0.9,
            context="classification",
        ),
    ]

    result: dict = {
        "query_complexity": complexity,
        "events": event_list,
    }

    if complexity == "simple":
        result["execution_plan"] = []
        result["current_step_index"] = 0

    return result


async def plan_node(state: OrchestratorState) -> dict:
    """Invoke PlannerAgent to decompose the query into steps."""
    from app.agents.planner import planner_subgraph

    planner_input: PlannerState = {
        "query": state.get("query", ""),
        "messages": state.get("messages") or [],
        "mode": state.get("mode"),
        "provider": state.get("provider"),
        "tier": state.get("tier"),
        "locale": state.get("locale", "en"),
        "revision_context": None,
        "events": [],
    }

    # Check for re-planning after verification failure
    verification_status = state.get("verification_status")
    if verification_status in ("failed", "partial"):
        step_results = state.get("step_results") or []
        revision_lines = [
            f"Step {sr['step_number']}: {sr['status']} - {sr['output'][:200]}"
            for sr in step_results
        ]
        planner_input["revision_context"] = "\n".join(revision_lines)

    result = await planner_subgraph.ainvoke(planner_input)
    plan_steps = result.get("plan_steps", [])
    event_list = list(result.get("events", []))

    event_list.append(events.plan_overview(
        steps=[
            {
                "id": i,
                "title": step["goal"],
                "description": ", ".join(step.get("tools_hint", [])),
                "status": "pending",
            }
            for i, step in enumerate(plan_steps)
        ],
        total_steps=len(plan_steps),
        completed_steps=0,
    ))

    return {
        "execution_plan": plan_steps,
        "current_step_index": 0,
        "events": event_list,
    }


async def dispatch_step_node(
    state: OrchestratorState, config: RunnableConfig | None = None,
) -> dict:
    """Invoke ExecutorAgent for the current step."""
    from app.agents.executor import executor_subgraph

    query = state.get("query", "")
    plan = state.get("execution_plan") or []
    step_index = state.get("current_step_index", 0)
    complexity = state.get("query_complexity", "simple")
    event_list: list[dict] = []

    if complexity == "simple" or not plan:
        step_goal = query
        step_number = None
        tools_hint: list[str] = []
    else:
        step = plan[step_index]
        step_goal = step["goal"]
        step_number = step["step_number"]
        tools_hint = step.get("tools_hint", [])
        event_list.append(events.step_activity(
            step_index=step_index,
            step_title=step_goal,
            status="running",
        ))
        event_list.append(events.plan_step(
            step_number=step_number,
            total_steps=len(plan),
            action=step_goal,
            status="running",
        ))

    executor_input: ExecutorState = {
        "step_goal": step_goal,
        "step_number": step_number,
        "tools_hint": tools_hint,
        "system_prompt": state.get("system_prompt", ""),
        "messages": state.get("messages") or [],
        "user_id": state.get("user_id"),
        "task_id": state.get("task_id"),
        "run_id": state.get("run_id"),
        "provider": state.get("provider"),
        "model": state.get("model"),
        "tier": state.get("tier"),
        "locale": state.get("locale", "en"),
        "skills": state.get("skills") or [],
        "mode": state.get("mode"),
        "attachment_ids": state.get("attachment_ids") or [],
        "image_attachments": state.get("image_attachments") or [],
        "attachment_context": state.get("attachment_context"),
        "hitl_enabled": state.get("hitl_enabled", True),
        "depth": state.get("depth"),
        "lc_messages": [],
        "tool_iterations": 0,
        "consecutive_errors": 0,
        "last_tool_calls_hash": [],
        "events": [],
    }

    timeout = DEFAULT_STEP_TIMEOUT
    if state.get("mode") in ("app", "research"):
        timeout = max(timeout, 600)

    try:
        async with asyncio.timeout(timeout):
            result = await executor_subgraph.ainvoke(executor_input, config=config)
    except asyncio.TimeoutError:
        logger.error("dispatch_step_timeout", step_index=step_index)
        result = {"result": "Step timed out", "status": "failed", "events": []}

    step_result = StepResult(
        step_number=step_number or 1,
        status=result.get("status", "completed"),
        output=result.get("result", ""),
        events=result.get("events", []),
    )

    if plan and step_number:
        event_list.append(events.plan_step(
            step_number=step_number,
            total_steps=len(plan),
            action=step_goal,
            status=step_result["status"],
        ))
        event_list.append(events.plan_step_completed(
            step_id=step_index,
            status=step_result["status"],
            completed_steps=step_index + 1,
            total_steps=len(plan),
            result_summary=step_result["output"][:200],
        ))

    event_list.extend(result.get("events", []))

    output: dict = {
        "step_results": [step_result],
        "current_step_index": step_index + 1,
        "events": event_list,
    }

    if complexity == "simple":
        output["response"] = result.get("result", "")

    handoff = result.get("pending_handoff")
    if handoff:
        output["pending_handoff"] = handoff

    return output


def step_check(state: OrchestratorState) -> Literal["dispatch_step", "verify", "finalize"]:
    """Check if more steps remain, need verification, or done."""
    complexity = state.get("query_complexity", "simple")
    if complexity == "simple":
        return "finalize"

    plan = state.get("execution_plan") or []
    step_index = state.get("current_step_index", 0)
    if step_index < len(plan):
        return "dispatch_step"

    return "verify"


async def verify_node(state: OrchestratorState) -> dict:
    """Verify plan execution results against the original goal."""
    from app.models.schemas import ModelTier  # deferred: avoids circular import with schemas

    step_results = state.get("step_results") or []
    query = state.get("query", "")
    provider = state.get("provider")
    event_list: list[dict] = []

    if not step_results:
        return {"verification_status": "passed", "events": []}

    steps_summary = "\n".join(
        f"- Step {r['step_number']}: {r['status']} - {r['output'][:200]}"
        for r in step_results
    )

    verification_prompt = (
        f"Review the completed task execution:\n\n"
        f"Original goal:\n<user_query>{query[:500]}</user_query>\n\n"
        f"Steps completed:\n{steps_summary}\n\n"
        f"Respond with:\n"
        f"- PASS: if the goal was achieved\n"
        f"- PARTIAL: if some goals were met\n"
        f"- FAIL: with explanation"
    )

    try:
        llm = llm_service.get_llm_for_tier(ModelTier.LITE, provider=provider)
        result = await llm.ainvoke([HumanMessage(content=verification_prompt)])
        verification_text = extract_text_from_content(result.content)
        verdict = verification_text.strip().split(":")[0].upper().strip()

        if verdict.startswith("PASS"):
            status = "passed"
        elif verdict.startswith("PARTIAL"):
            status = "partial"
        else:
            status = "failed"

        event_list.append(events.verification(status=status, message=verification_text))

        prev_attempts = state.get("verification_attempts", 0) or 0
        return {
            "verification_status": status,
            "verification_attempts": prev_attempts + (0 if status == "passed" else 1),
            "events": event_list,
        }
    except Exception as e:
        logger.error("verify_node_error", error=str(e))
        prev_attempts = state.get("verification_attempts", 0) or 0
        return {
            "verification_status": "error",
            "verification_attempts": prev_attempts + 1,
            "events": [events.verification(status="error", message=str(e))],
        }


def should_verify_or_finalize(state: OrchestratorState) -> Literal["plan", "finalize"]:
    """Route after verification."""
    status = (state.get("verification_status") or "passed").lower()
    if status == "passed":
        return "finalize"
    attempts = state.get("verification_attempts", 0) or 0
    if attempts >= 2:
        return "finalize"
    return "plan"


async def finalize_node(state: OrchestratorState) -> dict:
    """Produce final response from step results."""
    response = state.get("response", "")

    if not response and state.get("step_results"):
        results = state.get("step_results", [])
        parts = [r["output"] for r in results if r["output"]]
        response = "\n\n".join(parts) if parts else "Task completed."

    if not response:
        response = "I was unable to complete the task. Please try rephrasing."

    return {
        "response": response,
        "events": [events.stage("orchestrator", "Response generated", "completed")],
    }


def should_plan_or_dispatch(state: OrchestratorState) -> Literal["plan", "dispatch_step"]:
    """Route after classify."""
    if state.get("query_complexity") == "complex":
        return "plan"
    return "dispatch_step"


def create_orchestrator_graph(checkpointer=None):
    """Create the orchestrator graph."""
    graph = StateGraph(OrchestratorState)

    graph.add_node("classify", classify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("dispatch_step", dispatch_step_node)
    graph.add_node("verify", verify_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        should_plan_or_dispatch,
        {"plan": "plan", "dispatch_step": "dispatch_step"},
    )
    graph.add_edge("plan", "dispatch_step")
    graph.add_conditional_edges(
        "dispatch_step",
        step_check,
        {"dispatch_step": "dispatch_step", "verify": "verify", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "verify",
        should_verify_or_finalize,
        {"plan": "plan", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# AgentOrchestrator Wrapper
# =============================================================================


class AgentOrchestrator:
    """High-level wrapper for the orchestrator graph.

    Drop-in replacement for AgentSupervisor with the same run()/invoke() API.
    """

    def __init__(self, checkpointer=None):
        """Initialize the orchestrator.

        Args:
            checkpointer: Optional checkpointer for state persistence.
                          If None, a fresh MemorySaver is created per request
                          in run()/invoke() to avoid unbounded memory growth.
        """
        self._checkpointer_factory = checkpointer

    def _create_graph(self):
        """Create a graph with a fresh checkpointer per request.

        Using a fresh MemorySaver per request prevents unbounded memory
        growth from accumulating checkpoint data across many requests.
        """
        from langgraph.checkpoint.memory import MemorySaver

        cp = self._checkpointer_factory if self._checkpointer_factory else MemorySaver()
        return create_orchestrator_graph(checkpointer=cp)

    async def run(
        self,
        query: str,
        mode: str | None = None,
        task_id: str | None = None,
        user_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncGenerator[dict, None]:
        """Run the orchestrator and yield events.

        Same API as AgentSupervisor.run() for drop-in compatibility.

        Args:
            query: User query to process
            mode: Optional explicit agent mode (chat, research, app, data, image)
            task_id: Optional task ID for tracking
            user_id: Optional user ID
            messages: Optional chat history
            **kwargs: Additional parameters passed to subagents

        Yields:
            Event dictionaries for streaming to clients
        """
        from app.agents.budget import (
            apply_budget_pressure_defaults,
            derive_budget_pressure_state,
        )
        from app.agents.stream_processor import StreamProcessor
        from app.models.schemas import ResearchDepth
        from app.services.usage_tracker import create_usage_tracker

        effective_task_id = task_id or str(uuid.uuid4())
        effective_run_id = kwargs.get("run_id") or str(uuid.uuid4())
        original_mode = mode.lower() if isinstance(mode, str) else mode

        requested_tier = kwargs.get("tier")
        requested_depth = kwargs.get("depth", ResearchDepth.FAST)
        requested_budget = kwargs.get("budget") or {}
        effective_tier, effective_depth, budget_adjustment = apply_budget_pressure_defaults(
            budget=requested_budget,
            mode=original_mode,
            tier=requested_tier,
            depth=requested_depth,
        )

        # Build initial state
        initial_state: OrchestratorState = {
            "query": query,
            "mode": original_mode,
            "task_id": effective_task_id,
            "run_id": effective_run_id,
            "user_id": user_id,
            "messages": messages or [],
            "events": [],
            "budget": requested_budget,
            "execution_mode": kwargs.get("execution_mode", "auto"),
            "tier": effective_tier,
            "depth": effective_depth,
            "hitl_enabled": settings.hitl_enabled,
        }

        # Add any extra kwargs (like depth for research)
        for key, value in kwargs.items():
            if key in {"tier", "depth", "budget"}:
                continue
            initial_state[key] = value

        # Create config with unique run_id for each request
        run_id = str(uuid.uuid4())
        thread_id = effective_task_id

        # Create usage tracker callback for this request
        provider = kwargs.get("provider", "anthropic")
        tier = effective_tier or "pro"
        usage_tracker = create_usage_tracker(
            conversation_id=effective_task_id,
            user_id=user_id,
            tier=str(tier) if tier else "pro",
            provider=str(provider) if provider else "anthropic",
        )

        config = {
            "configurable": {"thread_id": run_id},
            "recursion_limit": settings.langgraph_recursion_limit,
            "callbacks": [usage_tracker],
        }

        logger.info(
            "orchestrator_run_started",
            query=query[:50],
            mode=original_mode,
            thread_id=thread_id,
            run_id=effective_run_id,
            depth=initial_state.get("depth"),
        )

        # Emit initial thinking stage
        yield {
            "type": "stage",
            "name": "thinking",
            "description": "Processing your request...",
            "status": "running",
        }
        if budget_adjustment:
            yield {
                "type": "stage",
                "name": "budget_adjustment",
                "description": "Applied budget-aware model controls before execution.",
                "status": "completed",
                "budget_state": {
                    "exhausted": False,
                    "pressure_ratio": 0.0,
                    "adjustment": budget_adjustment,
                },
            }

        # Initialize stream processor
        processor = StreamProcessor(
            user_id=user_id,
            task_id=effective_task_id,
            thread_id=thread_id,
            run_id=effective_run_id,
        )
        budget = initial_state.get("budget") or {}
        has_budget = bool(budget and any(budget.values()))
        budget_exhausted = False

        _run_start = _time.monotonic()
        _budget_check_counter = 0

        try:
            graph = self._create_graph()
            async for event in graph.astream_events(
                initial_state,
                config=config,
                version="v2",
            ):
                if not isinstance(event, dict):
                    continue

                async for processed_event in processor.process_event(event):
                    yield processed_event
                    # Throttle budget checks: only evaluate every 20 events
                    # (tool results, LLM completions) instead of every token
                    if has_budget:
                        _budget_check_counter += 1
                        if _budget_check_counter % 20 == 0:
                            usage_totals = usage_tracker.get_total_tokens()
                            tool_calls_count = len(processor.emitted_tool_call_ids)
                            elapsed_seconds = int(_time.monotonic() - _run_start)
                            budget_state = derive_budget_pressure_state(
                                budget=budget,
                                usage_totals=usage_totals,
                                tool_calls_count=tool_calls_count,
                                elapsed_seconds=elapsed_seconds,
                            )
                            if 0.8 <= budget_state["pressure_ratio"] < 1.0:
                                yield {
                                    "type": "reasoning",
                                    "thinking": "Budget pressure rising; execution may degrade depth or stop soon.",
                                    "context": "budget",
                                    "budget_state": budget_state,
                                }
                            if budget_state["exhausted"]:
                                budget_exhausted = True
                                yield {
                                    "type": "stage",
                                    "name": "budget_stop",
                                    "description": "Execution stopped because run budget was exhausted.",
                                    "status": "completed",
                                    "budget_state": budget_state,
                                }
                                break
                if budget_exhausted:
                    break

            # Emit usage metrics before completion
            stream_has_usage = processor.total_input_tokens > 0 or processor.total_output_tokens > 0
            usage_totals = usage_tracker.get_total_tokens()
            if not stream_has_usage and usage_totals.get("call_count", 0) > 0:
                yield events.usage(
                    input_tokens=usage_totals["input_tokens"],
                    output_tokens=usage_totals["output_tokens"],
                    cached_tokens=usage_totals.get("cached_tokens", 0),
                    cost_usd=usage_totals["cost_usd"],
                    model="aggregate",
                    tier=str(tier) if tier else "pro",
                )
            if usage_totals.get("call_count", 0) > 0:
                logger.info(
                    "usage_tracked",
                    total_tokens=usage_totals["total_tokens"],
                    cost_usd=usage_totals["cost_usd"],
                    call_count=usage_totals["call_count"],
                    source="stream" if stream_has_usage else "callback",
                )

            # Emit completion event
            yield {"type": "complete"}

            logger.info("orchestrator_run_completed", thread_id=thread_id)

            # Fire-and-forget: extract memories from this conversation
            memory_enabled = initial_state.get("memory_enabled", True)
            if user_id and messages and memory_enabled:
                _duration_s = round(_time.monotonic() - _run_start, 1)
                _tools_used = sorted(
                    {info.get("tool", "") for info in processor.pending_tool_calls.values() if info.get("tool")}
                    | set(processor.pending_tool_calls_by_tool.keys())
                    | {info.get("tool", "") for info in getattr(processor, "_completed_tools", []) if info.get("tool")}
                )

                episodic_context = {
                    "task_description": query[:500],
                    "mode": original_mode,
                    "tools_used": _tools_used,
                    "outcome": "completed",
                    "duration_seconds": _duration_s,
                }

                asyncio.create_task(
                    self._extract_memories(
                        messages=messages,
                        user_id=user_id,
                        conversation_id=effective_task_id,
                        episodic_context=episodic_context,
                    )
                )

        except Exception as e:
            logger.error("orchestrator_run_failed", error=str(e), thread_id=thread_id)
            yield {"type": "error", "error": str(e)}

    @staticmethod
    async def _extract_memories(
        messages: list[dict],
        user_id: str,
        conversation_id: str,
        episodic_context: dict | None = None,
    ) -> None:
        """Background task to extract and persist memories from a conversation."""
        try:
            from app.services.memory_service import extract_memories_from_conversation

            await extract_memories_from_conversation(
                messages=messages,
                user_id=user_id,
                conversation_id=conversation_id,
                episodic_context=episodic_context,
            )
        except Exception as e:
            logger.warning("background_memory_extraction_failed", error=str(e))

    async def invoke(
        self,
        query: str,
        mode: str | None = None,
        **kwargs,
    ) -> dict:
        """Run the orchestrator and return final result (non-streaming).

        Args:
            query: User query to process
            mode: Optional explicit agent mode
            **kwargs: Additional parameters

        Returns:
            Final result dictionary with response
        """
        initial_state: OrchestratorState = {
            "query": query,
            "mode": mode,
            "events": [],
            "hitl_enabled": settings.hitl_enabled,
            **kwargs,
        }

        config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": settings.langgraph_recursion_limit,
        }

        graph = self._create_graph()
        result = await graph.ainvoke(initial_state, config=config)
        return result


# Global instance for convenience
agent_orchestrator = AgentOrchestrator()
