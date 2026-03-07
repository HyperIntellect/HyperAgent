"""Supervisor/orchestrator for the multi-agent system with handoff support.

.. deprecated::
    This module is deprecated. Use ``app.agents.orchestrator`` instead.
    Kept for backward compatibility.
"""

import asyncio
import time as _time
import uuid
from typing import Any, AsyncGenerator, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents import events
from app.agents.routing import route_query
from app.agents.state import (
    AgentType,
    ErrorOutput,
    RouterOutput,
    SupervisorState,
    TaskOutput,
)
from app.agents.budget import (
    apply_budget_pressure_defaults,
    derive_budget_pressure_state,
)
from app.agents.executor import executor_subgraph as task_subgraph
from app.agents.tools.handoff import (
    HANDOFF_MATRIX,
    MAX_HANDOFFS,
    build_query_with_context,
    can_handoff,
    update_handoff_history,
)
from app.core.logging import get_logger
from app.models.schemas import ResearchDepth
from app.services.usage_tracker import create_usage_tracker

logger = get_logger(__name__)

# Default timeout for subgraph invocations (in seconds)
DEFAULT_SUBGRAPH_TIMEOUT = 300  # 5 minutes


async def router_node(state: SupervisorState) -> RouterOutput:
    """Route the query to the appropriate agent.

    Includes input validation and error handling.

    Args:
        state: Current supervisor state

    Returns:
        RouterOutput with selected_agent and routing_reason
    """
    try:
        # Input validation
        query = (state.get("query") or "").strip()
        if not query:
            logger.warning("router_empty_query")
            return RouterOutput(
                selected_agent=AgentType.TASK.value,
                routing_reason="Empty query - defaulting to task",
                routing_confidence=1.0,
                active_agent=AgentType.TASK.value,
                events=[
                    {
                        "type": "validation",
                        "warning": "Empty query received",
                    }
                ],
            )

        # Check if there's a pending handoff - use that instead of routing
        pending_handoff = state.get("pending_handoff")
        if pending_handoff:
            target_agent = pending_handoff.get("target_agent", "task")
            logger.info(
                "routing_from_handoff",
                target=target_agent,
                task=pending_handoff.get("task_description", "")[:50],
            )

            # Restore artifacts from source sandbox into target sandbox
            handoff_artifacts = pending_handoff.get("handoff_artifacts") or []
            artifact_summary = ""
            if handoff_artifacts:
                restored = await _restore_handoff_artifacts(state, handoff_artifacts)
                if restored:
                    from app.sandbox.artifact_transfer import format_artifact_summary
                    artifact_summary = format_artifact_summary(handoff_artifacts)
                    logger.info(
                        "handoff_artifacts_restored",
                        count=len(restored),
                        target=target_agent,
                    )

            handoff_context = pending_handoff.get("context", "")
            if artifact_summary:
                handoff_context = (
                    f"{handoff_context}\n\n{artifact_summary}"
                    if handoff_context
                    else artifact_summary
                )

            return RouterOutput(
                selected_agent=target_agent,
                routing_reason=f"Handoff from {pending_handoff.get('source_agent', 'unknown')}",
                active_agent=target_agent,
                delegated_task=pending_handoff.get("task_description"),
                handoff_context=handoff_context,
                pending_handoff=None,  # Clear the pending handoff
                events=[
                    {
                        "type": "handoff",
                        "source": pending_handoff.get("source_agent"),
                        "target": target_agent,
                        "task": pending_handoff.get("task_description"),
                    }
                ],
            )

        result = await route_query(state)

        # Set active_agent from routing
        selected_agent = result.get("selected_agent", "task")
        result["active_agent"] = selected_agent

        return RouterOutput(**result)

    except Exception as e:
        logger.error("router_node_failed", error=str(e))
        # Default to task on routing failure
        return RouterOutput(
            selected_agent=AgentType.TASK.value,
            routing_reason="Routing error occurred",
            routing_confidence=0.0,
            active_agent=AgentType.TASK.value,
            events=[
                {
                    "type": "error",
                    "node": "router",
                    "error": str(e),
                }
            ],
        )


async def task_node(state: SupervisorState, config: dict | None = None) -> TaskOutput | ErrorOutput:
    """Execute the task subagent.

    Args:
        state: Current supervisor state
        config: LangGraph runnable config (callbacks, metadata) — passed through
                so that custom events dispatched inside tools propagate to the
                supervisor's astream_events stream.

    Returns:
        TaskOutput with response, events, and potential handoff
    """
    try:
        from app.config import settings

        # Skip parallel decomposition when explicit skills are selected
        explicit_skills = state.get("skills") or []
        skip_parallel_events: list[dict] = []
        if explicit_skills and settings.parallel_executor_v1 and state.get("parallel_eligible"):
            logger.info(
                "parallel_skipped_explicit_skills",
                skills=explicit_skills,
            )
            skip_parallel_events.append({
                "type": "reasoning",
                "thinking": f"Skipping parallel decomposition: explicit skill(s) selected ({', '.join(explicit_skills)})",
                "context": "parallel_routing",
            })

        if (
            settings.parallel_executor_v1
            and state.get("parallel_eligible")
            and (state.get("mode") in (None, "task"))
            and not explicit_skills
        ):
            from app.agents.parallel import GeneralParallelExecutor
            from app.ai.model_tiers import get_quality_profile

            parallel_events: list[dict] = []

            def _on_progress(evt: dict):
                parallel_events.append(evt)

            profile = get_quality_profile(state.get("tier"))
            max_agents = min(profile.parallel_max_agents, settings.parallel_executor_max_agents)
            executor = GeneralParallelExecutor(max_agents=max_agents)
            result = await executor.execute(
                query=_build_query_with_context(state),
                provider=state.get("provider"),
                on_progress=_on_progress,
            )
            return TaskOutput(
                response=result.synthesis,
                events=parallel_events + [{
                    "type": "reasoning",
                    "thinking": (
                        f"Executed {result.successful_count + result.failed_count} sub-tasks in parallel; "
                        f"{result.successful_count} succeeded, {result.failed_count} failed."
                    ),
                    "context": "parallel_execution",
                }],
            )

        # Build input state with handoff context if present
        # Include task_id so browser tools use the same sandbox session as the stream viewer
        input_state = {
            "query": _build_query_with_context(state),
            "mode": state.get("mode"),
            "messages": state.get("messages") or [],
            "user_id": state.get("user_id"),
            "task_id": state.get("task_id"),
            "attachment_ids": state.get("attachment_ids") or [],
            "image_attachments": state.get("image_attachments") or [],
            "attachment_context": state.get("attachment_context"),
            "system_prompt": state.get("system_prompt"),
            "provider": state.get("provider"),
            "model": state.get("model"),
            "tier": state.get("tier"),
            "shared_memory": state.get("shared_memory") or {},
            "locale": state.get("locale", "en"),
            "skills": state.get("skills") or [],
            "depth": state.get("depth"),
            "hitl_enabled": state.get("hitl_enabled", settings.hitl_enabled),
        }

        # Invoke task subgraph with timeout
        # Pass config so callbacks (dispatch_custom_event) propagate for real-time streaming
        timeout = getattr(settings, "subgraph_timeout", DEFAULT_SUBGRAPH_TIMEOUT)
        # App builder and deep research need more time
        mode = state.get("mode")
        if mode in ("app", "research"):
            timeout = max(timeout, 600)
        async with asyncio.timeout(timeout):
            result = await task_subgraph.ainvoke(input_state, config=config)

        output: TaskOutput = {
            "response": result.get("response", ""),
            "events": skip_parallel_events + result.get("events", []),
        }

        await _process_handoff(output, state, result.get("pending_handoff"), "task")

        return output

    except asyncio.TimeoutError:
        logger.error("task_node_timeout", task_id=state.get("task_id"))
        return ErrorOutput(
            response="I'm sorry, the request took too long to process. Please try again with a simpler query.",
            events=[{"type": "error", "node": "task", "error": "Timeout"}],
            has_error=True,
        )
    except Exception as e:
        logger.error("task_node_failed", error=str(e), task_id=state.get("task_id"))
        return ErrorOutput(
            response="I encountered an error while processing your request. Please try again.",
            events=[{"type": "error", "node": "task", "error": str(e)}],
            has_error=True,
        )


def check_for_handoff(state: SupervisorState) -> Literal["router", "__end__"]:
    """Check if there's a pending handoff that needs processing.

    Args:
        state: Current supervisor state

    Returns:
        "router" if handoff pending, "__end__" otherwise
    """
    pending_handoff = state.get("pending_handoff")
    if pending_handoff:
        target = pending_handoff.get("target_agent", "")
        handoff_count = state.get("handoff_count", 0)

        # Check handoff limits
        if handoff_count >= MAX_HANDOFFS:
            logger.warning(
                "max_handoffs_reached",
                count=handoff_count,
                max=MAX_HANDOFFS,
            )
            return END

        # Validate handoff target
        current_agent = state.get("active_agent", "task")
        allowed_targets = HANDOFF_MATRIX.get(current_agent, [])
        if target not in allowed_targets:
            logger.warning(
                "invalid_handoff_target",
                current=current_agent,
                target=target,
                allowed=allowed_targets,
            )
            return END

        logger.info(
            "handoff_detected",
            from_agent=current_agent,
            to_agent=target,
            handoff_count=handoff_count,
        )
        return "router"

    return END


def _build_query_with_context(state: SupervisorState) -> str:
    """Build query string with handoff context if present.

    Thin wrapper that extracts state fields and delegates to handoff module.

    Args:
        state: Current supervisor state

    Returns:
        Query string with optional context
    """
    return build_query_with_context(
        query=state.get("query") or "",
        delegated_task=state.get("delegated_task"),
        handoff_context=state.get("handoff_context"),
        shared_memory=state.get("shared_memory"),
    )


def _can_handoff(state: SupervisorState, target_agent: str) -> bool:
    """Check if a handoff to the target agent is allowed.

    Thin wrapper that extracts state fields and delegates to handoff module.

    Args:
        state: Current supervisor state
        target_agent: Target agent for handoff

    Returns:
        True if handoff is allowed
    """
    return can_handoff(
        current_agent=state.get("active_agent") or "task",
        target_agent=target_agent,
        handoff_count=state.get("handoff_count", 0),
        handoff_history=state.get("handoff_history"),
    )


async def _process_handoff(
    output: dict,
    state: SupervisorState,
    handoff: dict | None,
    agent_name: str,
) -> None:
    """Validate and apply handoff to output dict, or set fallback response.

    Collects sandbox artifacts from the source agent when a valid handoff
    is detected. Artifact transfer is best-effort; the handoff proceeds
    even if artifact collection fails.

    Args:
        output: Output dict to update in-place
        state: Current supervisor state
        handoff: Handoff info dict from agent result, or None
        agent_name: Name of the current agent (for logging)
    """
    if not handoff:
        return

    if _can_handoff(state, handoff.get("target_agent", "")):
        # Attempt to collect artifacts from the source agent's sandbox
        artifacts = await _collect_handoff_artifacts(state)
        if artifacts:
            handoff["handoff_artifacts"] = artifacts

        output["pending_handoff"] = handoff
        output["handoff_count"] = state.get("handoff_count", 0) + 1
        output["handoff_history"] = update_handoff_history(
            history=list(state.get("handoff_history") or []),
            source_agent=state.get("active_agent") or "task",
            handoff=handoff,
        )
    else:
        logger.warning(
            "handoff_blocked",
            agent_name=agent_name,
            target=handoff.get("target_agent", ""),
        )
        if not output.get("response"):
            output["response"] = (
                "I couldn't hand off that request, so I'm continuing here. "
                "Please share more details or rephrase."
            )


async def _collect_handoff_artifacts(state: SupervisorState) -> list[dict]:
    """Best-effort collection of sandbox artifacts for handoff transfer.

    Args:
        state: Current supervisor state (used to find the active sandbox)

    Returns:
        List of artifact dicts, or empty list on failure
    """
    try:
        from app.sandbox import get_execution_sandbox_manager
        from app.sandbox.artifact_transfer import collect_artifacts

        manager = await get_execution_sandbox_manager()
        session = await manager.get_session(
            user_id=state.get("user_id"),
            task_id=state.get("task_id"),
        )
        if not session:
            return []

        runtime = session.executor.get_runtime()
        return await collect_artifacts(runtime)
    except Exception as e:
        logger.warning("handoff_artifact_collection_failed", error=str(e))
        return []


async def _restore_handoff_artifacts(
    state: SupervisorState,
    artifacts: list[dict],
) -> list[str]:
    """Restore handoff artifacts into the target agent's sandbox.

    Creates a sandbox session for the target if one doesn't exist yet.

    Args:
        state: Current supervisor state
        artifacts: List of artifact dicts from HandoffInfo

    Returns:
        List of restored file paths
    """
    try:
        from app.sandbox import get_execution_sandbox_manager
        from app.sandbox.artifact_transfer import restore_artifacts

        manager = await get_execution_sandbox_manager()
        session = await manager.get_or_create_sandbox(
            user_id=state.get("user_id"),
            task_id=state.get("task_id"),
        )
        runtime = session.executor.get_runtime()
        return await restore_artifacts(runtime, artifacts)
    except Exception as e:
        logger.warning("handoff_artifact_restore_failed", error=str(e))
        return []


def create_supervisor_graph(checkpointer=None):
    """Create the supervisor graph that orchestrates the task agent.

    Research is handled as a skill (deep_research) invoked by the task agent,
    so the supervisor only needs to route to the task node.

    Args:
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled supervisor graph
    """
    graph = StateGraph(SupervisorState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("task", task_node)

    # Set entry point
    graph.set_entry_point("router")

    # All queries route to task agent (research is now a skill)
    graph.add_edge("router", "task")

    # After task agent, check for handoff or end
    graph.add_conditional_edges(
        "task",
        check_for_handoff,
        {
            "router": "router",
            END: END,
        },
    )

    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# Factory Functions (for better testability and thread safety)
# =============================================================================



class AgentSupervisor:
    """High-level wrapper for the supervisor graph.

    Provides a clean interface for running the multi-agent system
    with support for both synchronous and streaming execution.
    """

    def __init__(self, checkpointer=None):
        """Initialize the supervisor.

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
        cp = self._checkpointer_factory if self._checkpointer_factory else MemorySaver()
        return create_supervisor_graph(checkpointer=cp)

    async def run(
        self,
        query: str,
        mode: str | None = None,
        task_id: str | None = None,
        user_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncGenerator[dict, None]:
        """Run the appropriate agent and yield events.

        This method maintains backward compatibility with the existing
        research agent interface while supporting the new multi-agent system.

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
        from app.agents.stream_processor import StreamProcessor
        from app.config import settings

        effective_task_id = task_id or str(uuid.uuid4())
        effective_run_id = kwargs.get("run_id") or str(uuid.uuid4())
        # Preserve original mode for subagents (important for image/app modes)
        # The router will handle normalization for routing decisions
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
        initial_state: SupervisorState = {
            "query": query,
            "mode": original_mode,  # Preserve original mode (e.g., "image", "app")
            "task_id": effective_task_id,
            "run_id": effective_run_id,
            "user_id": user_id,
            "messages": messages or [],
            "events": [],
            "handoff_count": 0,
            "handoff_history": [],
            "shared_memory": {},
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
        # Using a unique ID per request prevents checkpoint state (like events) from carrying over
        # Conversation history is passed explicitly via messages, not through checkpointing
        run_id = str(uuid.uuid4())  # Unique per request, not conversation
        thread_id = effective_task_id  # Still log with conversation ID

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
            "configurable": {"thread_id": run_id},  # Use unique run_id for checkpointing
            "recursion_limit": settings.langgraph_recursion_limit,
            "callbacks": [usage_tracker],  # Wire usage tracking into all LLM calls
        }

        logger.info(
            "supervisor_run_started",
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
            # Stream events from the graph
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

            # Emit usage metrics before completion.
            # If the stream processor already emitted per-call usage events
            # (from on_chat_model_end), skip the aggregate to avoid double counting.
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

            logger.info("supervisor_run_completed", thread_id=thread_id)

            # Fire-and-forget: extract memories from this conversation
            memory_enabled = initial_state.get("memory_enabled", True)
            if user_id and messages and memory_enabled:
                # Collect episodic context from the run
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
            logger.error("supervisor_run_failed", error=str(e), thread_id=thread_id)
            yield {"type": "error", "error": str(e)}

    @staticmethod
    async def _extract_memories(
        messages: list[dict],
        user_id: str,
        conversation_id: str,
        episodic_context: dict | None = None,
    ) -> None:
        """Background task to extract and persist memories from a conversation.

        Args:
            messages: Conversation messages
            user_id: User ID
            conversation_id: Conversation/task ID
            episodic_context: Optional dict with task_description, tools_used,
                outcome, duration_seconds for enriched episodic memories
        """
        try:
            from app.services.memory_service import extract_memories_from_conversation

            await extract_memories_from_conversation(
                messages=messages,
                user_id=user_id,
                conversation_id=conversation_id,
                episodic_context=episodic_context,
            )
        except Exception as e:
            # Never let memory extraction crash anything
            logger.warning("background_memory_extraction_failed", error=str(e))

    async def invoke(
        self,
        query: str,
        mode: str | None = None,
        **kwargs,
    ) -> dict:
        """Run the agent and return final result (non-streaming).

        Args:
            query: User query to process
            mode: Optional explicit agent mode
            **kwargs: Additional parameters

        Returns:
            Final result dictionary with response
        """
        from app.config import settings

        initial_state: SupervisorState = {
            "query": query,
            "mode": mode,
            "events": [],
            "handoff_count": 0,
            "handoff_history": [],
            "shared_memory": {},
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
agent_supervisor = AgentSupervisor()
