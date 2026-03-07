"""Executor subagent: a self-contained ReAct loop for executing a single step/task.

Extracted from task.py — runs reason → act → reason → ... → complete without
plan tracking or verification logic. Those concerns live in the orchestrator.
"""

import time as _time
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph

from app.agents import events
from app.agents.hitl.interrupt_manager import get_interrupt_manager
from app.agents.react_helpers import (
    RECOVERY_HINTS,
    TODO_INJECT_CAP,
    apply_anti_repetition,
    apply_compression_if_needed as _apply_compression_if_needed,
    apply_output_guardrail,
    build_initial_messages as _build_initial_messages,
    compute_prefix_hash,
    extract_last_ai_response,
    get_cached_tool_map,
    get_cached_tools,
    get_react_config_for_state as _get_react_config_for_state,
    handle_direct_skill as _handle_direct_skill,
    invoke_llm_streaming as _invoke_llm_streaming,
    read_sandbox_todo as _read_sandbox_todo,
)
from app.agents.reflection import reflect, should_reflect
from app.agents.state import ExecutorState
from app.agents.tools.react_tool import (
    ErrorCategory,
    classify_error,
    deduplicate_tool_messages,
)
from app.agents.tools.tool_pipeline import (
    TaskToolHooks,
    ToolExecutionContext,
    execute_tool,
    execute_tools_batch,
)
from app.ai.llm import extract_text_from_content, llm_service
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ===========================================================================
# Graph nodes
# ===========================================================================


async def executor_reason_node(state: ExecutorState) -> dict:
    """ReAct reason node: LLM reasons about what to do next.

    Uses ``step_goal`` as the user query. Does NOT inject plan execution
    guidance — that responsibility belongs to the orchestrator.
    """
    query = state.get("step_goal") or ""
    provider = state.get("provider")
    locale = state.get("locale", "en")
    existing_summary = state.get("context_summary")
    # Create a copy to avoid in-place mutation issues
    lc_messages = list(state.get("lc_messages", []))

    # Clear stale recovery SystemMessages from previous iterations
    lc_messages = [
        m for m in lc_messages
        if not (isinstance(m, SystemMessage) and m.additional_kwargs.get("recovery"))
    ]

    event_list: list[dict] = []
    new_context_summary = None
    did_upload_files = False

    # Initialize messages if empty
    if not lc_messages:
        lc_messages = await _build_initial_messages(state)

        # For dedicated modes or explicit skill selection, return early
        direct_result = _handle_direct_skill(state, lc_messages, query)
        if direct_result is not None:
            return direct_result

    # Debug: Log messages before deduplication
    logger.debug(
        "executor_reason_node_messages",
        message_count=len(lc_messages),
        message_types=[type(m).__name__ for m in lc_messages],
        has_tool_messages=any(isinstance(m, ToolMessage) for m in lc_messages),
    )

    # Deduplicate tool messages to prevent API errors
    lc_messages = deduplicate_tool_messages(lc_messages)

    # Apply shared context policy (compression + truncation fallback)
    lc_messages, new_context_summary, compression_events = await _apply_compression_if_needed(
        lc_messages, state, existing_summary, provider, locale,
    )
    event_list.extend(compression_events)

    # Get LLM with tools
    tier = state.get("tier")
    model = state.get("model")
    llm = llm_service.choose_llm_for_task(
        task_type="task",
        provider=provider,
        tier_override=tier,
        model_override=model,
    )

    # Get all tools for executor (cached)
    all_tools = get_cached_tools("task")
    llm_with_tools = llm.bind_tools(all_tools) if all_tools else llm

    # === KV-Cache Optimization: Stable Prefix Tracking ===
    from app.agents.tools.registry import get_soft_disabled_message, get_soft_disabled_tools

    result_prefix_hash = compute_prefix_hash(lc_messages, state.get("prefix_hash"))

    # Inject soft-disabled tools notice
    disabled_tools = get_soft_disabled_tools()
    if disabled_tools:
        disabled_msg = get_soft_disabled_message(disabled_tools)
        if disabled_msg:
            lc_messages.append(SystemMessage(content=disabled_msg))

    # === Context Engineering: Todo-list as Attention Manipulation ===
    # Simplified: no plan execution guidance; just inject active_todo if present.
    messages_for_llm = lc_messages
    active_todo = state.get("active_todo")

    if active_todo:
        todo_text = active_todo
        sandbox_todo = await _read_sandbox_todo(
            state.get("user_id"), state.get("task_id"),
        )
        if sandbox_todo:
            todo_text = sandbox_todo[:TODO_INJECT_CAP]
            logger.debug("sandbox_todo_injected")
        todo_reminder = SystemMessage(content=(
            f"[Active Task Context]\n{todo_text}"
        ))
        messages_for_llm = list(lc_messages) + [todo_reminder]
        logger.debug("active_todo_injected")

    try:
        ai_message = await _invoke_llm_streaming(llm_with_tools, messages_for_llm)
        lc_messages.append(ai_message)

        # Emit final response token if no tool calls
        if not ai_message.tool_calls:
            response_text = extract_text_from_content(ai_message.content)
            if response_text:
                await apply_output_guardrail(response_text, query, event_list)

        result: dict = {
            "lc_messages": lc_messages,
            "events": event_list,
            "has_error": False,
        }
        if did_upload_files:
            result["files_uploaded_to_sandbox"] = True
        if new_context_summary:
            result["context_summary"] = new_context_summary
        if result_prefix_hash:
            result["prefix_hash"] = result_prefix_hash
        # Set loop_start_time on first iteration for wall-clock timeout tracking
        if state.get("loop_start_time") is None:
            result["loop_start_time"] = _time.monotonic()
        return result
    except Exception as e:
        logger.error("executor_reason_failed", error=str(e))
        error_msg = str(e)

        if "prompt is too long" in error_msg:
            response = (
                "I apologize, but the conversation has become too long. "
                "Please start a new conversation or try a simpler request."
            )
        else:
            response = f"I apologize, but I encountered an error: {e}"

        event_list.append(events.token(response))

        return {
            "lc_messages": lc_messages,
            "result": response,
            "events": event_list,
            "has_error": True,
        }


async def executor_act_node(state: ExecutorState) -> dict:
    """ReAct act node: Execute tools based on LLM's tool calls.

    Simplified from task.py — no plan step advancement or plan revision logic.
    """
    from app.agents.hitl.tool_risk import requires_approval

    lc_messages = list(state.get("lc_messages", []))
    event_list: list[dict] = []

    # Get the last AI message (should have tool calls)
    ai_message = None
    for msg in reversed(lc_messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            ai_message = msg
            break

    if not ai_message or not ai_message.tool_calls:
        return {"events": event_list}

    # Check which tool calls already have results to avoid duplicates
    existing_tool_result_ids = {
        msg.tool_call_id
        for msg in lc_messages
        if isinstance(msg, ToolMessage)
    }

    pending_tool_calls = [
        tc for tc in ai_message.tool_calls
        if tc.get("id", "") not in existing_tool_result_ids
    ]

    if not pending_tool_calls:
        logger.warning(
            "executor_act_node_no_pending_tool_calls",
            existing_count=len(existing_tool_result_ids),
        )
        return {"events": event_list, "lc_messages": lc_messages}

    tool_map = get_cached_tool_map("task")
    config = _get_react_config_for_state(state)

    def hitl_check(tool_name: str) -> bool:
        return tool_name in {"ask_user", "invoke_skill"} or requires_approval(
            tool_name,
            auto_approve_tools=state.get("auto_approve_tools", []),
            hitl_enabled=state.get("hitl_enabled", True),
            risk_threshold=settings.hitl_default_risk_threshold,
        )

    hooks = TaskToolHooks(state=state)
    tool_messages, batch_events, error_count, pending_interrupt = (
        await execute_tools_batch(
            tool_calls=pending_tool_calls,
            tool_map=tool_map,
            config=config,
            hooks=hooks,
            user_id=state.get("user_id"),
            task_id=state.get("task_id"),
            run_id=state.get("run_id"),
            hitl_partition=True,
            hitl_check=hitl_check,
        )
    )

    event_list.extend(batch_events)
    lc_messages.extend(tool_messages)

    extras: dict = {}
    tool_iterations = state.get("tool_iterations", 0) + (1 if tool_messages else 0)

    if pending_interrupt:
        extras["pending_interrupt"] = pending_interrupt
        tool_iterations = state.get("tool_iterations", 0)

    # --- Self-correction: consecutive error tracking ---
    if tool_messages and not pending_interrupt:
        has_errors = error_count > 0
        prev_consecutive = state.get("consecutive_errors", 0)
        if has_errors:
            first_error = next(
                (msg.content for msg in tool_messages
                 if isinstance(msg, ToolMessage) and msg.content),
                "",
            )
            error_cat = classify_error(first_error or "")

            new_consecutive = prev_consecutive + 1
            extras["consecutive_errors"] = new_consecutive
            if new_consecutive >= 3:
                recovery_msg = (
                    "CRITICAL: You have failed 3 consecutive tool calls. "
                    "You MUST change your approach:\n"
                    f"- Error category: {error_cat.value}\n"
                    f"- Last error: {first_error[:300] if first_error else 'unknown'}\n\n"
                    "Recovery strategies (pick one):\n"
                    "1. Use a completely different tool or method\n"
                    "2. Simplify the task — break it into smaller sub-steps\n"
                    "3. Use ask_user to get clarification or help\n"
                    "4. Skip this step and move on if possible\n\n"
                    "Do NOT retry the same failing approach."
                )
                lc_messages.append(SystemMessage(
                    content=recovery_msg,
                    additional_kwargs={"recovery": True},
                ))
                event_list.append(events.reasoning(
                    thinking=f"3 consecutive errors ({error_cat.value}). Forcing strategy change.",
                    confidence=0.2,
                    context="error_recovery",
                ))
                logger.warning(
                    "consecutive_errors_threshold",
                    count=new_consecutive,
                    category=error_cat.value,
                )
            else:
                hint = RECOVERY_HINTS.get(
                    error_cat,
                    "The previous tool call failed. Analyze the error and try a different approach.",
                )
                lc_messages.append(SystemMessage(
                    content=f"Tool error ({error_cat.value}): {hint}",
                    additional_kwargs={"recovery": True},
                ))
                event_list.append(events.reasoning(
                    thinking=f"Tool error classified as {error_cat.value}. Recovery: {hint}",
                    context="error_recovery",
                ))
                logger.info(
                    "consecutive_error_detected",
                    count=new_consecutive,
                    category=error_cat.value,
                )
        else:
            if prev_consecutive > 0:
                logger.info("consecutive_errors_reset", previous_count=prev_consecutive)
            extras["consecutive_errors"] = 0

    # --- Anti-Repetition: Detect consecutive identical tool calls ---
    if tool_messages and not pending_interrupt:
        prev_hashes = list(state.get("last_tool_calls_hash", []) or [])
        apply_anti_repetition(
            pending_tool_calls, prev_hashes, lc_messages, event_list, extras,
        )

    # --- Context Engineering: Update active_todo (simplified, no plan progress) ---
    if tool_messages and not pending_interrupt:
        tool_names = [
            msg.name for msg in tool_messages
            if isinstance(msg, ToolMessage) and msg.name
        ]
        iteration_count = state.get("tool_iterations", 0) + 1
        if tool_names:
            extras["active_todo"] = f"Iteration {iteration_count}: Used {', '.join(tool_names)}"

    # Build the final result dict
    result = {
        "lc_messages": lc_messages,
        "events": event_list,
        "tool_iterations": tool_iterations,
        **extras,
    }
    return result


async def executor_complete_node(state: ExecutorState) -> dict:
    """Extract final result from messages and set status.

    Sets ``result`` and ``status`` for the orchestrator to consume.
    """
    result_text = extract_last_ai_response(state.get("lc_messages", []))
    if not result_text:
        result_text = "Step completed without explicit output."

    status = "failed" if state.get("has_error") else "completed"

    return {
        "result": result_text,
        "status": status,
        "events": [events.stage("executor", "Step completed", "completed")],
    }


# ===========================================================================
# Routing functions
# ===========================================================================


def executor_should_continue(
    state: ExecutorState,
) -> Literal["act", "complete", "reflect"]:
    """Decide whether to continue the ReAct loop, reflect, or complete.

    Routes to ``reflect`` when the LLM produces no tool calls but the
    reflection gate determines a quality check is warranted.
    """
    # If there was an error, stop immediately
    if state.get("has_error"):
        logger.info("executor_stopping_due_to_error")
        return "complete"

    # If result is already set (from error handling), stop
    if state.get("result"):
        return "complete"

    # Wall-clock timeout
    loop_start = state.get("loop_start_time")
    if loop_start is not None:
        elapsed = _time.monotonic() - loop_start
        if elapsed >= settings.task_agent_timeout:
            logger.warning(
                "executor_wall_clock_timeout",
                elapsed_seconds=round(elapsed, 1),
                timeout=settings.task_agent_timeout,
            )
            return "complete"

    lc_messages = state.get("lc_messages", [])

    # Check if last message has tool calls
    for msg in reversed(lc_messages):
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                # Check iteration limit
                config = _get_react_config_for_state(state)
                iters = state.get("tool_iterations", 0)
                if iters >= config.max_iterations:
                    logger.warning(
                        "executor_max_iterations_reached",
                        iterations=iters,
                    )
                    return "complete"
                return "act"
            else:
                # No tool calls — check if reflection should fire
                if should_reflect(state):
                    logger.info("executor_routing_to_reflect")
                    return "reflect"
                return "complete"

    return "complete"


def executor_should_wait_or_reason(
    state: ExecutorState,
) -> Literal["wait_interrupt", "reason"]:
    """Decide whether to wait for interrupt response or continue reasoning."""
    pending = state.get("pending_interrupt")
    if pending:
        logger.info(
            "executor_routing_to_wait_interrupt",
            interrupt_id=pending.get("interrupt_id"),
            thread_id=pending.get("thread_id"),
        )
        return "wait_interrupt"
    logger.debug("executor_routing_to_reason", has_pending=False)
    return "reason"


async def executor_wait_interrupt_node(state: ExecutorState) -> dict:
    """Wait for user response to a pending interrupt and add result."""
    logger.info("executor_wait_interrupt_node_started")

    pending = state.get("pending_interrupt")
    if not pending:
        logger.warning("executor_wait_interrupt_node_no_pending")
        return {"pending_interrupt": None}

    interrupt_id = pending.get("interrupt_id")
    thread_id = pending.get("thread_id", "default")
    tool_call_id = pending.get("tool_call_id")
    tool_name = pending.get("tool_name", "ask_user")

    interrupt_manager = get_interrupt_manager()
    event_list: list[dict] = []
    lc_messages = list(state.get("lc_messages", []))

    logger.debug(
        "executor_wait_interrupt_messages_before",
        message_count=len(lc_messages),
        message_types=[type(m).__name__ for m in lc_messages],
    )

    logger.info(
        "executor_wait_interrupt_subscribing",
        interrupt_id=interrupt_id,
        thread_id=thread_id,
        tool_call_id=tool_call_id,
    )

    tool_map = get_cached_tool_map("task")

    # Track auto_approve updates
    auto_approve_tools = list(state.get("auto_approve_tools", []))

    try:
        response = await interrupt_manager.wait_for_response(
            thread_id=thread_id,
            interrupt_id=interrupt_id,
            timeout_seconds=settings.hitl_decision_timeout,
        )

        action = response.get("action", "skip")
        value = response.get("value")

        logger.info(
            "executor_wait_interrupt_response_received",
            interrupt_id=interrupt_id,
            action=action,
            is_approval=pending.get("is_approval", False),
        )

        # Handle approval responses for high-risk tools
        if pending.get("is_approval"):
            tool_args = dict(pending.get("tool_args", {}) or {})
            user_id = state.get("user_id")
            task_id = state.get("task_id")

            if action in ("approve", "approve_always"):
                if action == "approve_always" and tool_name not in auto_approve_tools:
                    auto_approve_tools.append(tool_name)
                    logger.info("hitl_tool_auto_approved", tool_name=tool_name)

                config = _get_react_config_for_state(state)
                ctx = ToolExecutionContext(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_call_id=tool_call_id,
                    tool=tool_map.get(tool_name),
                    user_id=user_id,
                    task_id=task_id,
                    run_id=state.get("run_id"),
                )
                hooks = TaskToolHooks(state=state, skip_before_execution=True)
                exec_result = await execute_tool(ctx, hooks=hooks, config=config)
                if exec_result.message:
                    result_str = exec_result.message.content
                else:
                    result_str = "Tool execution returned no result."
                event_list.extend(exec_result.events)

                if action == "approve_always":
                    log_prefix = "hitl_tool_auto_approved_and_executed"
                else:
                    log_prefix = "hitl_tool_approved_and_executed"
                logger.info(log_prefix, tool_name=tool_name)

            elif action == "deny":
                result_str = f"User denied execution of {tool_name}. The tool was not executed."
                logger.info(
                    "hitl_tool_denied",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )

            else:
                result_str = f"Unknown approval action: {action}. Tool not executed."

        else:
            # Handle ask_user responses (non-approval interrupts)
            if action == "skip":
                result_str = "User skipped this question."
            elif action in ("select", "input"):
                result_str = f"User responded: {value}" if value else "User skipped this question."
            elif action in ("approve", "deny"):
                result_str = f"User responded: {value}" if value else (
                    "User confirmed." if action == "approve" else "User declined."
                )
            else:
                result_str = f"User action: {action}"

        # Add tool result message
        lc_messages.append(
            ToolMessage(
                content=result_str,
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

        # Emit response event
        event_list.append(events.tool_result(tool_name, result_str, tool_id=tool_call_id))

        logger.debug(
            "executor_wait_interrupt_messages_after",
            message_count=len(lc_messages),
            result_str=result_str,
            tool_call_id=tool_call_id,
        )

    except TimeoutError:
        logger.warning("executor_wait_interrupt_timeout", interrupt_id=interrupt_id)
        result_str = "timeout"
        lc_messages.append(
            ToolMessage(
                content=result_str,
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

    except Exception as e:
        logger.error("executor_wait_interrupt_error", error=str(e), interrupt_id=interrupt_id)
        result_str = f"error: {str(e)}"
        lc_messages.append(
            ToolMessage(
                content=result_str,
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

    result = {
        "lc_messages": lc_messages,
        "events": event_list,
        "pending_interrupt": None,  # Clear the pending interrupt
    }

    # Include auto_approve_tools if it was updated
    if auto_approve_tools != state.get("auto_approve_tools", []):
        result["auto_approve_tools"] = auto_approve_tools

    return result


# ===========================================================================
# Reflection node
# ===========================================================================


async def executor_reflect_node(state: ExecutorState) -> dict:
    """Quality gate: evaluate whether the executor's answer is adequate.

    Fires only when ``should_reflect`` returns True. Uses LITE tier for
    cost efficiency (~1-2s). On error, defaults to "pass" so reflection
    never blocks execution.
    """
    logger.info("executor_reflect_node_started")
    result = await reflect(state)

    reflection_count = state.get("reflection_count", 0) + 1
    event_list: list[dict] = [
        events.reasoning(
            thinking=(
                f"Reflection verdict: {result.verdict} "
                f"(confidence: {result.confidence:.2f}) — {result.note}"
            ),
            confidence=result.confidence,
            context="reflection",
        ),
    ]

    update: dict = {
        "reflection_count": reflection_count,
        "reflection_verdict": result.verdict,
        "confidence_score": result.confidence,
        "quality_assessment": result.note,
        "events": event_list,
    }

    if result.verdict == "retry":
        # Inject reflection feedback as a message so the LLM sees it
        feedback_msg = HumanMessage(
            content=(
                f"[Reflection feedback] Your previous response was evaluated as incomplete. "
                f"Note: {result.note}\n"
                f"Please review and improve your answer to fully address the step goal."
            ),
        )
        lc_messages = list(state.get("lc_messages", []))
        lc_messages.append(feedback_msg)
        update["lc_messages"] = lc_messages
        logger.info(
            "executor_reflect_retry",
            confidence=result.confidence,
            note=result.note,
        )

    return update


def executor_reflect_should_continue(
    state: ExecutorState,
) -> Literal["reason", "complete"]:
    """Route after reflection: retry goes back to reason, otherwise complete."""
    verdict = state.get("reflection_verdict")
    if verdict == "retry":
        return "reason"
    return "complete"


# ===========================================================================
# Graph construction
# ===========================================================================


def create_executor_graph():
    """Create the executor subgraph with explicit ReAct pattern.

    Graph structure:
    [reason] -> [act?] -> [wait_interrupt?] -> [reason] -> ...
             -> [reflect?] -> [reason] (retry) or [complete]
             -> [complete] -> END

    The reflect node is a lightweight quality gate that fires only when
    the LLM produces no tool calls and tools were actually used.
    """
    graph = StateGraph(ExecutorState)

    graph.add_node("reason", executor_reason_node)
    graph.add_node("act", executor_act_node)
    graph.add_node("wait_interrupt", executor_wait_interrupt_node)
    graph.add_node("reflect", executor_reflect_node)
    graph.add_node("complete", executor_complete_node)

    graph.set_entry_point("reason")

    graph.add_conditional_edges(
        "reason",
        executor_should_continue,
        {"act": "act", "reflect": "reflect", "complete": "complete"},
    )
    graph.add_conditional_edges(
        "act",
        executor_should_wait_or_reason,
        {"wait_interrupt": "wait_interrupt", "reason": "reason"},
    )
    graph.add_conditional_edges(
        "reflect",
        executor_reflect_should_continue,
        {"reason": "reason", "complete": "complete"},
    )
    graph.add_edge("wait_interrupt", "reason")
    graph.add_edge("complete", END)

    return graph.compile()


executor_subgraph = create_executor_graph()
