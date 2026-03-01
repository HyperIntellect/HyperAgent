"""Task subagent for general-purpose task handling with tool calling and handoff support."""

import json
import threading
import uuid
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph

from app.agents import events
from app.agents.context_compression import (
    CompressionConfig,
    ContextCompressor,
    inject_summary_as_context,
)
from app.agents.hitl.interrupt_manager import get_interrupt_manager
from app.agents.prompts import TASK_SYSTEM_MESSAGE, TASK_SYSTEM_PROMPT, get_task_system_prompt
from app.agents.state import TaskState
from app.agents.tools import (
    get_react_config,
    get_tools_for_agent,
)
from app.agents.tools.react_tool import (
    ErrorCategory,
    classify_error,
    deduplicate_tool_messages,
    truncate_messages_to_budget,
)
from app.agents.tools.tool_pipeline import (
    TaskToolHooks,
    ToolExecutionContext,
    execute_tool,
    execute_tools_batch,
)
from app.agents.utils import (
    append_history,
    build_image_context_message,
)
from app.ai.llm import extract_text_from_content, llm_service
from app.config import settings
from app.core.logging import get_logger
from app.guardrails.scanners.output_scanner import output_scanner

logger = get_logger(__name__)

# Module-level caches for tool lists and config (computed once, thread-safe)
_cached_task_tools: list | None = None
_cached_react_config = None
_cache_lock = threading.Lock()


def _get_cached_task_tools() -> list:
    """Get the task tools list, computing and caching on first call (thread-safe)."""
    global _cached_task_tools
    if _cached_task_tools is None:
        with _cache_lock:
            if _cached_task_tools is None:
                tools = get_tools_for_agent("task", include_handoffs=True)
                # Sort alphabetically for consistent KV-cache hits across requests
                _cached_task_tools = sorted(tools, key=lambda t: t.name)
    return _cached_task_tools


def _get_cached_react_config():
    """Get the react config for task, computing and caching on first call (thread-safe)."""
    global _cached_react_config
    if _cached_react_config is None:
        with _cache_lock:
            if _cached_react_config is None:
                _cached_react_config = get_react_config("task")
    return _cached_react_config


def clear_tool_cache() -> None:
    """Reset cached task tools and react config.

    Useful for testing or when tool registration changes at runtime.
    """
    global _cached_task_tools, _cached_react_config
    with _cache_lock:
        _cached_task_tools = None
        _cached_react_config = None


def _extract_task_plan_from_messages(
    tool_messages: list[ToolMessage],
) -> list[dict] | None:
    """Extract execution plan from task_planning skill output in tool messages.

    Scans tool messages for an invoke_skill result with skill_id="task_planning".
    If found and successful, returns the parsed plan steps list.

    Args:
        tool_messages: List of ToolMessage from the current act iteration

    Returns:
        List of plan step dicts if a plan was found, None otherwise
    """
    for msg in tool_messages:
        if not isinstance(msg, ToolMessage) or msg.name != "invoke_skill":
            continue
        try:
            parsed = json.loads(msg.content)
            if (
                parsed.get("skill_id") == "task_planning"
                and parsed.get("success")
            ):
                output = parsed.get("output", {})
                steps = output.get("steps", [])
                if steps and isinstance(steps, list):
                    logger.info(
                        "task_plan_extracted_from_skill_output",
                        step_count=len(steps),
                        complexity=output.get("complexity_assessment"),
                    )
                    return steps
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return None


def _build_todo_md(steps: list[dict], completed_up_to: int = 0) -> str:
    """Build a todo.md content from plan steps with completion status."""
    lines = ["# Task Plan\n"]
    for i, step in enumerate(steps):
        checkbox = "[x]" if i < completed_up_to else "[ ]"
        action = step.get("action", "Unknown step")
        tool_hint = step.get("tool_or_skill", "")
        lines.append(f"- {checkbox} Step {step.get('step_number', i + 1)}: {action}")
        if tool_hint:
            lines.append(f"  - Tool: {tool_hint}")
    return "\n".join(lines) + "\n"


_TODO_PATH = "/home/user/.hyperagent/todo.md"
_TODO_MAX_SIZE = 10 * 1024  # 10KB cap
_TODO_INJECT_CAP = 2000  # Max chars injected into prompt


def _count_todo_progress(content: str) -> tuple[int, int]:
    """Count checked and total checklist items in markdown content.

    Returns:
        Tuple of (checked, total)
    """
    checked = content.count("- [x]")
    unchecked = content.count("- [ ]")
    return checked, checked + unchecked


async def _read_sandbox_todo(user_id: str | None, task_id: str | None) -> str | None:
    """Read todo.md from sandbox, returning None if unavailable.

    Gracefully handles missing sandbox or file. Caps output at _TODO_MAX_SIZE.
    """
    try:
        from app.sandbox.execution_sandbox_manager import get_execution_sandbox_manager
        from app.sandbox.provider import is_provider_available

        available, _ = is_provider_available("execution")
        if not available:
            return None

        manager = get_execution_sandbox_manager()
        session = await manager.get_or_create_sandbox(user_id=user_id, task_id=task_id)
        runtime = session.executor

        if not runtime.sandbox_id:
            return None

        from app.sandbox import file_operations

        result = await file_operations.read_file(runtime, _TODO_PATH)
        if not result.get("success"):
            return None

        content = result.get("content", "")
        if len(content) > _TODO_MAX_SIZE:
            content = content[:_TODO_MAX_SIZE] + "\n... [truncated]"
        return content
    except Exception:
        return None


async def _write_sandbox_todo(
    user_id: str | None,
    task_id: str | None,
    content: str,
) -> bool:
    """Write todo.md to sandbox, returning True on success.

    Creates the .hyperagent directory if needed. Caps content at _TODO_MAX_SIZE.
    """
    try:
        from app.sandbox.execution_sandbox_manager import get_execution_sandbox_manager
        from app.sandbox.provider import is_provider_available

        available, _ = is_provider_available("execution")
        if not available:
            return False

        manager = get_execution_sandbox_manager()
        session = await manager.get_or_create_sandbox(user_id=user_id, task_id=task_id)
        runtime = session.executor

        if not runtime.sandbox_id:
            return False

        # Ensure directory exists
        await runtime.execute("mkdir -p /home/user/.hyperagent")

        # Cap content
        if len(content) > _TODO_MAX_SIZE:
            content = content[:_TODO_MAX_SIZE]

        from app.sandbox import file_operations

        result = await file_operations.write_file(runtime, _TODO_PATH, content)
        return result.get("success", False)
    except Exception:
        return False


async def reason_node(state: TaskState) -> dict:
    """ReAct reason node: LLM reasons about what to do next.

    Args:
        state: Current chat state

    Returns:
        Dict with updated messages and events
    """
    query = state.get("query") or ""
    mode = state.get("mode")
    locale = state.get("locale", "en")
    system_prompt = state.get("system_prompt") or get_task_system_prompt(locale)
    image_attachments = state.get("image_attachments") or []
    provider = state.get("provider")
    existing_summary = state.get("context_summary")
    # Create a copy to avoid in-place mutation issues
    lc_messages = list(state.get("lc_messages", []))

    event_list = []
    new_context_summary = None

    # Initialize messages if empty
    if not lc_messages:
        # Use pre-built cached message when using the default system prompt
        if system_prompt == TASK_SYSTEM_PROMPT:
            sys_msg = TASK_SYSTEM_MESSAGE
        else:
            sys_msg = SystemMessage(
                content=system_prompt,
                additional_kwargs={"cache_control": {"type": "ephemeral"}},
            )
        lc_messages = [sys_msg]

        # Inject persistent user memories if available (async DB-backed)
        user_id = state.get("user_id")
        if user_id:
            from app.services.memory_service import get_memory_store

            memory_text = await get_memory_store().format_memories_for_prompt_async(user_id)
            if memory_text:
                lc_messages.append(SystemMessage(content=memory_text))
                logger.info("user_memories_injected", user_id=user_id)

        # Add history if present
        history = state.get("messages", [])
        append_history(lc_messages, history)

        # Add multimodal image message if images are attached
        image_message = build_image_context_message(image_attachments)
        if image_message:
            lc_messages.append(image_message)
            logger.info("image_context_added_to_chat", image_count=len(image_attachments))

        # For dedicated modes (app, image, slide), directly synthesize a
        # tool call to invoke the corresponding skill.  This avoids relying
        # on the LLM to emit the correct invoke_skill call — some models
        # (especially with thinking/reasoning mode) return a text response
        # instead of using tools, which causes the mode to silently fail.
        _DIRECT_SKILL_MODES = {
            "image": {
                "skill_id": "image_generation",
                "param_key": "prompt",
            },
            "app": {
                "skill_id": "app_builder",
                "param_key": "description",
            },
            "slide": {
                "skill_id": "slide_generation",
                "param_key": "topic",
            },
            "data": {
                "skill_id": "data_analysis",
                "param_key": "query",
            },
        }

        skill_spec = _DIRECT_SKILL_MODES.get(mode)
        if skill_spec:
            lc_messages.append(HumanMessage(content=query))
            # Build a synthetic AI message with the invoke_skill tool call
            tool_call_id = f"direct_{skill_spec['skill_id']}_{uuid.uuid4().hex[:8]}"
            skill_params = {skill_spec["param_key"]: query}

            # Pass attachment_ids for data_analysis skill
            if skill_spec["skill_id"] == "data_analysis":
                attachment_ids = state.get("attachment_ids") or []
                if attachment_ids:
                    skill_params["attachment_ids"] = attachment_ids

            ai_message = AIMessage(
                content="",
                tool_calls=[{
                    "name": "invoke_skill",
                    "args": {
                        "skill_id": skill_spec["skill_id"],
                        "params": skill_params,
                    },
                    "id": tool_call_id,
                    "type": "tool_call",
                }],
            )
            lc_messages.append(ai_message)
            logger.info(
                "direct_skill_invocation",
                mode=mode,
                skill_id=skill_spec["skill_id"],
                original_query=query[:100],
            )
            return {
                "lc_messages": lc_messages,
                "events": event_list,
                "has_error": False,
            }
        else:
            # Add current query
            lc_messages.append(HumanMessage(content=query))

    # Debug: Log messages before deduplication
    logger.debug(
        "reason_node_messages",
        message_count=len(lc_messages),
        message_types=[type(m).__name__ for m in lc_messages],
        has_tool_messages=any(isinstance(m, ToolMessage) for m in lc_messages),
    )

    # Deduplicate tool messages to prevent API errors
    lc_messages = deduplicate_tool_messages(lc_messages)

    # Apply context compression before truncation (preserves semantic meaning)
    # Only create CompressionConfig/ContextCompressor if compression may actually run
    if settings.context_compression_enabled and len(lc_messages) > settings.context_compression_preserve_recent:
        # Quick token estimate to avoid creating compressor when clearly under threshold
        from app.agents.context_compression import estimate_message_tokens

        estimated_tokens = sum(estimate_message_tokens(m) for m in lc_messages)
        if existing_summary:
            from app.agents.context_compression import estimate_tokens
            estimated_tokens += estimate_tokens(existing_summary)

        if estimated_tokens > settings.context_compression_token_threshold:
            compression_config = CompressionConfig(
                token_threshold=settings.context_compression_token_threshold,
                preserve_recent=settings.context_compression_preserve_recent,
                enabled=settings.context_compression_enabled,
            )
            compressor = ContextCompressor(compression_config)

            try:
                new_summary, lc_messages = await compressor.compress(
                    lc_messages,
                    existing_summary,
                    provider,
                    locale=locale,
                )
                if new_summary:
                    # Inject summary as context after system message
                    lc_messages = inject_summary_as_context(lc_messages, new_summary)
                    new_context_summary = new_summary
                    logger.info("context_compressed", summary_length=len(new_summary))
                    event_list.append({
                        "type": "stage",
                        "name": "context",
                        "description": "Context compressed to preserve conversation history",
                        "status": "completed",
                    })
            except Exception as e:
                logger.warning("context_compression_skipped", error=str(e))

    # Apply message truncation to stay within token budget (fallback safety)
    config = _get_cached_react_config()
    lc_messages, was_truncated = truncate_messages_to_budget(
        lc_messages,
        max_tokens=config.max_message_tokens,
        preserve_recent=config.preserve_recent_messages,
    )
    if was_truncated:
        logger.info("task_messages_truncated_for_budget")
        event_list.append({
            "type": "stage",
            "name": "context",
            "description": "Message history truncated to fit context window",
            "status": "completed",
        })

    # Get LLM with tools
    tier = state.get("tier")
    model = state.get("model")
    llm = llm_service.choose_llm_for_task(
        task_type="task",
        provider=provider,
        tier_override=tier,
        model_override=model,
    )

    # Get all tools for chat agent (cached)
    all_tools = _get_cached_task_tools()
    llm_with_tools = llm.bind_tools(all_tools) if all_tools else llm

    # === KV-Cache Optimization: Stable Prefix Tracking ===
    # The stable prefix (system prompt + tool schemas) should not change
    # between iterations. Track its hash to detect accidental invalidation.
    from app.agents.context_compression import get_stable_prefix
    from app.agents.tools.registry import get_soft_disabled_message, get_soft_disabled_tools

    stable_prefix = get_stable_prefix(lc_messages)
    if stable_prefix:
        import hashlib

        prefix_content = "".join(
            str(m.content) for m in stable_prefix
        )
        current_prefix_hash = hashlib.md5(prefix_content.encode()).hexdigest()
        prev_prefix_hash = state.get("prefix_hash")
        if prev_prefix_hash and prev_prefix_hash != current_prefix_hash:
            logger.debug(
                "kv_cache_prefix_changed",
                prev_hash=prev_prefix_hash[:8],
                new_hash=current_prefix_hash[:8],
            )
        result_prefix_hash = current_prefix_hash
    else:
        result_prefix_hash = None

    # Inject soft-disabled tools notice at the end of the stable prefix
    # (preserves tool schemas in prefix for KV-cache while signaling unavailability)
    disabled_tools = get_soft_disabled_tools()
    if disabled_tools:
        disabled_msg = get_soft_disabled_message(disabled_tools)
        if disabled_msg:
            lc_messages.append(SystemMessage(content=disabled_msg))

    # === Context Engineering: Todo-list as Attention Manipulation ===
    # Inject the active todo/plan into messages after every tool call to keep
    # the global plan in the model's recent attention window. This prevents
    # the plan from getting "lost in the middle" across many tool iterations.
    messages_for_llm = lc_messages
    execution_plan = state.get("execution_plan", [])
    current_step_index = state.get("current_step_index", 0)
    active_todo = state.get("active_todo")

    if execution_plan and current_step_index < len(execution_plan):
        current_step = execution_plan[current_step_index]
        step_number = current_step.get("step_number", current_step_index + 1)
        total_steps = len(execution_plan)
        tool_hint = current_step.get("tool_or_skill") or "any appropriate tool"

        # Build full todo with progress for attention manipulation
        todo_lines = []
        for i, step in enumerate(execution_plan):
            status = "DONE" if i < current_step_index else (">>> CURRENT" if i == current_step_index else "pending")
            todo_lines.append(f"  {status}: Step {step.get('step_number', i+1)}: {step.get('action', '?')}")
        todo_block = "\n".join(todo_lines)

        step_guidance = SystemMessage(content=(
            f"[Plan Execution — Step {step_number} of {total_steps}]\n"
            f"Progress:\n{todo_block}\n\n"
            f"Current step: {current_step['action']}\n"
            f"Recommended tool: {tool_hint}\n"
            f"Focus on completing this specific step. "
            f"Do not repeat or re-invoke the task_planning skill."
        ))
        # Create a temporary copy with guidance appended — not persisted in state
        messages_for_llm = list(lc_messages) + [step_guidance]

        logger.info(
            "planned_execution_step_injected",
            step_number=step_number,
            total_steps=total_steps,
            action=current_step["action"][:100],
        )
    elif execution_plan and current_step_index >= len(execution_plan):
        # All plan steps completed — inject completion hint
        completion_hint = SystemMessage(content=(
            "[Plan Execution — All steps completed]\n"
            "The execution plan has been fully carried out. "
            "Provide a final summary of what was accomplished."
        ))
        messages_for_llm = list(lc_messages) + [completion_hint]
        logger.info("planned_execution_all_steps_completed")
    elif active_todo:
        # Non-planned execution but has an active todo from previous iterations.
        # Prefer sandbox-persisted todo.md for richer context, falling back
        # to in-state active_todo.
        todo_text = active_todo
        sandbox_todo = await _read_sandbox_todo(
            state.get("user_id"), state.get("task_id"),
        )
        if sandbox_todo:
            todo_text = sandbox_todo[:_TODO_INJECT_CAP]
            logger.debug("sandbox_todo_injected")
        todo_reminder = SystemMessage(content=(
            f"[Active Task Context]\n{todo_text}"
        ))
        messages_for_llm = list(lc_messages) + [todo_reminder]
        logger.debug("active_todo_injected")

    try:
        # Get AI response (may include tool calls)
        try:
            ai_message = await llm_with_tools.ainvoke(messages_for_llm)
        except Exception as invoke_err:
            # Providers with "thinking" mode (e.g. DeepSeek, Kimi) require
            # reasoning_content on every assistant message with tool_calls.
            # ChatOpenAI silently drops this field from responses, so the
            # ThinkingAwareChatOpenAI subclass patches the payload. If the
            # flag wasn't set upfront (e.g. enable_thinking not configured),
            # detect it here and enable the patch for all future calls.
            if "reasoning_content" in str(invoke_err):
                logger.info("thinking_mode_detected_enabling_patch")
                # Enable the payload-level patch on the underlying LLM client
                inner_llm = getattr(llm_with_tools, "bound", llm_with_tools)
                if hasattr(inner_llm, "thinking_mode"):
                    inner_llm.thinking_mode = True
                ai_message = await llm_with_tools.ainvoke(messages_for_llm)
            else:
                raise

        lc_messages.append(ai_message)

        # Stream tokens if no tool calls
        if not ai_message.tool_calls:
            response_text = extract_text_from_content(ai_message.content)
            if response_text:
                # Apply output guardrails
                scan_result = await output_scanner.scan(response_text, query)
                if scan_result.blocked:
                    logger.warning(
                        "output_guardrail_blocked",
                        violations=[v.value for v in scan_result.violations],
                        reason=scan_result.reason,
                    )
                    response_text = (
                        "I apologize, but I cannot provide that response. "
                        "Please ask a different question."
                    )
                elif scan_result.sanitized_content:
                    logger.info("output_guardrail_sanitized")
                    response_text = scan_result.sanitized_content

                event_list.append(events.token(response_text))

        result = {
            "lc_messages": lc_messages,
            "events": event_list,
            "has_error": False,
        }
        # Include context summary if compression occurred
        if new_context_summary:
            result["context_summary"] = new_context_summary
        # Track prefix hash for KV-cache stability monitoring
        if result_prefix_hash:
            result["prefix_hash"] = result_prefix_hash
        return result
    except Exception as e:
        logger.error("task_reason_failed", error=str(e))
        error_msg = str(e)

        # Provide user-friendly error messages
        if "prompt is too long" in error_msg:
            response = "I apologize, but the conversation has become too long. Please start a new conversation or try a simpler request."
        else:
            response = f"I apologize, but I encountered an error: {e}"

        event_list.append(events.token(response))

        return {
            "lc_messages": lc_messages,
            "response": response,
            "events": event_list,
            "has_error": True,  # Signal to stop the loop
        }


async def act_node(state: TaskState) -> dict:
    """ReAct act node: Execute tools based on LLM's tool calls.

    Includes Human-in-the-Loop (HITL) approval for high-risk tools.
    Delegates tool execution to the shared pipeline via execute_tools_batch.

    Args:
        state: Current chat state with tool calls

    Returns:
        Dict with tool results and events
    """
    from app.agents.hitl.tool_risk import requires_approval
    from app.config import settings

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
            "act_node_no_pending_tool_calls",
            existing_count=len(existing_tool_result_ids),
        )
        return {"events": event_list, "lc_messages": lc_messages}

    all_tools = _get_cached_task_tools()
    tool_map = {tool.name: tool for tool in all_tools}
    config = _get_cached_react_config()

    def hitl_check(tool_name: str) -> bool:
        # invoke_skill is always routed through HITL partitioning so per-skill
        # approval checks can run in TaskToolHooks.before_execution.
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

    result = {
        "lc_messages": lc_messages,
        "events": event_list,
        "tool_iterations": state.get("tool_iterations", 0) + (1 if tool_messages else 0),
    }

    if pending_interrupt:
        # Don't increment tool_iterations — the tool hasn't executed yet.
        result["pending_interrupt"] = pending_interrupt
        result["tool_iterations"] = state.get("tool_iterations", 0)

    # --- Self-correction: consecutive error tracking ---
    if tool_messages and not pending_interrupt:
        _ERROR_INDICATORS = ("Error:", "Failed:", "Error executing", "Traceback", "Exception:")
        has_errors = any(
            any(indicator in (msg.content or "") for indicator in _ERROR_INDICATORS)
            for msg in tool_messages
            if isinstance(msg, ToolMessage)
        )
        prev_consecutive = state.get("consecutive_errors", 0)
        if has_errors:
            # Classify the first error for category-specific recovery
            first_error = next(
                (msg.content for msg in tool_messages
                 if isinstance(msg, ToolMessage) and any(ind in (msg.content or "") for ind in _ERROR_INDICATORS)),
                "",
            )
            error_cat = classify_error(first_error or "")

            new_consecutive = prev_consecutive + 1
            result["consecutive_errors"] = new_consecutive
            if new_consecutive >= 3:
                # After 3 consecutive failures: comprehensive recovery
                recovery_msg = (
                    "CRITICAL: You have failed 3 consecutive tool calls. "
                    "You MUST change your approach:\n"
                    f"- Error category: {error_cat.value}\n"
                    f"- Last error: {first_error[:300] if first_error else 'unknown'}\n\n"
                    "Recovery strategies (pick one):\n"
                    "1. Use a completely different tool or method\n"
                    "2. Simplify the task — break it into smaller sub-steps\n"
                    "3. Use ask_user to get clarification or help\n"
                    "4. If you have a plan, invoke task_planning with mode='revise' "
                    "to create a new plan that works around the failures\n"
                    "5. Skip this step and move on if possible\n\n"
                    "Do NOT retry the same failing approach."
                )
                lc_messages.append(SystemMessage(content=recovery_msg))
                event_list.append(events.reasoning(
                    thinking=f"3 consecutive errors ({error_cat.value}). Forcing strategy change.",
                    confidence=0.2,
                    context="error_recovery",
                ))
                logger.warning("consecutive_errors_threshold", count=new_consecutive, category=error_cat.value)
            else:
                _RECOVERY_HINTS = {
                    ErrorCategory.TRANSIENT: (
                        "This appears to be a temporary issue (network, timeout, rate limit). "
                        "Wait a moment and retry the same call."
                    ),
                    ErrorCategory.INPUT: (
                        "The tool inputs are incorrect. Carefully review the error message, "
                        "fix the specific parameter that caused the error, and retry with corrected values. "
                        "Common fixes: check file paths, validate JSON format, verify parameter types."
                    ),
                    ErrorCategory.PERMISSION: (
                        "Access was denied. Options: (1) use ask_user to request help with permissions, "
                        "(2) try an alternative approach that doesn't require elevated access, "
                        "(3) check if there's a different path or resource you can use."
                    ),
                    ErrorCategory.RESOURCE: (
                        "The resource was not found. Options: (1) search for the correct path/URL using "
                        "file_find_by_name or web_search, (2) try an alternative resource, "
                        "(3) create the missing resource if appropriate."
                    ),
                    ErrorCategory.FATAL: (
                        "This is a critical error that cannot be retried. Stop this approach entirely "
                        "and try a completely different strategy. Consider using ask_user for guidance."
                    ),
                }
                hint = _RECOVERY_HINTS.get(error_cat, "The previous tool call failed. Analyze the error and try a different approach.")
                lc_messages.append(SystemMessage(content=f"Tool error ({error_cat.value}): {hint}"))
                event_list.append(events.reasoning(
                    thinking=f"Tool error classified as {error_cat.value}. Recovery: {hint}",
                    context="error_recovery",
                ))
                logger.info("consecutive_error_detected", count=new_consecutive, category=error_cat.value)
        else:
            if prev_consecutive > 0:
                logger.info("consecutive_errors_reset", previous_count=prev_consecutive)
            result["consecutive_errors"] = 0

        # --- Plan revision on step failure ---
        execution_plan_for_revision = state.get("execution_plan", [])
        current_idx_for_revision = state.get("current_step_index", 0)
        if has_errors and execution_plan_for_revision and current_idx_for_revision < len(execution_plan_for_revision):
            prev_revisions = state.get("plan_revision_count", 0)
            if prev_revisions < 2:  # Max 2 revisions
                lc_messages.append(SystemMessage(content=(
                    f"Step {current_idx_for_revision + 1} failed. You may revise the remaining plan steps. "
                    f"Consider: (1) retrying with a different approach, (2) skipping this step, "
                    f"or (3) modifying the remaining steps to work around the failure."
                )))
                result["plan_revision_count"] = prev_revisions + 1
                logger.info(
                    "plan_revision_prompted",
                    step=current_idx_for_revision + 1,
                    revision_count=prev_revisions + 1,
                )

    # --- Anti-Repetition: Detect consecutive identical tool calls ---
    if tool_messages and not pending_interrupt:
        import hashlib as _hashlib

        prev_hashes = list(state.get("last_tool_calls_hash", []) or [])

        # Compute hash for each tool call in the current batch
        for tc in pending_tool_calls:
            tc_name = tc.get("name", "")
            tc_args = tc.get("args", {})
            # Sort args for deterministic hashing
            try:
                args_str = json.dumps(tc_args, sort_keys=True, default=str)
            except (TypeError, ValueError):
                args_str = str(tc_args)
            call_hash = _hashlib.md5(f"{tc_name}:{args_str}".encode()).hexdigest()[:12]
            prev_hashes.append(call_hash)

        # Keep only last 5 hashes
        if len(prev_hashes) > 5:
            prev_hashes = prev_hashes[-5:]
        result["last_tool_calls_hash"] = prev_hashes

        # Count consecutive identical hashes from the end
        if len(prev_hashes) >= 2:
            last_hash = prev_hashes[-1]
            consecutive_count = 0
            for h in reversed(prev_hashes):
                if h == last_hash:
                    consecutive_count += 1
                else:
                    break

            if consecutive_count >= 3:
                # Force plan revision after 3+ repeats
                prev_revisions = state.get("plan_revision_count", 0)
                lc_messages.append(SystemMessage(content=(
                    f"[System: Repetition detected - you have called the same tool with identical "
                    f"arguments {consecutive_count} times consecutively. "
                    f"The previous approach is NOT working. You MUST change strategy:\n"
                    f"- Use a DIFFERENT tool\n"
                    f"- Modify the arguments significantly\n"
                    f"- Break the problem into smaller steps\n"
                    f"- Ask the user for clarification if stuck\n"
                    f"Do NOT retry the same call again."
                )))
                result["plan_revision_count"] = prev_revisions + 1
                event_list.append(events.reasoning(
                    thinking=f"Tool call repeated {consecutive_count}x. Forcing strategy change.",
                    confidence=0.1,
                    context="error_recovery",
                ))
                logger.warning(
                    "anti_repetition_force_revision",
                    consecutive_count=consecutive_count,
                    hash=last_hash,
                )
            elif consecutive_count >= 2:
                # Inject variation prompt after 2 repeats
                repeated_tool = pending_tool_calls[-1].get("name", "unknown") if pending_tool_calls else "unknown"
                lc_messages.append(SystemMessage(content=(
                    f"[System: Repetition detected - you have called {repeated_tool} with identical "
                    f"arguments {consecutive_count} times. "
                    f"The previous approach may not be working. Try a DIFFERENT strategy:\n"
                    f"- Use a different tool\n"
                    f"- Modify the arguments significantly\n"
                    f"- Break the problem into smaller steps\n"
                    f"- Ask the user for clarification if stuck"
                )))
                event_list.append(events.reasoning(
                    thinking=f"Tool call repeated {consecutive_count}x. Suggesting variation.",
                    confidence=0.3,
                    context="error_recovery",
                ))
                logger.info(
                    "anti_repetition_variation_prompt",
                    consecutive_count=consecutive_count,
                    tool=repeated_tool,
                )

    # --- Context Engineering: Update active_todo for attention manipulation ---
    if tool_messages and not pending_interrupt:
        # Build a summary of what was just done for the active todo
        tool_names = [msg.name for msg in tool_messages if isinstance(msg, ToolMessage) and msg.name]
        iteration_count = state.get("tool_iterations", 0) + 1
        if tool_names:
            active_todo_parts = [f"Iteration {iteration_count}: Used {', '.join(tool_names)}"]
            # Include execution plan progress if available
            exec_plan = state.get("execution_plan", [])
            cur_idx = state.get("current_step_index", 0)
            if exec_plan:
                completed = min(cur_idx + 1, len(exec_plan))
                active_todo_parts.append(f"Plan progress: {completed}/{len(exec_plan)} steps")
            result["active_todo"] = " | ".join(active_todo_parts)

    # --- Planned execution mode ---

    # 1. Check if task_planning skill just returned a plan → parse into state
    if not state.get("execution_plan"):
        parsed_plan = _extract_task_plan_from_messages(tool_messages)
        if parsed_plan:
            result["execution_plan"] = parsed_plan
            result["current_step_index"] = 0
            logger.info(
                "execution_plan_parsed",
                step_count=len(parsed_plan),
            )
            # Write initial todo.md to sandbox (persisted)
            todo_content = _build_todo_md(parsed_plan)
            wrote = await _write_sandbox_todo(
                state.get("user_id"), state.get("task_id"), todo_content,
            )
            if wrote:
                checked, total = _count_todo_progress(todo_content)
                event_list.append(events.todo_update(
                    content=todo_content, checked=checked, total=total,
                ))
                logger.debug("sandbox_todo_written", path=_TODO_PATH)
            # Emit plan_overview event with all steps for the frontend progress view
            event_list.append(events.plan_overview(
                steps=[
                    {
                        "id": i,
                        "title": step.get("action", f"Step {i + 1}"),
                        "description": step.get("tool_or_skill", ""),
                        "status": "pending",
                    }
                    for i, step in enumerate(parsed_plan)
                ],
                total_steps=len(parsed_plan),
                completed_steps=0,
            ))
            # Emit plan_step event for the first step
            first_step = parsed_plan[0]
            event_list.append(events.plan_step(
                step_number=first_step.get("step_number", 1),
                total_steps=len(parsed_plan),
                action=first_step.get("action", ""),
                status="running",
            ))

    # 2. If already in planned execution, advance step and emit progress events
    elif state.get("execution_plan") and tool_messages and not pending_interrupt:
        execution_plan = state["execution_plan"]
        current_idx = state.get("current_step_index", 0)

        if current_idx < len(execution_plan):
            # Mark current step completed
            completed_step = execution_plan[current_idx]
            event_list.append(events.plan_step(
                step_number=completed_step.get("step_number", current_idx + 1),
                total_steps=len(execution_plan),
                action=completed_step.get("action", ""),
                status="completed",
            ))
            # Emit plan_step_completed for frontend progress view
            event_list.append(events.plan_step_completed(
                step_id=current_idx,
                status="completed",
                completed_steps=current_idx + 1,
                total_steps=len(execution_plan),
            ))

            # Record step result for verification
            step_summary = " | ".join(
                (msg.content or "")[:200] for msg in tool_messages if isinstance(msg, ToolMessage)
            )
            result["completed_step_results"] = [{
                "step_number": completed_step.get("step_number", current_idx + 1),
                "action": completed_step.get("action", ""),
                "result_summary": step_summary[:500],
            }]

            # Advance to next step
            next_idx = current_idx + 1
            result["current_step_index"] = next_idx

            # Update todo.md with completed step (persisted)
            todo_content = _build_todo_md(execution_plan, completed_up_to=next_idx)
            wrote = await _write_sandbox_todo(
                state.get("user_id"), state.get("task_id"), todo_content,
            )
            if wrote:
                checked, total = _count_todo_progress(todo_content)
                event_list.append(events.todo_update(
                    content=todo_content, checked=checked, total=total,
                ))
                logger.debug("sandbox_todo_updated", step=next_idx)

            if next_idx < len(execution_plan):
                # Emit running event for next step
                next_step = execution_plan[next_idx]
                event_list.append(events.plan_step(
                    step_number=next_step.get("step_number", next_idx + 1),
                    total_steps=len(execution_plan),
                    action=next_step.get("action", ""),
                    status="running",
                ))
                logger.info(
                    "planned_execution_step_advanced",
                    completed_step=current_idx + 1,
                    next_step=next_idx + 1,
                    total_steps=len(execution_plan),
                )
            else:
                logger.info(
                    "planned_execution_plan_completed",
                    total_steps=len(execution_plan),
                )

    return result


async def verify_node(state: TaskState) -> dict:
    """Verify planned execution results against the original goal.

    Enhanced verification that:
    1. Checks each step's results against expected outcomes
    2. For code tasks: checks for errors in tool outputs
    3. For file tasks: verifies file references in results
    4. For app tasks: checks for build/server success indicators
    5. On failure: provides actionable feedback for self-correction

    Only runs when there is an execution_plan. Uses FLASH tier LLM
    for cost efficiency.

    Args:
        state: Current task state with completed_step_results

    Returns:
        Dict with verification events and results
    """
    from app.models.schemas import ModelTier

    execution_plan = state.get("execution_plan", [])
    if not execution_plan:
        return {"events": []}

    original_query = state.get("query", "")
    completed_results = state.get("completed_step_results", [])
    provider = state.get("provider")

    event_list = []
    verification_results = []
    verified_steps = []
    verification_status = "error"
    verification_retry_hint: str | None = None

    # --- Phase 1: Automated checks on tool outputs ---
    _ERROR_PATTERNS = ("Error:", "Failed:", "Traceback", "Exception:", "error:", "FAIL")
    _SUCCESS_PATTERNS = ("Successfully", "Created", "Written", "Deployed", "Running on", "Build succeeded")

    for step_result in completed_results:
        step_num = step_result.get("step_number", 0)
        result_summary = step_result.get("result_summary", "")
        action = step_result.get("action", "")

        # Auto-check: look for error indicators in tool outputs
        has_errors = any(pat in result_summary for pat in _ERROR_PATTERNS)
        has_success = any(pat in result_summary for pat in _SUCCESS_PATTERNS)

        step_verification = {
            "step_number": step_num,
            "action": action,
            "auto_check": "fail" if has_errors else ("pass" if has_success else "unknown"),
            "has_errors": has_errors,
        }

        if has_errors:
            finding = f"Step {step_num} output contains error indicators."
            event_list.append(events.verification(
                status="warning",
                message=f"Step {step_num} ({action}) had errors in output",
                step=step_num,
                findings=[finding],
                retry_hint="Review failed output and adjust tool/arguments before retrying.",
            ))
        elif has_success:
            verified_steps.append(step_num)

        verification_results.append(step_verification)

    # --- Phase 2: LLM-based holistic verification ---
    if completed_results:
        steps_summary = "\n".join(
            f"- Step {r.get('step_number', '?')}: {r.get('action', 'N/A')} -> {r.get('result_summary', 'No result')[:200]}"
            for r in completed_results
        )
    else:
        steps_summary = "(No step results recorded)"

    # Include auto-check results in the verification prompt
    auto_check_summary = "\n".join(
        f"  Step {v['step_number']}: auto-check={'PASS' if v['auto_check'] == 'pass' else 'FAIL' if v['auto_check'] == 'fail' else 'UNKNOWN'}"
        for v in verification_results
    )

    verification_prompt = (
        f"Review the completed task execution:\n\n"
        f"Original goal: {original_query}\n\n"
        f"Steps completed:\n{steps_summary}\n\n"
        f"Automated checks:\n{auto_check_summary}\n\n"
        f"Evaluate whether the execution achieved the original goal.\n"
        f"Consider:\n"
        f"1. Were all required steps completed successfully?\n"
        f"2. Do the outputs match what was expected?\n"
        f"3. Are there any missing deliverables?\n\n"
        f"Respond with:\n"
        f"- PASS: if the goal was achieved (brief explanation)\n"
        f"- PARTIAL: if some goals were met but others remain (list what's missing)\n"
        f"- FAIL: with explanation of what went wrong and suggested fixes"
    )

    try:
        llm = llm_service.get_llm_for_tier(ModelTier.FLASH, provider=provider)
        result = await llm.ainvoke([HumanMessage(content=verification_prompt)])
        verification_text = extract_text_from_content(result.content)

        verdict = verification_text.strip().split(":")[0].upper().strip()
        if verdict.startswith("PASS"):
            verification_status = "passed"
            event_list.append(events.verification(
                status="passed",
                message=verification_text,
                findings=["Verifier confirmed all required outcomes were satisfied."],
            ))
            logger.info("verification_passed", query=original_query[:100])
        elif verdict.startswith("PARTIAL"):
            verification_status = "partial"
            verification_retry_hint = "Address missing deliverables and regenerate an updated plan."
            event_list.append(events.verification(
                status="partial",
                message=verification_text,
                findings=["Verifier found partially completed objectives."],
                retry_hint=verification_retry_hint,
            ))
            logger.info("verification_partial", query=original_query[:100], detail=verification_text[:200])
        else:
            verification_status = "failed"
            verification_retry_hint = "Revise plan and use an alternative approach for failed steps."
            event_list.append(events.verification(
                status="failed",
                message=verification_text,
                findings=["Verifier determined the goal was not achieved."],
                retry_hint=verification_retry_hint,
            ))
            logger.warning("verification_failed", query=original_query[:100], detail=verification_text[:200])
    except Exception as e:
        logger.error("verification_node_error", error=str(e))
        verification_status = "error"
        verification_retry_hint = "Verification failed due to model error. Retry with a simplified plan."
        event_list.append(events.verification(
            status="error",
            message=f"Verification could not be completed: {e}",
            findings=["Verifier runtime error."],
            retry_hint=verification_retry_hint,
        ))

    prev_attempts = int(state.get("verification_attempts", 0))
    should_adapt = verification_status in {"partial", "failed", "error"}
    next_attempts = prev_attempts + 1 if should_adapt else prev_attempts

    result: dict = {
        "events": event_list,
        "verification_results": verification_results,
        "verified_steps": verified_steps,
        "verification_status": verification_status,
        "verification_attempts": next_attempts,
        "verification_retry_hint": verification_retry_hint,
    }

    if should_adapt:
        retry_hint = verification_retry_hint or "Revise plan and retry with a different approach."
        lc_messages = list(state.get("lc_messages", []))
        lc_messages.append(SystemMessage(content=(
            "Verification gate failed. You must adapt your plan before proceeding.\n"
            f"Retry hint: {retry_hint}\n"
            "Actions required:\n"
            "1. Re-plan remaining work (use task_planning mode='revise' if useful).\n"
            "2. Avoid repeating the same failed tool sequence.\n"
            "3. Continue only after producing new actionable steps."
        )))
        result["lc_messages"] = lc_messages
        result["execution_plan"] = []
        result["current_step_index"] = 0
        result["tool_iterations"] = 0

    return result


def should_adapt_or_finalize(state: TaskState) -> Literal["reason", "finalize"]:
    """Route post-verification flow to adaptation or finalization."""
    status = (state.get("verification_status") or "").lower()
    if status == "passed":
        return "finalize"

    attempts = int(state.get("verification_attempts", 0))
    execution_mode = (state.get("execution_mode") or "auto").lower()
    max_attempts = 1 if execution_mode == "strict" else 2
    if attempts >= max_attempts:
        logger.warning(
            "verification_attempt_limit_reached",
            status=status,
            attempts=attempts,
            max_attempts=max_attempts,
            execution_mode=execution_mode,
        )
        return "finalize"

    logger.info(
        "verification_routing_to_adaptation",
        status=status,
        attempts=attempts,
        max_attempts=max_attempts,
    )
    return "reason"


async def finalize_node(state: TaskState) -> dict:
    """Extract final response from messages and prepare output.

    Args:
        state: Current chat state

    Returns:
        Dict with final response and events (only new events; state.events uses operator.add)
    """
    lc_messages = state.get("lc_messages", [])

    # Extract final response from last AI message (prefer one without tool_calls)
    response = ""
    for msg in reversed(lc_messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            response = extract_text_from_content(msg.content)
            break

    # Fallback: if every AIMessage had tool_calls (e.g., loop hit max_iterations),
    # extract text from the last AIMessage even though it has tool_calls
    if not response:
        for msg in reversed(lc_messages):
            if isinstance(msg, AIMessage):
                text = extract_text_from_content(msg.content)
                if text:
                    response = text
                    break

    # Last resort: generate a generic message so the user isn't left with silence
    if not response:
        response = "I was unable to complete the task within the allowed number of steps. Please try rephrasing your request or breaking it into smaller parts."

    # Emit only the new stage event; state.events uses operator.add so we must not return full list
    final_event = events.stage("task", "Response generated", "completed")

    return {
        "response": response,
        "events": [final_event],
    }


def should_continue(state: TaskState) -> Literal["act", "verify", "finalize"]:
    """Decide whether to continue ReAct loop, verify, or finalize.

    Args:
        state: Current chat state

    Returns:
        "act" if tools were called, "verify" if plan completed and needs verification,
        "finalize" otherwise
    """
    # If there was an error, stop immediately
    if state.get("has_error"):
        logger.info("task_stopping_due_to_error")
        return "finalize"

    # If response is already set (from error handling), stop
    if state.get("response"):
        return "finalize"

    lc_messages = state.get("lc_messages", [])

    # Log plan progress if in planned execution mode
    execution_plan = state.get("execution_plan", [])
    if execution_plan:
        current_idx = state.get("current_step_index", 0)
        total = len(execution_plan)
        logger.info(
            "planned_execution_progress",
            current_step=current_idx + 1,
            total_steps=total,
            completed=current_idx >= total,
        )

    # Check if last message has tool calls
    for msg in reversed(lc_messages):
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                # Check iteration limit
                config = _get_cached_react_config()
                iters = state.get("tool_iterations", 0)
                if iters >= config.max_iterations:
                    logger.warning(
                        "task_max_iterations_reached",
                        iterations=iters,
                    )
                    # Even at max iterations, verify if plan was active
                    if execution_plan:
                        return "verify"
                    return "finalize"
                return "act"
            else:
                # No tool calls — if planned execution completed, verify first
                if execution_plan and state.get("current_step_index", 0) >= len(execution_plan):
                    return "verify"
                return "finalize"

    return "finalize"


def should_wait_or_reason(state: TaskState) -> Literal["wait_interrupt", "reason"]:
    """Decide whether to wait for interrupt response or continue reasoning.

    Args:
        state: Current chat state

    Returns:
        "wait_interrupt" if pending interrupt exists, "reason" otherwise
    """
    pending = state.get("pending_interrupt")
    if pending:
        logger.info(
            "routing_to_wait_interrupt",
            interrupt_id=pending.get("interrupt_id"),
            thread_id=pending.get("thread_id"),
        )
        return "wait_interrupt"
    logger.debug("routing_to_reason", has_pending=False)
    return "reason"


async def wait_interrupt_node(state: TaskState) -> dict:
    """Wait for user response to a pending interrupt and add result.

    This node handles the async wait for user input when the agent
    has asked a question via ask_user tool.

    Args:
        state: Current chat state with pending_interrupt

    Returns:
        Dict with tool result message and cleared pending_interrupt
    """
    logger.info("wait_interrupt_node_started")

    pending = state.get("pending_interrupt")
    if not pending:
        logger.warning("wait_interrupt_node_no_pending")
        return {"pending_interrupt": None}

    interrupt_id = pending.get("interrupt_id")
    thread_id = pending.get("thread_id", "default")
    tool_call_id = pending.get("tool_call_id")
    tool_name = pending.get("tool_name", "ask_user")

    interrupt_manager = get_interrupt_manager()
    event_list = []
    lc_messages = list(state.get("lc_messages", []))

    # Debug: Log the messages before adding tool result
    logger.debug(
        "wait_interrupt_messages_before",
        message_count=len(lc_messages),
        message_types=[type(m).__name__ for m in lc_messages],
    )

    logger.info(
        "wait_interrupt_subscribing",
        interrupt_id=interrupt_id,
        thread_id=thread_id,
        tool_call_id=tool_call_id,
    )

    # Get tools for potential execution (needed for approval flow) - cached
    all_tools = _get_cached_task_tools()
    tool_map = {tool.name: tool for tool in all_tools}

    # Track auto_approve updates
    auto_approve_tools = list(state.get("auto_approve_tools", []))

    try:
        # Wait for user response
        response = await interrupt_manager.wait_for_response(
            thread_id=thread_id,
            interrupt_id=interrupt_id,
            timeout_seconds=settings.hitl_decision_timeout,
        )

        action = response.get("action", "skip")
        value = response.get("value")

        logger.info(
            "wait_interrupt_response_received",
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

                config = _get_cached_react_config()
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
                # Confirm type sends approve/deny
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

        # Debug: Log after adding tool result
        logger.debug(
            "wait_interrupt_messages_after",
            message_count=len(lc_messages),
            result_str=result_str,
            tool_call_id=tool_call_id,
        )

    except TimeoutError:
        logger.warning("wait_interrupt_timeout", interrupt_id=interrupt_id)
        result_str = "timeout"
        lc_messages.append(
            ToolMessage(
                content=result_str,
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

    except Exception as e:
        logger.error("wait_interrupt_error", error=str(e), interrupt_id=interrupt_id)
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


def create_task_graph() -> StateGraph:
    """Create the task subagent graph with explicit ReAct pattern.

    Graph structure:
    [reason] -> [act?] -> [wait_interrupt?] -> [reason] -> ... -> [verify?] -> [finalize] -> END

    The ReAct loop:
    1. reason: LLM reasons and may call tools
    2. act: Execute tools if called (may set pending_interrupt for ask_user)
    3. wait_interrupt: If pending_interrupt, wait for user response
    4. Loop back to reason with tool results
    5. verify: If planned execution completed, verify results
    6. End when no more tool calls

    Returns:
        Compiled task graph
    """
    graph = StateGraph(TaskState)

    # Add ReAct nodes
    graph.add_node("reason", reason_node)
    graph.add_node("act", act_node)
    graph.add_node("wait_interrupt", wait_interrupt_node)
    graph.add_node("verify", verify_node)
    graph.add_node("finalize", finalize_node)

    # Set entry point
    graph.set_entry_point("reason")

    # ReAct loop: reason -> act (if tools called) -> reason -> ... -> verify/finalize
    graph.add_conditional_edges(
        "reason",
        should_continue,
        {
            "act": "act",
            "verify": "verify",
            "finalize": "finalize",
        },
    )

    # After acting, check if we need to wait for interrupt or continue reasoning
    graph.add_conditional_edges(
        "act",
        should_wait_or_reason,
        {
            "wait_interrupt": "wait_interrupt",
            "reason": "reason",
        },
    )

    # After waiting for interrupt, go back to reasoning
    graph.add_edge("wait_interrupt", "reason")

    # Verify can route to adaptation loop or finalize
    graph.add_conditional_edges(
        "verify",
        should_adapt_or_finalize,
        {
            "reason": "reason",
            "finalize": "finalize",
        },
    )

    # Finalize and end
    graph.add_edge("finalize", END)

    return graph.compile()


# Compiled subgraph for use by supervisor
task_subgraph = create_task_graph()
