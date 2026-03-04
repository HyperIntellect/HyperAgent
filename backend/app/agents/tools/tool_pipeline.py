"""Unified tool execution pipeline with hooks for agent-specific behavior.

This module provides a shared tool execution infrastructure that all agents
(task, research, canonical loop) use. Each agent plugs in its own hooks for
HITL, event extraction, guardrails, etc., while sharing the core execution
pipeline (context injection, retry, truncation).
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from app.agents.tools.context_injection import inject_tool_context
from app.agents.tools.react_tool import (
    ErrorCategory,
    ReActLoopConfig,
    classify_error,
    execute_tool_with_retry,
    truncate_tool_result,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


# Fields injected by the pipeline (not provided by the LLM)
_INJECTED_FIELDS = {"user_id", "task_id"}


def _validate_required_args(tool: BaseTool, args: dict) -> list[str]:
    """Check if required args (excluding injected fields) are present.

    Uses the tool's args_schema (Pydantic model) to identify required fields.

    Returns:
        List of missing required field names, empty if all present.
    """
    schema: type[BaseModel] | None = getattr(tool, "args_schema", None)
    if schema is None:
        return []

    missing = []
    for field_name, field_info in schema.model_fields.items():
        if field_name in _INJECTED_FIELDS:
            continue
        if field_info.is_required() and (
            field_name not in args or args[field_name] is None
        ):
            missing.append(field_name)

    return missing


# Tools that require sequential execution due to side effects
SEQUENTIAL_TOOLS = {
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_scroll",
    "browser_press_key",
}


@dataclass
class ToolExecutionContext:
    """Context passed to hook methods for a single tool execution."""

    tool_name: str
    tool_args: dict
    tool_call_id: str
    tool: BaseTool | None  # Resolved tool from tool_map
    user_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    policy_decision: str | None = None
    policy_reason: str | None = None


@dataclass
class ToolExecutionResult:
    """Result from executing a single tool through the pipeline."""

    message: ToolMessage | None  # None if interrupted (HITL)
    events: list[dict] = field(default_factory=list)
    is_error: bool = False
    pending_interrupt: dict | None = None
    error_category: ErrorCategory | None = None


class ToolExecutionHooks:
    """Base hook class — override points for agent-specific behavior.

    Default implementations are no-ops, allowing agents to override only
    the hooks they need.
    """

    skip_before_execution: bool = False

    async def before_execution(
        self, ctx: ToolExecutionContext
    ) -> ToolExecutionResult | None:
        """Called before tool execution. Return non-None to short-circuit.

        Use cases: HITL approval check, ask_user handling, guardrail scan.
        """
        return None

    def after_execution(
        self, ctx: ToolExecutionContext, result_str: str, events: list[dict]
    ) -> str:
        """Post-process result string before truncation.

        Use cases: event extraction (images, slides, app builder), source parsing.
        Returns the (possibly modified) result string.
        """
        return result_str

    def on_tool_call(
        self, tool_name: str, tool_args: dict, tool_call_id: str
    ) -> list[dict]:
        """Emit events when a tool is called. Returns events to add."""
        return []

    def on_tool_result(
        self, tool_name: str, result: str, tool_call_id: str
    ) -> list[dict]:
        """Emit events when a tool returns. Returns events to add."""
        return []


async def execute_tool(
    ctx: ToolExecutionContext,
    hooks: ToolExecutionHooks,
    config: ReActLoopConfig,
    use_retry: bool = False,
) -> ToolExecutionResult:
    """Execute a single tool through the unified pipeline.

    Pipeline steps:
    1. hooks.before_execution() — can short-circuit (HITL, guardrails)
    2. inject_tool_context() — add user_id/task_id to tool args
    3. hooks.on_tool_call() — emit tool-call event
    4. Execute tool (with or without retry)
    5. hooks.after_execution() — event extraction, source parsing
    6. truncate_tool_result() — truncate to config max
    7. hooks.on_tool_result() — emit tool-result event

    Args:
        ctx: Tool execution context
        hooks: Agent-specific hook implementations
        config: ReAct loop config (for truncation, retry settings)
        use_retry: Whether to use retry logic (canonical loop only)

    Returns:
        ToolExecutionResult with message, events, error status
    """
    all_events: list[dict] = []
    final_status = "failed"
    final_error: str | None = None
    final_result_summary: str | None = None

    # 1. Before-execution hook (HITL, guardrails, ask_user)
    if not hooks.skip_before_execution:
        short_circuit = await hooks.before_execution(ctx)
        if short_circuit is not None:
            if short_circuit.pending_interrupt:
                final_status = "pending_approval"
            elif short_circuit.is_error:
                final_status = "failed"
                final_error = short_circuit.message.content if short_circuit.message else None
            else:
                final_status = "completed"
            return short_circuit

    # Persist tool execution start in run ledger if available.
    if ctx.run_id and ctx.tool_call_id:
        try:
            from app.services.run_ledger import run_ledger_service

            await run_ledger_service.record_tool_execution_start(
                run_id=ctx.run_id,
                step_id=ctx.tool_call_id,
                tool_call_id=ctx.tool_call_id,
                tool_name=ctx.tool_name,
                tool_args=ctx.tool_args,
                policy_decision=ctx.policy_decision,
                policy_reason=ctx.policy_reason,
            )
        except Exception as e:
            logger.warning("tool_execution_start_persist_failed", error=str(e))

    # 2. Inject context (user_id / task_id)
    inject_tool_context(ctx.tool_name, ctx.tool_args, ctx.user_id, ctx.task_id)

    # 3. Emit tool-call event
    call_events = hooks.on_tool_call(ctx.tool_name, ctx.tool_args, ctx.tool_call_id)
    if ctx.policy_decision:
        for evt in call_events:
            if evt.get("type") == "tool_call":
                evt["policy_decision"] = ctx.policy_decision
                if ctx.policy_reason:
                    evt["policy_reason"] = ctx.policy_reason
    all_events.extend(call_events)

    # 4. Tool not found
    if ctx.tool is None:
        error_msg = f"Tool not found: {ctx.tool_name}"
        final_status = "failed"
        final_error = error_msg
        result_events = hooks.on_tool_result(
            ctx.tool_name, error_msg, ctx.tool_call_id
        )
        if ctx.policy_decision:
            for evt in result_events:
                if evt.get("type") == "tool_result":
                    evt["policy_decision"] = ctx.policy_decision
                    if ctx.policy_reason:
                        evt["policy_reason"] = ctx.policy_reason
        all_events.extend(result_events)
        return ToolExecutionResult(
            message=ToolMessage(
                content=error_msg,
                tool_call_id=ctx.tool_call_id,
                name=ctx.tool_name,
            ),
            events=all_events,
            is_error=True,
            error_category=ErrorCategory.RESOURCE,
        )

    # 4b. Pre-validate required args before invocation
    missing_args = _validate_required_args(ctx.tool, ctx.tool_args)
    if missing_args:
        error_msg = (
            f"Tool '{ctx.tool_name}' requires the following argument(s) "
            f"that were not provided: {', '.join(missing_args)}. "
            f"Please call this tool again with all required arguments."
        )
        logger.warning(
            "tool_args_validation_failed",
            tool=ctx.tool_name,
            missing_args=missing_args,
            provided_args=list(ctx.tool_args.keys()),
        )
        final_status = "failed"
        final_error = error_msg
        result_events = hooks.on_tool_result(
            ctx.tool_name, error_msg, ctx.tool_call_id
        )
        all_events.extend(result_events)
        return ToolExecutionResult(
            message=ToolMessage(
                content=error_msg,
                tool_call_id=ctx.tool_call_id,
                name=ctx.tool_name,
            ),
            events=all_events,
            is_error=True,
            error_category=ErrorCategory.INPUT,
        )

    # 5. Execute tool
    try:
        if use_retry:
            result = await execute_tool_with_retry(
                ctx.tool,
                ctx.tool_args,
                max_retries=config.max_retries_per_tool,
                base_delay=config.retry_base_delay,
            )
            result_str = result  # execute_tool_with_retry returns str
        else:
            result = await ctx.tool.ainvoke(ctx.tool_args)
            if isinstance(result, (dict, list)):
                result_str = json.dumps(result, default=str)
            elif result is not None:
                result_str = str(result)
            else:
                result_str = ""

        # 6. After-execution hook (event extraction)
        result_str = hooks.after_execution(ctx, result_str, all_events)

        # 7. Truncate
        if config.truncate_tool_results and result_str:
            result_str = truncate_tool_result(
                result_str, config.tool_result_max_chars
            )

        # 8. Emit tool-result event
        result_events = hooks.on_tool_result(
            ctx.tool_name,
            result_str[:500] if result_str else "",
            ctx.tool_call_id,
        )
        if ctx.policy_decision:
            for evt in result_events:
                if evt.get("type") == "tool_result":
                    evt["policy_decision"] = ctx.policy_decision
                    if ctx.policy_reason:
                        evt["policy_reason"] = ctx.policy_reason
        all_events.extend(result_events)
        final_status = "completed"
        final_result_summary = result_str[:500] if result_str else ""

        return ToolExecutionResult(
            message=ToolMessage(
                content=result_str,
                tool_call_id=ctx.tool_call_id,
                name=ctx.tool_name,
            ),
            events=all_events,
            is_error=False,
        )

    except Exception as e:
        logger.error(
            "tool_execution_failed", tool=ctx.tool_name, error=str(e)
        )
        error_msg = f"Error executing {ctx.tool_name}: {e}"
        final_status = "failed"
        final_error = error_msg
        error_category = classify_error(str(e), ctx.tool_name)
        result_events = hooks.on_tool_result(
            ctx.tool_name, error_msg, ctx.tool_call_id
        )
        if ctx.policy_decision:
            for evt in result_events:
                if evt.get("type") == "tool_result":
                    evt["policy_decision"] = ctx.policy_decision
                    if ctx.policy_reason:
                        evt["policy_reason"] = ctx.policy_reason
        all_events.extend(result_events)
        return ToolExecutionResult(
            message=ToolMessage(
                content=error_msg,
                tool_call_id=ctx.tool_call_id,
                name=ctx.tool_name,
            ),
            events=all_events,
            is_error=True,
            error_category=error_category,
        )
    finally:
        if ctx.run_id and ctx.tool_call_id:
            try:
                from app.services.run_ledger import run_ledger_service

                await run_ledger_service.record_tool_execution_end(
                    run_id=ctx.run_id,
                    tool_call_id=ctx.tool_call_id,
                    status=final_status,
                    result_summary=final_result_summary,
                    error=final_error,
                )
            except Exception:
                pass


async def execute_tools_batch(
    tool_calls: list[dict],
    tool_map: dict[str, BaseTool],
    config: ReActLoopConfig,
    hooks: ToolExecutionHooks,
    user_id: str | None = None,
    task_id: str | None = None,
    run_id: str | None = None,
    hitl_partition: bool = False,
    hitl_check: Callable[[str], bool] | None = None,
    use_retry: bool = False,
) -> tuple[list[ToolMessage], list[dict], int, dict | None]:
    """Execute a batch of tool calls with parallel/sequential partitioning.

    Partitioning logic:
    - SEQUENTIAL_TOOLS (browser tools) always run sequentially
    - If hitl_partition=True and hitl_check provided: HITL-requiring tools
      run sequentially after parallel tools
    - Remaining tools run in parallel with semaphore

    Args:
        tool_calls: List of tool call dicts (name, args, id)
        tool_map: Mapping of tool names to tool instances
        config: ReAct loop config
        hooks: Agent-specific hook implementations
        user_id: User ID for context injection
        task_id: Task ID for context injection
        hitl_partition: Whether to partition HITL-requiring tools
        hitl_check: Function that returns True if a tool requires HITL
        use_retry: Whether to use retry logic for tool execution

    Returns:
        Tuple of (tool_messages, events, error_count, pending_interrupt)
    """
    all_messages: list[ToolMessage] = []
    all_events: list[dict] = []
    error_count = 0
    pending_interrupt: dict | None = None

    # Partition tool calls
    hitl_calls: list[dict] = []
    sequential_calls: list[dict] = []
    parallel_calls: list[dict] = []

    for tc in tool_calls:
        tool_name = tc.get("name", "")
        if hitl_partition and hitl_check and hitl_check(tool_name):
            hitl_calls.append(tc)
        elif tool_name in SEQUENTIAL_TOOLS:
            sequential_calls.append(tc)
        else:
            parallel_calls.append(tc)

    # Helper to execute a single tool call through the pipeline
    async def _run_one(tc: dict) -> ToolExecutionResult:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool_call_id = tc.get("id", "")
        tool = tool_map.get(tool_name)

        ctx = ToolExecutionContext(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_call_id=tool_call_id,
            tool=tool,
            user_id=user_id,
            task_id=task_id,
            run_id=run_id,
        )
        return await execute_tool(ctx, hooks=hooks, config=config, use_retry=use_retry)

    # Execute parallel tools concurrently
    if len(parallel_calls) > 1 and config.parallel_tool_execution:
        semaphore = asyncio.Semaphore(config.max_parallel_tools or 5)

        async def _run_with_sem(tc: dict) -> ToolExecutionResult:
            async with semaphore:
                return await _run_one(tc)

        parallel_results = await asyncio.gather(
            *[_run_with_sem(tc) for tc in parallel_calls],
            return_exceptions=True,
        )

        for i, result in enumerate(parallel_results):
            if isinstance(result, Exception):
                tc = parallel_calls[i]
                logger.error(
                    "parallel_tool_execution_failed",
                    tool=tc.get("name", ""),
                    error=str(result),
                )
                error_msg = f"Error executing {tc.get('name', '')}: {result}"
                all_messages.append(
                    ToolMessage(
                        content=error_msg,
                        tool_call_id=tc.get("id", ""),
                        name=tc.get("name", ""),
                    )
                )
                error_count += 1
            else:
                all_events.extend(result.events)
                if result.is_error:
                    error_count += 1
                if result.message:
                    all_messages.append(result.message)
    elif parallel_calls:
        # Single or no parallelism — run sequentially
        for tc in parallel_calls:
            result = await _run_one(tc)
            all_events.extend(result.events)
            if result.is_error:
                error_count += 1
            if result.message:
                all_messages.append(result.message)

    # Execute sequential tools (browser tools) one by one
    for tc in sequential_calls:
        result = await _run_one(tc)
        all_events.extend(result.events)
        if result.is_error:
            error_count += 1
        if result.message:
            all_messages.append(result.message)

    # Execute HITL-requiring tools sequentially (they may trigger interrupts)
    for tc in hitl_calls:
        result = await _run_one(tc)
        all_events.extend(result.events)

        if result.pending_interrupt:
            # Stop processing — return collected messages + the interrupt
            pending_interrupt = result.pending_interrupt
            return all_messages, all_events, error_count, pending_interrupt

        if result.is_error:
            error_count += 1
        if result.message:
            all_messages.append(result.message)

    return all_messages, all_events, error_count, pending_interrupt


# ---------------------------------------------------------------------------
# Agent-specific hook implementations
# ---------------------------------------------------------------------------


class TaskToolHooks(ToolExecutionHooks):
    """Hook implementation for the task agent.

    Handles HITL approval, ask_user interrupts, guardrail scanning,
    and event extraction (images, skills, app builder, slides).
    """

    def __init__(self, state: dict, skip_before_execution: bool = False):
        self.state = state
        self.skip_before_execution = skip_before_execution
        self._image_event_count = 0
        # Lazy-loaded caches
        self._app_builder_tools: set[str] | None = None

    def _get_app_builder_tools(self) -> set[str]:
        if self._app_builder_tools is None:
            from app.agents.tools import ToolCategory, get_tools_by_category

            self._app_builder_tools = {
                t.name for t in get_tools_by_category(ToolCategory.APP_BUILDER)
            }
        return self._app_builder_tools

    async def before_execution(
        self, ctx: ToolExecutionContext
    ) -> ToolExecutionResult | None:
        """HITL approval check, ask_user handling, guardrail scan, lazy upload."""
        from app.agents.hitl.interrupt_manager import get_interrupt_manager
        from app.agents.hitl.tool_risk import requires_approval, requires_approval_for_skill
        from app.agents.policy import PolicyInput, get_policy_engine
        from app.agents.policy.engine import PolicyDecision
        from app.config import settings
        from app.guardrails.scanners.tool_scanner import tool_scanner
        from app.agents.tools.registry import get_tool_contract

        interrupt_manager = get_interrupt_manager()
        user_id = self.state.get("user_id")
        task_id = self.state.get("task_id")

        # Lazy attachment upload: upload files to sandbox on first sandbox tool use
        _SANDBOX_TOOLS = {
            "execute_code", "execute_script", "sandbox_file",
            "shell_exec", "shell_view", "shell_wait", "shell_kill",
            "file_read", "file_write", "file_str_replace",
            "file_find_by_name", "file_find_in_content",
        }

        if (
            ctx.tool_name in _SANDBOX_TOOLS
            and not self.state.get("files_uploaded_to_sandbox")
        ):
            attachment_ids = self.state.get("attachment_ids") or []
            if attachment_ids and user_id:
                from app.agents.skills.builtin.data_analysis_skill import (
                    _normalize_attachment_ids,
                    _upload_attachments_to_sandbox,
                )

                attachment_ids = _normalize_attachment_ids(attachment_ids)

                _upload_evts, _ok, _ = await _upload_attachments_to_sandbox(
                    attachment_ids, user_id, task_id,
                )
                if _ok:
                    self.state["files_uploaded_to_sandbox"] = True

        # 0. Skill-level HITL policy (invoke_skill can encapsulate high-risk tools)
        if ctx.tool_name == "invoke_skill":
            skill_id = str(ctx.tool_args.get("skill_id", "")).strip()
            user_intent_source = str(
                ctx.tool_args.get("user_intent_source") or "agent_selected"
            )

            skill_policy = get_policy_engine().decide(
                PolicyInput(
                    tool_name="invoke_skill",
                    tool_args=ctx.tool_args,
                    auto_approve_tools=self.state.get("auto_approve_tools", []),
                    hitl_enabled=self.state.get("hitl_enabled", True),
                    risk_threshold=settings.hitl_default_risk_threshold,
                    contract=get_tool_contract("invoke_skill"),
                    is_skill_invocation=True,
                    user_intent_source=user_intent_source,  # audit context only
                )
            )
            ctx.policy_decision = skill_policy.decision.value
            ctx.policy_reason = skill_policy.reason_code
            if requires_approval_for_skill(
                skill_id=skill_id,
                auto_approve_tools=self.state.get("auto_approve_tools", []),
                hitl_enabled=self.state.get("hitl_enabled", True),
                risk_threshold=settings.hitl_default_risk_threshold,
            ):
                from app.agents import events as agent_events
                from app.agents.hitl.interrupt_manager import create_approval_interrupt

                thread_id = task_id or user_id or "default"
                interrupt_event = create_approval_interrupt(
                    tool_name=f"invoke_skill:{skill_id}" if skill_id else "invoke_skill",
                    args=ctx.tool_args,
                    timeout_seconds=settings.hitl_approval_timeout,
                )
                interrupt_id = interrupt_event["interrupt_id"]

                events = [
                    agent_events.tool_call(
                        tool=ctx.tool_name, args=ctx.tool_args, tool_id=ctx.tool_call_id
                    ),
                    interrupt_event,
                ]

                await interrupt_manager.create_interrupt(
                    thread_id=thread_id,
                    interrupt_id=interrupt_id,
                    interrupt_data=interrupt_event,
                )

                return ToolExecutionResult(
                    message=None,
                    events=events,
                    pending_interrupt={
                        "interrupt_id": interrupt_id,
                        "thread_id": thread_id,
                        "tool_call_id": ctx.tool_call_id,
                        "tool_name": ctx.tool_name,
                        "tool_args": ctx.tool_args,
                        "is_approval": True,
                    },
                )
            if skill_policy.decision == PolicyDecision.DENY:
                return ToolExecutionResult(
                    message=ToolMessage(
                        content=f"Tool denied by policy: {skill_policy.reason_code}",
                        tool_call_id=ctx.tool_call_id,
                        name=ctx.tool_name,
                    ),
                    events=[],
                    is_error=True,
                )

        # 1. HITL approval check
        tool_policy = get_policy_engine().decide(
            PolicyInput(
                tool_name=ctx.tool_name,
                tool_args=ctx.tool_args,
                auto_approve_tools=self.state.get("auto_approve_tools", []),
                hitl_enabled=self.state.get("hitl_enabled", True),
                risk_threshold=settings.hitl_default_risk_threshold,
                contract=get_tool_contract(ctx.tool_name),
            )
        )
        ctx.policy_decision = tool_policy.decision.value
        ctx.policy_reason = tool_policy.reason_code
        if tool_policy.decision == PolicyDecision.DENY:
            return ToolExecutionResult(
                message=ToolMessage(
                    content=f"Tool denied by policy: {tool_policy.reason_code}",
                    tool_call_id=ctx.tool_call_id,
                    name=ctx.tool_name,
                ),
                events=[],
                is_error=True,
            )
        if requires_approval(
            ctx.tool_name,
            auto_approve_tools=self.state.get("auto_approve_tools", []),
            hitl_enabled=self.state.get("hitl_enabled", True),
            risk_threshold=settings.hitl_default_risk_threshold,
        ):
            from app.agents import events as agent_events
            from app.agents.hitl.interrupt_manager import create_approval_interrupt

            thread_id = task_id or user_id or "default"
            interrupt_event = create_approval_interrupt(
                tool_name=ctx.tool_name,
                args=ctx.tool_args,
                timeout_seconds=settings.hitl_approval_timeout,
            )
            interrupt_id = interrupt_event["interrupt_id"]

            events = [
                agent_events.tool_call(
                    tool=ctx.tool_name, args=ctx.tool_args, tool_id=ctx.tool_call_id
                ),
                interrupt_event,
            ]

            logger.info(
                "hitl_tool_approval_required",
                tool_name=ctx.tool_name,
                interrupt_id=interrupt_id,
                thread_id=thread_id,
                tool_call_id=ctx.tool_call_id,
            )

            await interrupt_manager.create_interrupt(
                thread_id=thread_id,
                interrupt_id=interrupt_id,
                interrupt_data=interrupt_event,
            )

            return ToolExecutionResult(
                message=None,
                events=events,
                pending_interrupt={
                    "interrupt_id": interrupt_id,
                    "thread_id": thread_id,
                    "tool_call_id": ctx.tool_call_id,
                    "tool_name": ctx.tool_name,
                    "tool_args": ctx.tool_args,
                    "is_approval": True,
                },
            )

        # 2. ask_user handling
        if ctx.tool_name == "ask_user":
            from app.agents import events as agent_events
            from app.agents.hitl.interrupt_manager import (
                create_confirm_interrupt,
                create_decision_interrupt,
                create_input_interrupt,
            )

            tool_events: list[dict] = [
                agent_events.tool_call(
                    tool=ctx.tool_name, args=ctx.tool_args, tool_id=ctx.tool_call_id
                )
            ]

            logger.info(
                "hitl_ask_user_args",
                tool_call_id=ctx.tool_call_id,
                args=ctx.tool_args,
            )

            question = ctx.tool_args.get("question", "")
            question_type = ctx.tool_args.get("question_type", "input")
            options = ctx.tool_args.get("options")
            context = ctx.tool_args.get("context")
            thread_id = task_id or user_id or "default"

            message = f"{context}\n\n{question}" if context else question

            if question_type == "confirmation":
                interrupt_event = create_confirm_interrupt(
                    title="Confirmation",
                    message=message,
                    timeout_seconds=settings.hitl_decision_timeout,
                )
            elif question_type == "decision" and options:
                interrupt_event = create_decision_interrupt(
                    title="Agent Question",
                    message=message,
                    options=options,
                    timeout_seconds=settings.hitl_decision_timeout,
                )
            else:
                interrupt_event = create_input_interrupt(
                    title="Agent Question",
                    message=message,
                    timeout_seconds=settings.hitl_decision_timeout,
                )

            interrupt_id = interrupt_event["interrupt_id"]
            tool_events.append(interrupt_event)

            logger.info(
                "hitl_ask_user",
                question_type=question_type,
                interrupt_id=interrupt_id,
                thread_id=thread_id,
                question=question[:100] if question else "",
                message=message[:100] if message else "",
                options_count=len(options) if options else 0,
                tool_call_id=ctx.tool_call_id,
            )

            await interrupt_manager.create_interrupt(
                thread_id=thread_id,
                interrupt_id=interrupt_id,
                interrupt_data=interrupt_event,
            )

            return ToolExecutionResult(
                message=None,
                events=tool_events,
                pending_interrupt={
                    "interrupt_id": interrupt_id,
                    "thread_id": thread_id,
                    "tool_call_id": ctx.tool_call_id,
                    "tool_name": ctx.tool_name,
                },
            )

        # 3. Tool not found (handled by execute_tool, but guard here for scan)
        if ctx.tool is None:
            return None  # Let execute_tool handle it

        # 4. Guardrail scan
        tool_scan_result = await tool_scanner.scan(ctx.tool_name, ctx.tool_args)
        if tool_scan_result.blocked:
            from app.agents import events as agent_events

            logger.warning(
                "tool_guardrail_blocked",
                tool_name=ctx.tool_name,
                violations=[v.value for v in tool_scan_result.violations],
                reason=tool_scan_result.reason,
            )
            error_msg = f"Tool blocked: {tool_scan_result.reason}"
            return ToolExecutionResult(
                message=ToolMessage(
                    content=error_msg,
                    tool_call_id=ctx.tool_call_id,
                    name=ctx.tool_name,
                ),
                events=[
                    agent_events.tool_call(
                        tool=ctx.tool_name, args=ctx.tool_args, tool_id=ctx.tool_call_id
                    ),
                    agent_events.tool_result(
                        tool=ctx.tool_name, content=error_msg, tool_id=ctx.tool_call_id
                    ),
                ],
                is_error=True,
            )

        return None

    def after_execution(
        self, ctx: ToolExecutionContext, result_str: str, events: list[dict]
    ) -> str:
        """Extract events from tool results (images, skills, app builder, shell/code, slides)."""
        from app.agents.tools.event_extraction import (
            extract_app_builder_events,
            extract_image_events,
            extract_shell_and_code_events,
            extract_skill_events,
            extract_slide_events,
        )

        user_id = self.state.get("user_id")
        task_id = self.state.get("task_id")
        app_builder_tools = self._get_app_builder_tools()

        self._image_event_count = extract_image_events(
            ctx.tool_name, result_str, events, self._image_event_count
        )
        self._image_event_count = extract_skill_events(
            ctx.tool_name,
            result_str,
            events,
            self._image_event_count,
            task_id,
            user_id,
        )
        extract_app_builder_events(
            ctx.tool_name,
            result_str,
            events,
            app_builder_tools,
            task_id,
            user_id,
        )
        extract_slide_events(ctx.tool_name, result_str, events)
        extract_shell_and_code_events(ctx.tool_name, result_str, events)

        return result_str

    def on_tool_call(
        self, tool_name: str, tool_args: dict, tool_call_id: str
    ) -> list[dict]:
        from app.agents import events as agent_events

        return [agent_events.tool_call(tool=tool_name, args=tool_args, tool_id=tool_call_id)]

    def on_tool_result(
        self, tool_name: str, result: str, tool_call_id: str
    ) -> list[dict]:
        from app.agents import events as agent_events

        return [agent_events.tool_result(tool=tool_name, content=result, tool_id=tool_call_id)]


class ResearchToolHooks(ToolExecutionHooks):
    """Hook implementation for the research agent.

    Handles source parsing from search results and source event emission.
    """

    def __init__(self, state: dict | None = None, skip_before_execution: bool = False) -> None:
        self.state = state or {}
        self.skip_before_execution = skip_before_execution
        self.collected_sources: list[Any] = []

    async def before_execution(
        self, ctx: ToolExecutionContext
    ) -> ToolExecutionResult | None:
        """Apply tool guardrails for research tool calls."""
        from app.agents import events as agent_events
        from app.agents.hitl.interrupt_manager import create_approval_interrupt, get_interrupt_manager
        from app.agents.hitl.tool_risk import requires_approval, requires_approval_for_skill
        from app.agents.policy import PolicyInput, get_policy_engine
        from app.agents.policy.engine import PolicyDecision
        from app.config import settings
        from app.guardrails.scanners.tool_scanner import tool_scanner
        from app.agents.tools.registry import get_tool_contract

        if ctx.tool is None:
            return None

        user_id = self.state.get("user_id")
        task_id = self.state.get("task_id")
        thread_id = task_id or user_id or "default"

        # HITL approval for invoke_skill (skill-specific) and direct high-risk tools.
        if ctx.tool_name == "invoke_skill":
            skill_id = str(ctx.tool_args.get("skill_id", "")).strip()
            skill_policy = get_policy_engine().decide(
                PolicyInput(
                    tool_name="invoke_skill",
                    tool_args=ctx.tool_args,
                    auto_approve_tools=self.state.get("auto_approve_tools", []),
                    hitl_enabled=self.state.get("hitl_enabled", True),
                    risk_threshold=settings.hitl_default_risk_threshold,
                    contract=get_tool_contract("invoke_skill"),
                    is_skill_invocation=True,
                )
            )
            ctx.policy_decision = skill_policy.decision.value
            ctx.policy_reason = skill_policy.reason_code
            if requires_approval_for_skill(
                skill_id=skill_id,
                auto_approve_tools=self.state.get("auto_approve_tools", []),
                hitl_enabled=self.state.get("hitl_enabled", True),
                risk_threshold=settings.hitl_default_risk_threshold,
            ):
                interrupt_manager = get_interrupt_manager()
                interrupt_event = create_approval_interrupt(
                    tool_name=f"invoke_skill:{skill_id}" if skill_id else "invoke_skill",
                    args=ctx.tool_args,
                    timeout_seconds=settings.hitl_approval_timeout,
                )
                await interrupt_manager.create_interrupt(
                    thread_id=thread_id,
                    interrupt_id=interrupt_event["interrupt_id"],
                    interrupt_data=interrupt_event,
                )
                return ToolExecutionResult(
                    message=None,
                    events=[
                        agent_events.tool_call(
                            tool=ctx.tool_name, args=ctx.tool_args, tool_id=ctx.tool_call_id
                        ),
                        interrupt_event,
                    ],
                    pending_interrupt={
                        "interrupt_id": interrupt_event["interrupt_id"],
                        "thread_id": thread_id,
                        "tool_call_id": ctx.tool_call_id,
                        "tool_name": ctx.tool_name,
                        "tool_args": ctx.tool_args,
                        "is_approval": True,
                    },
                )
            if skill_policy.decision == PolicyDecision.DENY:
                return ToolExecutionResult(
                    message=ToolMessage(
                        content=f"Tool denied by policy: {skill_policy.reason_code}",
                        tool_call_id=ctx.tool_call_id,
                        name=ctx.tool_name,
                    ),
                    events=[],
                    is_error=True,
                )

        tool_policy = get_policy_engine().decide(
            PolicyInput(
                tool_name=ctx.tool_name,
                tool_args=ctx.tool_args,
                auto_approve_tools=self.state.get("auto_approve_tools", []),
                hitl_enabled=self.state.get("hitl_enabled", True),
                risk_threshold=settings.hitl_default_risk_threshold,
                contract=get_tool_contract(ctx.tool_name),
            )
        )
        ctx.policy_decision = tool_policy.decision.value
        ctx.policy_reason = tool_policy.reason_code
        if tool_policy.decision == PolicyDecision.DENY:
            return ToolExecutionResult(
                message=ToolMessage(
                    content=f"Tool denied by policy: {tool_policy.reason_code}",
                    tool_call_id=ctx.tool_call_id,
                    name=ctx.tool_name,
                ),
                events=[],
                is_error=True,
            )
        if requires_approval(
            ctx.tool_name,
            auto_approve_tools=self.state.get("auto_approve_tools", []),
            hitl_enabled=self.state.get("hitl_enabled", True),
            risk_threshold=settings.hitl_default_risk_threshold,
        ):
            interrupt_manager = get_interrupt_manager()
            interrupt_event = create_approval_interrupt(
                tool_name=ctx.tool_name,
                args=ctx.tool_args,
                timeout_seconds=settings.hitl_approval_timeout,
            )
            await interrupt_manager.create_interrupt(
                thread_id=thread_id,
                interrupt_id=interrupt_event["interrupt_id"],
                interrupt_data=interrupt_event,
            )
            return ToolExecutionResult(
                message=None,
                events=[
                    agent_events.tool_call(
                        tool=ctx.tool_name, args=ctx.tool_args, tool_id=ctx.tool_call_id
                    ),
                    interrupt_event,
                ],
                pending_interrupt={
                    "interrupt_id": interrupt_event["interrupt_id"],
                    "thread_id": thread_id,
                    "tool_call_id": ctx.tool_call_id,
                    "tool_name": ctx.tool_name,
                    "tool_args": ctx.tool_args,
                    "is_approval": True,
                },
            )

        tool_scan_result = await tool_scanner.scan(ctx.tool_name, ctx.tool_args)
        if tool_scan_result.blocked:
            error_msg = f"Tool blocked: {tool_scan_result.reason}"
            logger.warning(
                "research_tool_guardrail_blocked",
                tool_name=ctx.tool_name,
                violations=[v.value for v in tool_scan_result.violations],
                reason=tool_scan_result.reason,
            )
            return ToolExecutionResult(
                message=ToolMessage(
                    content=error_msg,
                    tool_call_id=ctx.tool_call_id,
                    name=ctx.tool_name,
                ),
                events=[
                    agent_events.tool_call(
                        tool=ctx.tool_name, args=ctx.tool_args, tool_id=ctx.tool_call_id
                    ),
                    agent_events.tool_result(
                        tool=ctx.tool_name, content=error_msg, tool_id=ctx.tool_call_id
                    ),
                ],
                is_error=True,
            )
        return None

    def after_execution(
        self, ctx: ToolExecutionContext, result_str: str, events: list[dict]
    ) -> str:
        """Parse search results and collect sources."""
        from app.agents import events as agent_events
        from app.agents.tools import parse_search_results

        new_sources = parse_search_results(result_str)
        self.collected_sources.extend(new_sources)

        for source in new_sources:
            events.append(
                agent_events.source(
                    title=source.title,
                    url=source.url,
                    snippet=source.snippet,
                    relevance_score=source.relevance_score,
                )
            )

        return result_str

    def on_tool_result(
        self, tool_name: str, result: str, tool_call_id: str
    ) -> list[dict]:
        from app.agents import events as agent_events

        return [agent_events.tool_result(tool=tool_name, content=result, tool_id=tool_call_id)]


class CanonicalToolHooks(ToolExecutionHooks):
    """Hook implementation for the canonical ReAct loop (react_tool.py).

    Wraps existing flat callbacks (on_tool_call, on_tool_result, on_token)
    and handles generate_image, invoke_skill image, and browser_navigate
    post-processing.
    """

    def __init__(
        self,
        on_tool_call: Callable[[str, dict, str], None] | None = None,
        on_tool_result: Callable[[str, str, str], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        all_events: list[dict] | None = None,
    ):
        self._on_tool_call = on_tool_call
        self._on_tool_result = on_tool_result
        self._on_token = on_token
        # Reference to the shared events list for image event counting
        self._all_events = all_events or []

    async def before_execution(
        self, ctx: ToolExecutionContext
    ) -> ToolExecutionResult | None:
        """Apply tool guardrails in canonical ReAct loops."""
        from app.guardrails.scanners.tool_scanner import tool_scanner

        if ctx.tool is None:
            return None

        tool_scan_result = await tool_scanner.scan(ctx.tool_name, ctx.tool_args)
        if tool_scan_result.blocked:
            error_msg = f"Tool blocked: {tool_scan_result.reason}"
            logger.warning(
                "canonical_tool_guardrail_blocked",
                tool_name=ctx.tool_name,
                violations=[v.value for v in tool_scan_result.violations],
                reason=tool_scan_result.reason,
            )
            return ToolExecutionResult(
                message=ToolMessage(
                    content=error_msg,
                    tool_call_id=ctx.tool_call_id,
                    name=ctx.tool_name,
                ),
                is_error=True,
            )
        return None

    def after_execution(
        self, ctx: ToolExecutionContext, result_str: str, events: list[dict]
    ) -> str:
        """Handle generate_image, invoke_skill image, and browser_navigate."""
        llm_result = result_str

        # Special handling for generate_image
        if ctx.tool_name == "generate_image" and result_str:
            try:
                parsed = json.loads(result_str)
                if parsed.get("success") and parsed.get("images"):
                    from app.agents.utils import extract_and_add_image_events

                    start_index = len(
                        [
                            e
                            for e in self._all_events
                            if isinstance(e, dict) and e.get("type") == "image"
                        ]
                    )
                    extract_and_add_image_events(
                        result_str, self._all_events, start_index=start_index
                    )

                    image_count = len(parsed["images"])
                    if self._on_token:
                        placeholders = (
                            "\n\n"
                            + "\n\n".join(
                                f"![generated-image:{start_index + i}]"
                                for i in range(image_count)
                            )
                            + "\n\n"
                        )
                        self._on_token(placeholders)

                    llm_result = json.dumps(
                        {
                            "success": True,
                            "message": f"Generated {image_count} image(s). Displayed to user.",
                            "count": image_count,
                        }
                    )
            except Exception as e:
                logger.warning(
                    "generate_image_result_processing_error", error=str(e)
                )

        # Special handling for invoke_skill image generation
        if ctx.tool_name == "invoke_skill" and result_str:
            try:
                parsed = json.loads(result_str)
                if parsed.get("skill_id") == "image_generation":
                    output = parsed.get("output") or {}
                    images = output.get("images") or []
                    if images:
                        from app.agents.utils import extract_and_add_image_events

                        start_index = len(
                            [
                                e
                                for e in self._all_events
                                if isinstance(e, dict) and e.get("type") == "image"
                            ]
                        )
                        extract_and_add_image_events(
                            json.dumps({"success": True, "images": images}),
                            self._all_events,
                            start_index=start_index,
                        )
                        image_count = len(images)
                        if self._on_token:
                            placeholders = (
                                "\n\n"
                                + "\n\n".join(
                                    f"![generated-image:{start_index + i}]"
                                    for i in range(image_count)
                                )
                                + "\n\n"
                            )
                            self._on_token(placeholders)
                        llm_result = json.dumps(
                            {
                                "success": True,
                                "message": f"Generated {image_count} image(s). Displayed to user.",
                                "count": image_count,
                            }
                        )
            except Exception as e:
                logger.warning(
                    "invoke_skill_image_result_processing_error", error=str(e)
                )

        # Special handling for browser_navigate: strip screenshot, keep content
        if ctx.tool_name == "browser_navigate" and result_str:
            try:
                parsed = json.loads(result_str)
                if (
                    parsed.get("success")
                    and parsed.get("screenshot")
                    and parsed.get("content")
                ):
                    llm_result = json.dumps(
                        {
                            "success": True,
                            "url": parsed.get("url", ""),
                            "content": parsed.get("content", ""),
                            "content_length": parsed.get("content_length", 0),
                            "sandbox_id": parsed.get("sandbox_id", ""),
                        }
                    )
            except Exception as e:
                logger.warning(
                    "browser_navigate_result_processing_error", error=str(e)
                )

        return llm_result

    def on_tool_call(
        self, tool_name: str, tool_args: dict, tool_call_id: str
    ) -> list[dict]:
        if self._on_tool_call:
            self._on_tool_call(tool_name, tool_args, tool_call_id)
        return []  # Canonical loop manages its own events list

    def on_tool_result(
        self, tool_name: str, result: str, tool_call_id: str
    ) -> list[dict]:
        if self._on_tool_result:
            self._on_tool_result(tool_name, result, tool_call_id)
        return []  # Canonical loop manages its own events list
