"""Type-safe event schema definitions for the multi-agent system.

This module provides structured event types and factory functions for
consistent event emission across all agents.
"""

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _timestamp() -> int:
    """Return current timestamp in milliseconds."""
    return int(time.time() * 1000)


class EventType(str, Enum):
    """Available event types for streaming to clients."""

    # Lifecycle events
    STAGE = "stage"
    COMPLETE = "complete"
    ERROR = "error"

    # Content events
    TOKEN = "token"
    IMAGE = "image"

    # Tool events
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # Routing events
    ROUTING = "routing"
    HANDOFF = "handoff"
    CONFIG = "config"

    # Research-specific events
    SOURCE = "source"

    # Code-specific events
    CODE_RESULT = "code_result"

    # Browser/sandbox events
    BROWSER_STREAM = "browser_stream"
    BROWSER_ACTION = "browser_action"

    # Terminal events (for app builder sandbox)
    TERMINAL_COMMAND = "terminal_command"
    TERMINAL_OUTPUT = "terminal_output"
    TERMINAL_ERROR = "terminal_error"
    TERMINAL_COMPLETE = "terminal_complete"

    # Skill events
    SKILL_OUTPUT = "skill_output"

    # Plan execution events
    PLAN_STEP = "plan_step"

    # Human-in-the-loop events
    INTERRUPT = "interrupt"
    INTERRUPT_RESPONSE = "interrupt_response"

    # Verification events (self-correction loop)
    VERIFICATION = "verification"

    # Workspace events (file updates in sandboxes)
    WORKSPACE_UPDATE = "workspace_update"

    # Usage tracking events
    USAGE = "usage"

    # Reasoning transparency events
    REASONING = "reasoning"

    # Parallel execution events
    PARALLEL_TASK = "parallel_task"

    # Task plan overview events (for Task Decomposition & Progress View)
    PLAN_OVERVIEW = "plan_overview"
    PLAN_STEP_COMPLETED = "plan_step_completed"

    # Todo-list persistence events (sandbox todo.md updates)
    TODO_UPDATE = "todo_update"

    # Web research (agentic search) events
    SEARCH_PLAN = "search_plan"
    SUB_QUERY_STATUS = "sub_query_status"
    KNOWLEDGE_UPDATE = "knowledge_update"
    CONFIDENCE_UPDATE = "confidence_update"
    REFINEMENT_START = "refinement_start"


class InterruptType(str, Enum):
    """Types of human-in-the-loop interrupts."""

    DECISION = "decision"  # Multiple choice decision point
    INPUT = "input"  # Free-form text input
    APPROVAL = "approval"  # Yes/no approval for high-risk action
    CONFIRM = "confirm"  # Yes/no confirmation


class StageStatus(str, Enum):
    """Stage lifecycle statuses."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Pydantic models for type-safe event creation


class StageEvent(BaseModel):
    """Event indicating a processing stage change."""

    type: Literal["stage"] = "stage"
    name: str = Field(..., description="Stage name (e.g., 'search', 'analyze')")
    description: str = Field(..., description="Human-readable description")
    status: StageStatus = Field(default=StageStatus.RUNNING)
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class TokenEvent(BaseModel):
    """Event for streaming token content."""

    type: Literal["token"] = "token"
    content: str = Field(..., description="Token content to stream")


class ImageEvent(BaseModel):
    """Event containing generated image data."""

    type: Literal["image"] = "image"
    data: str = Field(..., description="Base64-encoded image data")
    mime_type: str = Field(default="image/png", description="MIME type of the image")
    index: int | None = Field(default=None, description="Index for inline rendering with placeholders")


class ToolCallEvent(BaseModel):
    """Event indicating a tool is being called."""

    type: Literal["tool_call"] = "tool_call"
    tool: str = Field(..., description="Tool name")
    args: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    id: str | None = Field(default=None, description="Unique tool call ID for matching results")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class ToolResultEvent(BaseModel):
    """Event containing tool execution result."""

    type: Literal["tool_result"] = "tool_result"
    tool: str = Field(..., description="Tool name")
    content: str = Field(..., description="Tool result content (may be truncated)")
    id: str | None = Field(default=None, description="Tool call ID to match with corresponding tool_call")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class RoutingEvent(BaseModel):
    """Event indicating agent routing decision."""

    type: Literal["routing"] = "routing"
    agent: str = Field(..., description="Selected agent name")
    reason: str = Field(..., description="Routing reason")
    confidence: float | None = Field(default=None, description="Confidence score (0.0 to 1.0)")
    low_confidence: bool = Field(default=False, description="True if confidence is below threshold")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class HandoffEvent(BaseModel):
    """Event indicating agent handoff."""

    type: Literal["handoff"] = "handoff"
    source: str = Field(..., description="Source agent")
    target: str = Field(..., description="Target agent")
    task: str = Field(..., description="Delegated task description")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class ErrorEvent(BaseModel):
    """Event indicating an error occurred."""

    type: Literal["error"] = "error"
    error: str = Field(..., description="Error message")
    node: str | None = Field(default=None, description="Node where error occurred")
    name: str | None = Field(default=None, description="Stage/component name")
    description: str | None = Field(default=None, description="Human-readable description")
    status: Literal["failed"] = "failed"
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class CompleteEvent(BaseModel):
    """Event indicating processing is complete."""

    type: Literal["complete"] = "complete"


class SourceEvent(BaseModel):
    """Event containing a research source."""

    type: Literal["source"] = "source"
    title: str = Field(..., description="Source title")
    url: str = Field(..., description="Source URL")
    snippet: str = Field(..., description="Source snippet/description")
    relevance_score: float | None = Field(default=None, description="Relevance score")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class CodeResultEvent(BaseModel):
    """Event containing code execution result."""

    type: Literal["code_result"] = "code_result"
    output: str = Field(..., description="Execution output")
    exit_code: int | None = Field(default=None, description="Exit code")
    error: str | None = Field(default=None, description="Error message if failed")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class ConfigEvent(BaseModel):
    """Event containing configuration info."""

    type: Literal["config"] = "config"
    depth: str | None = Field(default=None, description="Research depth")
    scenario: str | None = Field(default=None, description="Research scenario")


class BrowserStreamEvent(BaseModel):
    """Event containing browser stream URL for live viewing."""

    type: Literal["browser_stream"] = "browser_stream"
    stream_url: str = Field(..., description="URL to view the browser stream")
    sandbox_id: str = Field(..., description="Sandbox identifier")
    auth_key: str | None = Field(default=None, description="Authentication key if required")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class BrowserActionEvent(BaseModel):
    """Event indicating a browser action is being performed."""

    type: Literal["browser_action"] = "browser_action"
    action: str = Field(..., description="Action being performed (navigate, click, type, scroll, etc.)")
    description: str = Field(..., description="Human-readable description of the action")
    target: str | None = Field(default=None, description="Target of the action (URL, coordinates, text)")
    status: str = Field(default="running", description="Action status (running, completed)")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class TerminalCommandEvent(BaseModel):
    """Event indicating a terminal command is being executed."""

    type: Literal["terminal_command"] = "terminal_command"
    command: str = Field(..., description="Command being executed")
    cwd: str = Field(default="/home/user", description="Working directory")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class TerminalOutputEvent(BaseModel):
    """Event containing terminal output."""

    type: Literal["terminal_output"] = "terminal_output"
    content: str = Field(..., description="Output content")
    stream: Literal["stdout", "stderr"] = Field(default="stdout", description="Output stream")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class TerminalErrorEvent(BaseModel):
    """Event indicating a terminal error."""

    type: Literal["terminal_error"] = "terminal_error"
    content: str = Field(..., description="Error message")
    exit_code: int | None = Field(default=None, description="Exit code if available")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class TerminalCompleteEvent(BaseModel):
    """Event indicating a terminal command completed."""

    type: Literal["terminal_complete"] = "terminal_complete"
    exit_code: int = Field(..., description="Command exit code")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class SkillOutputEvent(BaseModel):
    """Event containing skill execution output."""

    type: Literal["skill_output"] = "skill_output"
    skill_id: str = Field(..., description="Skill identifier")
    output: dict[str, Any] = Field(..., description="Skill execution output")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class PlanStepEvent(BaseModel):
    """Event indicating plan step progress during planned execution mode."""

    type: Literal["plan_step"] = "plan_step"
    step_number: int = Field(..., description="Current step number (1-based)")
    total_steps: int = Field(..., description="Total number of steps in the plan")
    action: str = Field(..., description="Description of the current step action")
    status: str = Field(default="running", description="Step status: running, completed, skipped")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class InterruptOption(BaseModel):
    """An option for a decision-type interrupt."""

    label: str = Field(..., description="Display text for the option")
    value: str = Field(..., description="Value to return when selected")
    description: str | None = Field(default=None, description="Additional description")


class InterruptEvent(BaseModel):
    """Event requesting human-in-the-loop input.

    Used to pause agent execution and collect user input for:
    - Decision: Multiple choice decision points
    - Input: Free-form text input collection
    - Approval: Yes/no approval for high-risk tool execution
    """

    type: Literal["interrupt"] = "interrupt"
    interrupt_id: str = Field(..., description="Unique ID for matching response")
    interrupt_type: InterruptType = Field(..., description="Type of interrupt")
    title: str = Field(..., description="Title displayed in dialog")
    message: str = Field(..., description="Detailed message/question")
    options: list[InterruptOption] | None = Field(
        default=None, description="Options for DECISION type"
    )
    tool_info: dict[str, Any] | None = Field(
        default=None, description="Tool details for APPROVAL type"
    )
    default_action: str | None = Field(
        default=None, description="Action to take on timeout (e.g., 'deny', 'approve', 'skip')"
    )
    timeout_seconds: int = Field(default=120, description="Timeout in seconds")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class InterruptResponseEvent(BaseModel):
    """Event containing user response to an interrupt."""

    type: Literal["interrupt_response"] = "interrupt_response"
    interrupt_id: str = Field(..., description="ID of the interrupt being responded to")
    action: str = Field(..., description="User action: approve, deny, skip, select, input")
    value: str | None = Field(default=None, description="Selected value or input text")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class WorkspaceUpdateEvent(BaseModel):
    """Event indicating a file was created, modified, or deleted in the sandbox workspace."""

    type: Literal["workspace_update"] = "workspace_update"
    operation: Literal["create", "modify", "delete"] = Field(
        ..., description="Type of file operation"
    )
    path: str = Field(..., description="Full path to the file")
    name: str = Field(..., description="File or directory name")
    is_directory: bool = Field(default=False, description="Whether this is a directory")
    size: int | None = Field(default=None, description="File size in bytes")
    sandbox_type: Literal["execution", "app"] = Field(
        ..., description="Type of sandbox (execution or app)"
    )
    sandbox_id: str = Field(..., description="Sandbox identifier")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class VerificationEvent(BaseModel):
    """Event containing verification result for planned execution."""

    type: Literal["verification"] = "verification"
    status: str = Field(..., description="Verification status: passed or failed")
    message: str = Field(default="", description="Verification result message")
    step: int | None = Field(default=None, description="Step number if applicable")
    step_number: int | None = Field(default=None, description="Canonical step number for clients")
    findings: list[str] = Field(default_factory=list, description="Machine-readable verification findings")
    retry_hint: str | None = Field(default=None, description="Suggested retry/adaptation hint")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class ReasoningEvent(BaseModel):
    """Agent reasoning/decision explanation."""

    type: Literal["reasoning"] = "reasoning"
    thinking: str = Field(..., description="The agent's reasoning or decision explanation")
    confidence: float | None = Field(default=None, description="Confidence level 0.0-1.0")
    context: str | None = Field(
        default=None,
        description="Context: routing, tool_selection, error_recovery, verification",
    )
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class ParallelTaskEvent(BaseModel):
    """Event tracking individual parallel sub-task progress."""

    type: Literal["parallel_task"] = "parallel_task"
    task_id: str = Field(..., description="Sub-task identifier")
    query: str | None = Field(default=None, description="Sub-task query")
    focus_area: str = Field(..., description="Focus area label")
    status: str = Field(..., description="pending, running, completed, failed")
    duration_ms: int | None = Field(default=None, description="Duration in ms")
    timestamp: int = Field(default_factory=_timestamp)


class TodoUpdateEvent(BaseModel):
    """Event containing updated todo.md content from sandbox."""

    type: Literal["todo_update"] = "todo_update"
    content: str = Field(..., description="Markdown checklist content of the todo file")
    checked: int = Field(default=0, description="Number of checked items")
    total: int = Field(default=0, description="Total number of checklist items")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class SearchPlanEvent(BaseModel):
    """Event containing the planned sub-queries for web research."""

    type: Literal["search_plan"] = "search_plan"
    sub_queries: list[dict[str, Any]] = Field(..., description="Planned sub-queries")
    timestamp: int = Field(default_factory=_timestamp)


class SubQueryStatusEvent(BaseModel):
    """Event tracking individual sub-query progress."""

    type: Literal["sub_query_status"] = "sub_query_status"
    id: str = Field(..., description="Sub-query ID")
    status: str = Field(..., description="pending|searching|done|gap")
    coverage: float | None = Field(default=None, description="Coverage score 0-1")
    timestamp: int = Field(default_factory=_timestamp)


class KnowledgeUpdateEvent(BaseModel):
    """Event with accumulated knowledge counts."""

    type: Literal["knowledge_update"] = "knowledge_update"
    facts_count: int = Field(..., description="Total extracted facts")
    sources_count: int = Field(..., description="Total unique sources")
    timestamp: int = Field(default_factory=_timestamp)


class ConfidenceUpdateEvent(BaseModel):
    """Event with overall search confidence."""

    type: Literal["confidence_update"] = "confidence_update"
    confidence: float = Field(..., description="Overall confidence 0-1")
    coverage_summary: dict[str, float] = Field(
        default_factory=dict, description="Per sub-query coverage"
    )
    timestamp: int = Field(default_factory=_timestamp)


class RefinementStartEvent(BaseModel):
    """Event indicating a refinement round is starting."""

    type: Literal["refinement_start"] = "refinement_start"
    round: int = Field(..., description="Refinement round number")
    follow_up_queries: list[str] = Field(default_factory=list)
    timestamp: int = Field(default_factory=_timestamp)


class PlanOverviewEvent(BaseModel):
    """Event containing the full task plan for display in the progress view."""

    type: Literal["plan_overview"] = "plan_overview"
    steps: list[dict[str, Any]] = Field(..., description="List of plan step objects")
    total_steps: int = Field(..., description="Total number of steps")
    completed_steps: int = Field(default=0, description="Number of completed steps")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class PlanStepCompletedEvent(BaseModel):
    """Event indicating a plan step has completed or failed."""

    type: Literal["plan_step_completed"] = "plan_step_completed"
    step_id: int = Field(..., description="Step index (0-based)")
    status: str = Field(..., description="completed or failed")
    completed_steps: int = Field(..., description="Total completed steps so far")
    total_steps: int = Field(..., description="Total number of steps")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


class UsageEvent(BaseModel):
    """Event containing token usage and cost information for an LLM call."""

    type: Literal["usage"] = "usage"
    input_tokens: int = Field(..., description="Number of input/prompt tokens")
    output_tokens: int = Field(..., description="Number of output/completion tokens")
    cached_tokens: int = Field(default=0, description="Number of cached input tokens")
    cost_usd: float = Field(..., description="Estimated cost in USD")
    model: str = Field(..., description="Model name used")
    tier: str = Field(..., description="Model tier (max, pro, lite)")
    timestamp: int = Field(default_factory=_timestamp, description="Event timestamp in ms")


# Factory functions for backward compatibility and convenience


def stage(
    name: str,
    description: str,
    status: str = "running",
) -> dict[str, Any]:
    """Create a stage event dictionary.

    Args:
        name: Stage name
        description: Human-readable description
        status: Stage status

    Returns:
        Stage event dictionary
    """
    return StageEvent(
        name=name,
        description=description,
        status=StageStatus(status),
    ).model_dump()


def token(content: str) -> dict[str, Any]:
    """Create a token event dictionary.

    Args:
        content: Token content

    Returns:
        Token event dictionary
    """
    return TokenEvent(content=content).model_dump()


def image(
    data: str,
    mime_type: str = "image/png",
    index: int | None = None,
) -> dict[str, Any]:
    """Create an image event dictionary.

    Args:
        data: Base64-encoded image data
        mime_type: MIME type of the image
        index: Index for inline rendering (matches ![generated-image:INDEX] placeholders)

    Returns:
        Image event dictionary
    """
    return ImageEvent(data=data, mime_type=mime_type, index=index).model_dump()


def tool_call(
    tool: str,
    args: dict[str, Any] | None = None,
    tool_id: str | None = None,
) -> dict[str, Any]:
    """Create a tool call event dictionary.

    Args:
        tool: Tool name
        args: Tool arguments
        tool_id: Unique ID for matching with tool_result

    Returns:
        Tool call event dictionary
    """
    return ToolCallEvent(tool=tool, args=args or {}, id=tool_id).model_dump()


def tool_result(
    tool: str,
    content: str,
    max_length: int = 500,
    tool_id: str | None = None,
) -> dict[str, Any]:
    """Create a tool result event dictionary.

    Args:
        tool: Tool name
        content: Tool result content
        max_length: Maximum content length
        tool_id: ID to match with corresponding tool_call

    Returns:
        Tool result event dictionary
    """
    truncated = content[:max_length] if len(content) > max_length else content
    return ToolResultEvent(tool=tool, content=truncated, id=tool_id).model_dump()


def routing(
    agent: str,
    reason: str,
    confidence: float | None = None,
    low_confidence: bool = False,
) -> dict[str, Any]:
    """Create a routing event dictionary.

    Args:
        agent: Selected agent name
        reason: Routing reason
        confidence: Confidence score (0.0 to 1.0)
        low_confidence: Whether confidence is below threshold

    Returns:
        Routing event dictionary
    """
    return RoutingEvent(
        agent=agent,
        reason=reason,
        confidence=confidence,
        low_confidence=low_confidence,
    ).model_dump()


def handoff(source: str, target: str, task: str) -> dict[str, Any]:
    """Create a handoff event dictionary.

    Args:
        source: Source agent
        target: Target agent
        task: Delegated task

    Returns:
        Handoff event dictionary
    """
    return HandoffEvent(source=source, target=target, task=task).model_dump()


def error(
    error_msg: str,
    node: str | None = None,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Create an error event dictionary.

    Args:
        error_msg: Error message
        node: Node where error occurred
        name: Stage/component name
        description: Human-readable description

    Returns:
        Error event dictionary
    """
    return ErrorEvent(
        error=error_msg,
        node=node,
        name=name,
        description=description,
    ).model_dump()


def complete() -> dict[str, Any]:
    """Create a complete event dictionary.

    Returns:
        Complete event dictionary
    """
    return CompleteEvent().model_dump()


def source(
    title: str,
    url: str,
    snippet: str,
    relevance_score: float | None = None,
) -> dict[str, Any]:
    """Create a source event dictionary.

    Args:
        title: Source title
        url: Source URL
        snippet: Source snippet
        relevance_score: Relevance score

    Returns:
        Source event dictionary
    """
    return SourceEvent(
        title=title,
        url=url,
        snippet=snippet,
        relevance_score=relevance_score,
    ).model_dump()


def code_result(
    output: str,
    exit_code: int | None = None,
    error_msg: str | None = None,
) -> dict[str, Any]:
    """Create a code result event dictionary.

    Args:
        output: Execution output
        exit_code: Exit code
        error_msg: Error message

    Returns:
        Code result event dictionary
    """
    return CodeResultEvent(
        output=output,
        exit_code=exit_code,
        error=error_msg,
    ).model_dump()


def config(
    depth: str | None = None,
    scenario: str | None = None,
) -> dict[str, Any]:
    """Create a config event dictionary.

    Args:
        depth: Research depth
        scenario: Research scenario

    Returns:
        Config event dictionary
    """
    return ConfigEvent(depth=depth, scenario=scenario).model_dump()


def browser_stream(
    stream_url: str,
    sandbox_id: str,
    auth_key: str | None = None,
) -> dict[str, Any]:
    """Create a browser stream event dictionary.

    This event provides a URL that can be embedded in an iframe
    to show live browser activity in the E2B sandbox.

    Args:
        stream_url: URL to view the browser stream
        sandbox_id: Sandbox identifier
        auth_key: Authentication key if required

    Returns:
        Browser stream event dictionary
    """
    return BrowserStreamEvent(
        stream_url=stream_url,
        sandbox_id=sandbox_id,
        auth_key=auth_key,
    ).model_dump()


def browser_action(
    action: str,
    description: str,
    target: str | None = None,
    status: str = "running",
) -> dict[str, Any]:
    """Create a browser action event dictionary.

    This event indicates a browser action is being performed,
    helping sync the UI progress with what's visible in the browser stream.

    Args:
        action: Action type (navigate, click, type, scroll, screenshot, etc.)
        description: Human-readable description
        target: Target of the action (URL, coordinates, text)
        status: Action status (running, completed)

    Returns:
        Browser action event dictionary
    """
    return BrowserActionEvent(
        action=action,
        description=description,
        target=target,
        status=status,
    ).model_dump()


def terminal_command(
    command: str,
    cwd: str = "/home/user",
) -> dict[str, Any]:
    """Create a terminal command event dictionary.

    Args:
        command: Command being executed
        cwd: Working directory

    Returns:
        Terminal command event dictionary
    """
    return TerminalCommandEvent(
        command=command,
        cwd=cwd,
    ).model_dump()


def terminal_output(
    content: str,
    stream: str = "stdout",
) -> dict[str, Any]:
    """Create a terminal output event dictionary.

    Args:
        content: Output content
        stream: Output stream (stdout or stderr)

    Returns:
        Terminal output event dictionary
    """
    return TerminalOutputEvent(
        content=content,
        stream=stream,
    ).model_dump()


def terminal_error(
    content: str,
    exit_code: int | None = None,
) -> dict[str, Any]:
    """Create a terminal error event dictionary.

    Args:
        content: Error message
        exit_code: Exit code if available

    Returns:
        Terminal error event dictionary
    """
    return TerminalErrorEvent(
        content=content,
        exit_code=exit_code,
    ).model_dump()


def terminal_complete(
    exit_code: int,
) -> dict[str, Any]:
    """Create a terminal complete event dictionary.

    Args:
        exit_code: Command exit code

    Returns:
        Terminal complete event dictionary
    """
    return TerminalCompleteEvent(
        exit_code=exit_code,
    ).model_dump()


def skill_output(
    skill_id: str,
    output: dict[str, Any],
) -> dict[str, Any]:
    """Create a skill output event dictionary.

    Args:
        skill_id: Skill identifier
        output: Skill execution output

    Returns:
        Skill output event dictionary
    """
    return SkillOutputEvent(
        skill_id=skill_id,
        output=output,
    ).model_dump()


def plan_step(
    step_number: int,
    total_steps: int,
    action: str,
    status: str = "running",
) -> dict[str, Any]:
    """Create a plan step event dictionary.

    Args:
        step_number: Current step number (1-based)
        total_steps: Total number of steps in the plan
        action: Description of the current step action
        status: Step status (running, completed, skipped)

    Returns:
        Plan step event dictionary
    """
    return PlanStepEvent(
        step_number=step_number,
        total_steps=total_steps,
        action=action,
        status=status,
    ).model_dump()


def interrupt(
    interrupt_id: str,
    interrupt_type: InterruptType | str,
    title: str,
    message: str,
    options: list[dict[str, str]] | None = None,
    tool_info: dict[str, Any] | None = None,
    default_action: str | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Create an interrupt event dictionary.

    Args:
        interrupt_id: Unique ID for matching response
        interrupt_type: Type of interrupt (decision, input, approval)
        title: Title displayed in dialog
        message: Detailed message/question
        options: Options for DECISION type [{"label": "...", "value": "...", "description": "..."}]
        tool_info: Tool details for APPROVAL type
        default_action: Action to take on timeout
        timeout_seconds: Timeout in seconds

    Returns:
        Interrupt event dictionary
    """
    interrupt_type_enum = (
        InterruptType(interrupt_type)
        if isinstance(interrupt_type, str)
        else interrupt_type
    )
    parsed_options = None
    if options:
        parsed_options = [
            InterruptOption(
                label=opt.get("label", ""),
                value=opt.get("value", ""),
                description=opt.get("description"),
            )
            for opt in options
        ]

    return InterruptEvent(
        interrupt_id=interrupt_id,
        interrupt_type=interrupt_type_enum,
        title=title,
        message=message,
        options=parsed_options,
        tool_info=tool_info,
        default_action=default_action,
        timeout_seconds=timeout_seconds,
    ).model_dump()


def interrupt_response(
    interrupt_id: str,
    action: str,
    value: str | None = None,
) -> dict[str, Any]:
    """Create an interrupt response event dictionary.

    Args:
        interrupt_id: ID of the interrupt being responded to
        action: User action (approve, deny, skip, select, input)
        value: Selected value or input text

    Returns:
        Interrupt response event dictionary
    """
    return InterruptResponseEvent(
        interrupt_id=interrupt_id,
        action=action,
        value=value,
    ).model_dump()


def workspace_update(
    operation: str,
    path: str,
    name: str,
    sandbox_type: str,
    sandbox_id: str,
    is_directory: bool = False,
    size: int | None = None,
) -> dict[str, Any]:
    """Create a workspace update event dictionary.

    This event is emitted when files are created, modified, or deleted
    in an E2B sandbox workspace.

    Args:
        operation: Type of operation (create, modify, delete)
        path: Full path to the file
        name: File or directory name
        sandbox_type: Type of sandbox (execution or app)
        sandbox_id: Sandbox identifier
        is_directory: Whether this is a directory
        size: File size in bytes

    Returns:
        Workspace update event dictionary
    """
    return WorkspaceUpdateEvent(
        operation=operation,
        path=path,
        name=name,
        sandbox_type=sandbox_type,
        sandbox_id=sandbox_id,
        is_directory=is_directory,
        size=size,
    ).model_dump()


def verification(
    status: str,
    message: str = "",
    step: int | None = None,
    findings: list[str] | None = None,
    retry_hint: str | None = None,
) -> dict[str, Any]:
    """Create a verification event dictionary.

    Args:
        status: Verification status (passed or failed)
        message: Verification result message
        step: Step number if applicable

    Returns:
        Verification event dictionary
    """
    return VerificationEvent(
        status=status,
        message=message,
        step=step,
        step_number=step,
        findings=findings or [],
        retry_hint=retry_hint,
    ).model_dump()


def usage(
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    cost_usd: float,
    model: str,
    tier: str,
) -> dict[str, Any]:
    """Create a usage event dictionary.

    Args:
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        cached_tokens: Number of cached input tokens
        cost_usd: Estimated cost in USD
        model: Model name used
        tier: Model tier (max, pro, lite)

    Returns:
        Usage event dictionary
    """
    return UsageEvent(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cost_usd=cost_usd,
        model=model,
        tier=tier,
    ).model_dump()


def reasoning(
    thinking: str,
    confidence: float | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    """Create a reasoning event dictionary.

    Args:
        thinking: The agent's reasoning or decision explanation
        confidence: Confidence level (0.0 to 1.0)
        context: Context category (routing, tool_selection, error_recovery, verification)

    Returns:
        Reasoning event dictionary
    """
    return ReasoningEvent(
        thinking=thinking,
        confidence=confidence,
        context=context,
    ).model_dump()


def plan_overview(
    steps: list[dict[str, Any]],
    total_steps: int,
    completed_steps: int = 0,
) -> dict[str, Any]:
    """Create a plan overview event dictionary.

    Args:
        steps: List of plan step objects with id, title, description, status
        total_steps: Total number of steps
        completed_steps: Number of completed steps

    Returns:
        Plan overview event dictionary
    """
    return PlanOverviewEvent(
        steps=steps,
        total_steps=total_steps,
        completed_steps=completed_steps,
    ).model_dump()


def plan_step_completed(
    step_id: int,
    status: str,
    completed_steps: int,
    total_steps: int,
) -> dict[str, Any]:
    """Create a plan step completed event dictionary.

    Args:
        step_id: Step index (0-based)
        status: Step status (completed or failed)
        completed_steps: Total completed steps so far
        total_steps: Total number of steps

    Returns:
        Plan step completed event dictionary
    """
    return PlanStepCompletedEvent(
        step_id=step_id,
        status=status,
        completed_steps=completed_steps,
        total_steps=total_steps,
    ).model_dump()


def parallel_task(
    task_id: str,
    focus_area: str,
    status: str,
    query: str | None = None,
    duration_ms: int | None = None,
) -> dict[str, Any]:
    """Create a parallel task event dictionary.

    Args:
        task_id: Sub-task identifier
        focus_area: Focus area label
        status: Task status (pending, running, completed, failed)
        query: Optional sub-task query
        duration_ms: Optional duration in milliseconds

    Returns:
        Parallel task event dictionary
    """
    return ParallelTaskEvent(
        task_id=task_id,
        focus_area=focus_area,
        status=status,
        query=query,
        duration_ms=duration_ms,
    ).model_dump()


def todo_update(
    content: str,
    checked: int = 0,
    total: int = 0,
) -> dict[str, Any]:
    """Create a todo_update event dictionary.

    Args:
        content: Markdown checklist content of the todo file
        checked: Number of checked items
        total: Total number of checklist items

    Returns:
        Todo update event dictionary
    """
    return TodoUpdateEvent(
        content=content,
        checked=checked,
        total=total,
    ).model_dump()


def search_plan(sub_queries: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a search_plan event dictionary."""
    return SearchPlanEvent(sub_queries=sub_queries).model_dump()


def sub_query_status(
    id: str, status: str, coverage: float | None = None
) -> dict[str, Any]:
    """Create a sub_query_status event dictionary."""
    return SubQueryStatusEvent(id=id, status=status, coverage=coverage).model_dump()


def knowledge_update(facts_count: int, sources_count: int) -> dict[str, Any]:
    """Create a knowledge_update event dictionary."""
    return KnowledgeUpdateEvent(
        facts_count=facts_count, sources_count=sources_count
    ).model_dump()


def confidence_update(
    confidence: float, coverage_summary: dict[str, float] | None = None
) -> dict[str, Any]:
    """Create a confidence_update event dictionary."""
    return ConfidenceUpdateEvent(
        confidence=confidence, coverage_summary=coverage_summary or {}
    ).model_dump()


def refinement_start(
    round: int, follow_up_queries: list[str] | None = None
) -> dict[str, Any]:
    """Create a refinement_start event dictionary."""
    return RefinementStartEvent(
        round=round, follow_up_queries=follow_up_queries or []
    ).model_dump()
