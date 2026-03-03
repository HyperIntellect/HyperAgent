"""Shared state definitions for the multi-agent system."""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage

from app.models.schemas import ModelTier, ResearchDepth, ResearchScenario
from app.services.search import SearchResult


def _override_reducer(a, b):
    """Override reducer: always takes the latest value (not accumulated)."""
    return b


class AgentType(str, Enum):
    """Available agent types.

    Simplified architecture:
    - TASK: General-purpose handler with skills (Q&A, images,
      writing, coding, data analysis, etc.)
    - RESEARCH: Deep research workflows with comprehensive analysis

    Note: Deprecated agent types (IMAGE, DATA, etc.) are mapped
    to TASK at the routing layer via AGENT_NAME_MAP.
    """

    TASK = "task"
    RESEARCH = "research"


# Re-export HandoffInfo and SharedAgentMemory from handoff module for backward compatibility
# These types are now consolidated in the handoff module
from app.agents.tools.handoff import HandoffInfo, SharedAgentMemory


class SupervisorState(TypedDict, total=False):
    """Base state for the supervisor graph.

    All subagent states extend this base state to ensure compatibility
    with the supervisor routing system.
    """

    # Input
    query: str
    mode: str | None  # Explicit mode override (AgentType value)
    messages: list[dict[str, Any]]  # Chat history for context

    # Routing
    selected_agent: str  # The agent type selected by router
    routing_reason: str  # Explanation for routing decision
    routing_confidence: float  # Confidence score from router (0.0 to 1.0)
    parallel_eligible: bool  # Whether query is eligible for parallel execution

    # Handoff support (multi-agent collaboration)
    active_agent: str | None  # Currently active agent (for handoff tracking)
    delegated_task: str | None  # Task description if this was a handoff
    handoff_context: str | None  # Additional context from handoff
    handoff_count: int  # Number of handoffs in current request
    handoff_history: list[HandoffInfo]  # History of handoffs
    pending_handoff: HandoffInfo | None  # Pending handoff to process
    shared_memory: SharedAgentMemory  # Shared state across agents

    # Tool execution tracking (shared across all agents)
    tool_iterations: int  # Count of tool-call loops for current agent

    # Context compression
    context_summary: str | None  # Compressed summary of older conversation history

    # Human-in-the-loop (HITL) support
    pending_interrupt: dict[str, Any] | None  # Current interrupt waiting for response
    interrupt_response: dict[str, Any] | None  # Response to pending interrupt
    interrupt_history: list[dict[str, Any]]  # History of interrupts in this session
    # auto_approve_tools: List of tool names the user has approved with "Approve Always"
    # for this session. When a tool is in this list, it bypasses the approval dialog
    # and executes directly. Updated when user selects "approve_always" action.
    # Example: ["execute_code", "browser_navigate"]
    auto_approve_tools: list[str]
    hitl_enabled: bool  # Whether HITL is enabled for this request

    # Output
    response: str  # Final text response
    # WARNING: Uses operator.add reducer — nodes must only return NEW events,
    # never the full accumulated list. Returning all events causes duplication
    # because operator.add appends the returned list to the existing one.
    events: Annotated[list[dict[str, Any]], operator.add]  # Streaming events

    # Metadata
    task_id: str | None
    run_id: str | None  # Durable run ledger id
    user_id: str | None
    attachment_ids: list[str]  # IDs of attached files for tool access
    image_attachments: list[
        dict
    ]  # Base64-encoded image data for vision tools [{id, filename, base64_data, mime_type}]
    provider: str
    model: str | None
    tier: ModelTier | None  # User-specified tier override
    locale: str  # User's preferred language (e.g., 'en', 'zh-CN')
    skills: list[str]  # Optional skill IDs to guide agent execution
    budget: dict[str, Any] | None  # Optional budget constraints for this run
    execution_mode: str | None  # auto|guided|strict


class TaskState(SupervisorState, total=False):
    """State for the task subagent."""

    # Chat-specific fields
    system_prompt: str
    lc_messages: list[BaseMessage]  # LangChain messages for tool calling
    has_error: bool  # Flag to signal error and stop the loop
    # Note: tool_iterations is inherited from SupervisorState

    # Planned execution mode (Phase 2)
    # When task_planning skill returns a plan, steps are parsed into execution_plan.
    # The agent then tracks progress through the plan step-by-step.
    execution_plan: list[dict]  # Parsed plan steps [{step_number, action, tool_or_skill, ...}]
    current_step_index: int  # 0-indexed pointer into execution_plan

    # Self-correction / verification loop
    # consecutive_errors uses an override reducer (lambda a, b: b) so it is
    # always replaced by the latest value rather than accumulated.
    consecutive_errors: Annotated[int, _override_reducer]  # Consecutive tool errors (override reducer)
    completed_step_results: Annotated[list[dict], operator.add]  # Accumulated step results
    plan_revision_count: Annotated[int, _override_reducer]  # Number of plan revisions

    # Todo-list as attention manipulation (Context Engineering)
    # The active_todo is rewritten into the prompt after every tool call to keep
    # the global plan in the model's recent attention window across many iterations.
    active_todo: str | None  # Current plan/todo state for attention manipulation

    # KV-cache optimization: hash of the stable prefix (system prompt + tool schemas)
    # so we can detect unintended prefix changes between iterations.
    prefix_hash: str | None  # MD5 hash of the stable prefix content

    # Anti-repetition: hashes of recent tool calls to detect stuck loops.
    # Each hash = md5(tool_name + sorted(args)). Max 5 recent entries.
    # Uses override reducer so the node always sets the full list.
    last_tool_calls_hash: Annotated[list[str], _override_reducer]

    # Whether attached files have been uploaded to the sandbox (one-time flag)
    files_uploaded_to_sandbox: bool

    # Enhanced verification
    verification_results: list[dict]  # Detailed per-step verification outcomes
    verified_steps: list[int]  # Step numbers that have been verified
    verification_status: str | None  # Latest verifier status: passed|partial|failed|error
    verification_attempts: int  # Number of adaptation attempts after failed verification
    verification_retry_hint: str | None  # Suggested retry/adaptation guidance


class ResearchState(SupervisorState, total=False):
    """DEPRECATED: Research is now handled by DeepResearchSkill.

    This state is kept for backward compatibility with existing DB records
    and migration scripts. Do not use for new code.
    """

    # Research configuration
    depth: ResearchDepth
    scenario: ResearchScenario
    system_prompt: str
    report_structure: list[str]
    depth_config: dict[str, Any]

    # Tool calling support
    lc_messages: list[BaseMessage]  # LangChain messages for ReAct loop
    search_complete: bool  # Flag to exit search loop
    # Note: tool_iterations (inherited from SupervisorState) tracks search iterations

    # Error tracking
    consecutive_errors: int  # Consecutive tool errors for circuit breaker

    # Handoff tracking
    deferred_handoff: HandoffInfo | None  # Handoff deferred until search tools complete

    # Research outputs
    sources: list[SearchResult]
    analysis: str
    synthesis: str
    report_chunks: list[str]


# =============================================================================
# Node Output Types (for type-safe node returns)
# =============================================================================


class RouterOutput(TypedDict, total=False):
    """Return type for router_node."""

    selected_agent: str
    routing_reason: str
    routing_confidence: float
    parallel_eligible: bool
    active_agent: str
    delegated_task: str | None
    handoff_context: str | None
    pending_handoff: HandoffInfo | None
    events: list[dict[str, Any]]


class TaskOutput(TypedDict, total=False):
    """Return type for task_node."""

    response: str
    events: list[dict[str, Any]]
    pending_handoff: HandoffInfo | None
    handoff_count: int
    handoff_history: list[HandoffInfo]


class ErrorOutput(TypedDict, total=False):
    """Return type for error cases in nodes."""

    response: str
    events: list[dict[str, Any]]
    has_error: bool
