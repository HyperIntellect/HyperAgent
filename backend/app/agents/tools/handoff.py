"""Handoff tools for agent-to-agent delegation.

This module implements the Manus/LangGraph swarm pattern for multi-agent collaboration.
Agents can delegate tasks to other specialized agents using handoff tools.

This module consolidates all handoff-related functionality:
- Handoff tool creation and validation
- Handoff state management (HandoffInfo, SharedAgentMemory)
- Shared memory truncation for context management
- Handoff routing and history tracking
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# Agent type values (string literals to avoid circular imports with state.py)
# These should match the values in app.agents.state.AgentType
AGENT_TASK = "task"
AGENT_RESEARCH = "research"


# =============================================================================
# Type Definitions
# =============================================================================


class HandoffInfo(TypedDict, total=False):
    """Information about a handoff request."""

    source_agent: str  # Agent that initiated the handoff
    target_agent: str  # Agent to transfer control to
    task_description: str  # What the target agent should do
    context: str  # Additional context for the handoff
    handoff_artifacts: NotRequired[list[dict[str, Any]]]  # [{path, storage_key, size}]


class SharedAgentMemory(TypedDict, total=False):
    """Shared memory accessible across agents during handoffs.

    This allows agents to share findings, intermediate results, and context
    when delegating tasks to other agents.
    """

    # Research findings from research agent
    research_findings: str
    research_sources: list[dict[str, Any]]

    # Code artifacts from code agent
    generated_code: str
    code_language: str
    execution_results: str

    # General context that any agent can add
    additional_context: str


class HandoffInput(BaseModel):
    """Input schema for handoff tools."""

    task_description: str = Field(
        ...,
        description="A clear description of what the target agent should accomplish.",
    )
    context: str = Field(
        default="",
        description="Additional context or information to help the target agent.",
    )


# =============================================================================
# Constants
# =============================================================================

# Maximum number of handoffs allowed to prevent infinite loops
MAX_HANDOFFS = 3

# Define which agents can delegate to which other agents
# SCAFFOLDING: The handoff system is currently inactive — all agent types route
# to the TASK agent, and the matrix below has no valid targets. The code is
# preserved as scaffolding for future multi-agent expansion. When adding new
# agents, populate HANDOFF_MATRIX with valid source→target pairs.
HANDOFF_MATRIX: dict[str, list[str]] = {
    AGENT_TASK: [],  # Research is now a skill, no standalone agents to hand off to
    AGENT_RESEARCH: [],  # Kept for backward compatibility
}

# Agent descriptions for handoff tool docstrings
AGENT_DESCRIPTIONS: dict[str, str] = {
    AGENT_TASK: "General-purpose task handling: Q&A, images, code, writing, data analysis, and more (has skills for all these)",
    AGENT_RESEARCH: "In-depth web research, analysis, and comprehensive reports",
}

# Shared memory context budget configuration
# Total budget for shared memory context in characters
SHARED_MEMORY_TOTAL_BUDGET = 8000

# Priority weights for different memory types (higher = more budget allocation)
SHARED_MEMORY_PRIORITIES = {
    "research_findings": 3,  # High priority - core research output
    "research_sources": 2,  # Medium priority - supporting evidence
    "generated_code": 3,  # High priority - code artifacts
    "code_language": 1,  # Low priority - just metadata
    "execution_results": 2,  # Medium priority - results
    "additional_context": 2,  # Medium priority
}

# Minimum allocation per field (prevents complete truncation)
SHARED_MEMORY_MIN_CHARS = 200


# =============================================================================
# Tool Creation
# =============================================================================

# SCAFFOLDING: These tool creation functions are inactive while HANDOFF_MATRIX
# is empty. They will produce handoff tools once targets are configured.


def create_handoff_tool(
    source_agent: str,
    target_agent: str,
    description: str | None = None,
) -> BaseTool:
    """Create a handoff tool for transferring control to another agent.

    This creates a tool that, when called by the source agent, signals
    that control should be transferred to the target agent with the
    specified task and context.

    Args:
        source_agent: The agent that will use this tool
        target_agent: The agent to transfer control to
        description: Optional custom description for the tool

    Returns:
        A LangChain BaseTool configured for handoff
    """
    if description is None:
        description = f"Transfer this task to the {target_agent} agent for {AGENT_DESCRIPTIONS.get(target_agent, 'specialized processing')}."

    tool_name = f"handoff_to_{target_agent}"

    @tool(args_schema=HandoffInput)
    def handoff_tool(task_description: str, context: str = "") -> dict[str, Any]:
        """Handoff to another agent."""
        logger.info(
            "handoff_requested",
            source=source_agent,
            target=target_agent,
            task=task_description[:100],
        )
        return {
            "handoff": True,
            "target_agent": target_agent,
            "task_description": task_description,
            "context": context,
            "source_agent": source_agent,
        }

    # Update the function metadata
    handoff_tool.name = tool_name
    handoff_tool.description = description

    return handoff_tool


def get_handoff_tools_for_agent(agent_type: str) -> list[BaseTool]:
    """Get all handoff tools available for a specific agent.

    Args:
        agent_type: The agent type (e.g., "task", "research")

    Returns:
        List of handoff tools the agent can use
    """
    allowed_targets = HANDOFF_MATRIX.get(agent_type, [])
    tools = [
        create_handoff_tool(source_agent=agent_type, target_agent=target)
        for target in allowed_targets
    ]

    logger.debug(
        "handoff_tools_created",
        agent=agent_type,
        targets=allowed_targets,
    )

    return tools


# =============================================================================
# Handoff Response Handling
# =============================================================================


def is_handoff_response(tool_result: Any) -> bool:
    """Check if a tool result indicates a handoff request.

    Args:
        tool_result: The result from a tool call

    Returns:
        True if this is a handoff request
    """
    if isinstance(tool_result, dict):
        return tool_result.get("handoff", False)
    return False


def parse_handoff_response(tool_result: dict) -> tuple[str, str, str]:
    """Parse a handoff response to extract target and task info.

    Args:
        tool_result: The handoff tool result dict

    Returns:
        Tuple of (target_agent, task_description, context)
    """
    return (
        tool_result.get("target_agent", ""),
        tool_result.get("task_description", ""),
        tool_result.get("context", ""),
    )


# =============================================================================
# Shared Memory Management
# =============================================================================

# SCAFFOLDING: Shared memory management is designed for cross-agent context
# transfer during handoffs. Currently unused while handoffs are disabled.


def _smart_truncate(text: str, max_chars: int) -> str:
    """Truncate text intelligently at sentence/paragraph boundaries.

    Tries to truncate at:
    1. Paragraph boundary
    2. Sentence boundary
    3. Word boundary

    Args:
        text: Text to truncate
        max_chars: Maximum characters

    Returns:
        Truncated text with ellipsis indicator
    """
    if len(text) <= max_chars:
        return text

    # Reserve space for truncation indicator
    max_chars -= 20

    # Try to find a paragraph boundary
    truncated = text[:max_chars]
    last_para = truncated.rfind("\n\n")
    if last_para > max_chars * 0.6:  # At least 60% of content
        return truncated[:last_para] + "\n\n[...truncated]"

    # Try to find a sentence boundary
    for end_marker in [". ", "! ", "? ", ".\n"]:
        last_sentence = truncated.rfind(end_marker)
        if last_sentence > max_chars * 0.5:  # At least 50% of content
            return truncated[: last_sentence + 1] + " [...truncated]"

    # Fall back to word boundary
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.4:
        return truncated[:last_space] + " [...truncated]"

    # Last resort - hard truncate
    return truncated + "...[truncated]"


def truncate_shared_memory(
    memory: SharedAgentMemory,
    budget: int = SHARED_MEMORY_TOTAL_BUDGET,
) -> SharedAgentMemory:
    """Dynamically truncate shared memory based on priority and budget.

    Uses priority weights to allocate more space to important content
    while ensuring all fields get at least a minimum allocation.

    Args:
        memory: The shared memory dict to truncate
        budget: Total character budget for all memory fields

    Returns:
        Truncated shared memory dict
    """
    if not memory:
        return {}

    # Calculate current sizes and identify string fields
    field_sizes: dict[str, int] = {}
    for key, value in memory.items():
        if isinstance(value, str):
            field_sizes[key] = len(value)
        elif isinstance(value, list):
            # For lists (sources, images), estimate size
            field_sizes[key] = len(str(value))

    total_size = sum(field_sizes.values())

    # If within budget, no truncation needed
    if total_size <= budget:
        return dict(memory)

    # Calculate budget allocation based on priorities
    total_priority = sum(SHARED_MEMORY_PRIORITIES.get(key, 1) for key in field_sizes.keys())

    allocations: dict[str, int] = {}
    for key in field_sizes.keys():
        priority = SHARED_MEMORY_PRIORITIES.get(key, 1)
        # Proportional allocation based on priority
        allocation = int((priority / total_priority) * budget)
        # Ensure minimum allocation
        allocations[key] = max(allocation, SHARED_MEMORY_MIN_CHARS)

    # Truncate each field
    truncated: SharedAgentMemory = {}
    for key, value in memory.items():
        if key not in allocations:
            truncated[key] = value
            continue

        max_chars = allocations[key]

        if isinstance(value, str):
            if len(value) > max_chars:
                # Smart truncation - try to end at sentence boundary
                truncated_value = _smart_truncate(value, max_chars)
                truncated[key] = truncated_value
                logger.debug(
                    "shared_memory_truncated",
                    field=key,
                    original_len=len(value),
                    truncated_len=len(truncated_value),
                    budget=max_chars,
                )
            else:
                truncated[key] = value
        elif isinstance(value, list):
            # For lists, truncate number of items
            truncated[key] = value[:10]  # Keep max 10 items
        else:
            truncated[key] = value

    return truncated


# =============================================================================
# Handoff Validation and Routing
# =============================================================================


def can_handoff(
    current_agent: str,
    target_agent: str,
    handoff_count: int,
    handoff_history: list[HandoffInfo] | None = None,
) -> bool:
    """Check if a handoff to the target agent is allowed.

    Args:
        current_agent: Current active agent
        target_agent: Target agent for handoff
        handoff_count: Current number of handoffs
        handoff_history: History of previous handoffs

    Returns:
        True if handoff is allowed
    """
    # Check max handoffs
    if handoff_count >= MAX_HANDOFFS:
        logger.warning(
            "max_handoffs_reached",
            count=handoff_count,
            max=MAX_HANDOFFS,
        )
        return False

    # Check if target is in allowed list
    allowed_targets = HANDOFF_MATRIX.get(current_agent, [])
    if target_agent not in allowed_targets:
        logger.warning(
            "handoff_not_allowed",
            source=current_agent,
            target=target_agent,
            allowed=allowed_targets,
        )
        return False

    # Prevent immediate back-and-forth (A -> B -> A)
    if handoff_history and len(handoff_history) >= 2:
        if handoff_history[-2].get("target_agent") == target_agent:
            logger.warning(
                "handoff_loop_detected",
                history=[h.get("target_agent") for h in handoff_history],
                target=target_agent,
            )
            return False

    return True


def update_handoff_history(
    history: list[HandoffInfo],
    source_agent: str,
    handoff: HandoffInfo,
) -> list[HandoffInfo]:
    """Create updated handoff history with new entry.

    Args:
        history: Current handoff history
        source_agent: Agent initiating the handoff
        handoff: Handoff info to add

    Returns:
        New history list with added entry
    """
    new_history = list(history)
    new_history.append(
        {
            "source_agent": source_agent,
            "target_agent": handoff.get("target_agent", ""),
            "task_description": handoff.get("task_description", ""),
            "context": handoff.get("context", ""),
        }
    )
    return new_history


def build_query_with_context(
    query: str,
    delegated_task: str | None = None,
    handoff_context: str | None = None,
    shared_memory: SharedAgentMemory | None = None,
    artifact_summary: str | None = None,
) -> str:
    """Build query string with handoff context if present.

    Uses dynamic truncation for shared memory to maximize relevant context
    while staying within reasonable limits.

    Args:
        query: Original query
        delegated_task: Task description from handoff
        handoff_context: Additional context from handoff
        shared_memory: Shared memory from previous agents
        artifact_summary: Summary of transferred sandbox artifacts

    Returns:
        Query string with optional context
    """
    result = query

    if delegated_task:
        result = delegated_task
        if handoff_context:
            result = f"{result}\n\nContext: {handoff_context}"

    if artifact_summary:
        result = f"{result}\n\n{artifact_summary}"

    # Include shared memory context if available (with dynamic truncation)
    if shared_memory:
        # Apply priority-based truncation
        truncated_memory = truncate_shared_memory(shared_memory)
        context_parts = []

        if truncated_memory.get("research_findings"):
            context_parts.append(
                f"Research findings from previous agent:\n{truncated_memory['research_findings']}"
            )
        if truncated_memory.get("research_sources"):
            sources = truncated_memory["research_sources"]
            if isinstance(sources, list) and sources:
                sources_text = "\n".join(
                    [
                        f"- [{s.get('title', 'Source')}]({s.get('url', '')}): {s.get('snippet', '')[:200]}"
                        for s in sources[:5]  # Limit to top 5 sources for context budget
                    ]
                )
                context_parts.append(f"Research sources:\n{sources_text}")
        if truncated_memory.get("generated_code"):
            code_lang = truncated_memory.get("code_language", "python")
            context_parts.append(
                f"Code from previous agent:\n```{code_lang}\n{truncated_memory['generated_code']}\n```"
            )
        if truncated_memory.get("execution_results"):
            context_parts.append(f"Execution results:\n{truncated_memory['execution_results']}")
        if truncated_memory.get("additional_context"):
            context_parts.append(truncated_memory["additional_context"])

        if context_parts:
            result = f"{result}\n\n---\nShared Context:\n" + "\n\n".join(context_parts)

    return result
