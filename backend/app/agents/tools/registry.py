"""Tool Registry with Categorization.

This module provides a centralized registry for all agent tools,
organized by category. Agents can request tools by category to get
only the tools relevant to their specialization.
"""

from enum import Enum
from typing import Any

from langchain_core.tools import BaseTool

from app.agents.state import AgentType
from app.agents.policy.contracts import (
    CapabilityContract,
    DataSensitivity,
    NetworkScope,
    SideEffectLevel,
)
from app.agents.tools.app_builder import get_app_builder_tools
from app.agents.tools.codeact import execute_script
from app.agents.tools.browser_use import (
    browser_click,
    browser_console_exec,
    browser_dom_query,
    browser_get_accessibility_tree,
    browser_get_stream_url,
    browser_navigate,
    browser_press_key,
    browser_screenshot,
    browser_scroll,
    browser_select_option,
    browser_type,
    browser_wait_for_element,
)
from app.agents.tools.code_execution import execute_code
from app.agents.tools.database import execute_sql
from app.agents.tools.deployment import deploy_expose_port, deploy_get_url, deploy_to_production
from app.agents.tools.file_tools import (
    file_find_by_name,
    file_find_in_content,
    file_read,
    file_str_replace,
    file_write,
)
from app.agents.tools.handoff import get_handoff_tools_for_agent
from app.agents.tools.http_client import http_request
from app.agents.tools.notification import send_notification
from app.agents.tools.shell_tools import (
    shell_exec,
    shell_kill,
    shell_view,
    shell_wait,
)
from app.agents.tools.hitl_tool import ask_user_tool
from app.agents.tools.image_generation import generate_image
from app.agents.tools.skill_invocation import get_skill_tools
from app.agents.tools.slide_generation import generate_slides
from app.agents.tools.tool_search import search_tools
from app.agents.tools.vision import analyze_image
from app.agents.tools.web_search import web_extract_structured, web_search
from app.core.logging import get_logger
from app.sandbox import sandbox_file

logger = get_logger(__name__)


class ToolCategory(str, Enum):
    """Categories of tools available to agents."""

    SEARCH = "search"  # Web search capabilities
    IMAGE = "image"  # Image generation and analysis
    BROWSER = "browser"  # Browser automation (E2B Desktop sandbox)
    CODE_EXEC = "code_exec"  # Code execution in sandbox
    DATA = "data"  # Data processing and analysis
    APP_BUILDER = "app_builder"  # App building and running tools
    HANDOFF = "handoff"  # Agent-to-agent delegation
    SKILL = "skill"  # Skill invocation
    SLIDES = "slides"  # Slide/PPTX generation
    FILE_OPS = "file_ops"  # File management (read, write, search, edit)
    SHELL = "shell"  # Shell/process management (exec, view, wait, kill)
    HITL = "hitl"  # Human-in-the-loop tools (ask user for input/decisions)
    DEPLOY = "deploy"  # Deployment and port exposure
    HTTP_CLIENT = "http_client"  # HTTP API client for external requests
    DATABASE = "database"  # Database query execution
    NOTIFICATION = "notification"  # Notifications and webhooks
    TOOL_SEARCH = "tool_search"  # Meta-tool for discovering tools
    MCP = "mcp"  # MCP (Model Context Protocol) tools from external servers
    CODEACT = "codeact"  # CodeAct hybrid execution (multi-line Python with helpers)


# Tool instances by category
# Note: Some tools may belong to multiple categories
TOOL_CATALOG: dict[ToolCategory, list[BaseTool]] = {
    ToolCategory.SEARCH: [web_search, web_extract_structured],
    ToolCategory.IMAGE: [generate_image, analyze_image],
    ToolCategory.SLIDES: [generate_slides],
    ToolCategory.BROWSER: [
        browser_navigate,
        browser_screenshot,
        browser_click,
        browser_type,
        browser_press_key,
        browser_scroll,
        browser_get_stream_url,
        browser_console_exec,
        browser_select_option,
        browser_wait_for_element,
        browser_dom_query,
        browser_get_accessibility_tree,
    ],
    ToolCategory.CODE_EXEC: [execute_code],
    ToolCategory.FILE_OPS: [
        file_read,
        file_write,
        file_str_replace,
        file_find_by_name,
        file_find_in_content,
    ],
    ToolCategory.SHELL: [
        shell_exec,
        shell_view,
        shell_wait,
        shell_kill,
    ],
    ToolCategory.DATA: [sandbox_file],
    ToolCategory.APP_BUILDER: get_app_builder_tools(),
    # HANDOFF tools are created dynamically per agent
    ToolCategory.HANDOFF: [],
    # SKILL tools for invoking skills
    ToolCategory.SKILL: get_skill_tools(),
    # HITL tools for asking user for input/decisions
    ToolCategory.HITL: [ask_user_tool],
    # Deployment and port exposure tools
    ToolCategory.DEPLOY: [deploy_expose_port, deploy_get_url, deploy_to_production],
    # HTTP API client
    ToolCategory.HTTP_CLIENT: [http_request],
    # Database query execution
    ToolCategory.DATABASE: [execute_sql],
    # Notifications and webhooks
    ToolCategory.NOTIFICATION: [send_notification],
    # Meta-tool for discovering available tools
    ToolCategory.TOOL_SEARCH: [search_tools],
    # MCP tools - dynamically populated via register_tool()
    ToolCategory.MCP: [],
    # CodeAct - multi-line Python execution with helper library
    ToolCategory.CODEACT: [execute_script],
}


# Capability contracts used by policy engine and startup validation.
TOOL_CONTRACTS: dict[str, CapabilityContract] = {
    "web_search": CapabilityContract(SideEffectLevel.NONE, DataSensitivity.PUBLIC, NetworkScope.EXTERNAL, True),
    "web_extract_structured": CapabilityContract(
        SideEffectLevel.NONE, DataSensitivity.PUBLIC, NetworkScope.EXTERNAL, True
    ),
    "generate_image": CapabilityContract(
        SideEffectLevel.LOW, DataSensitivity.INTERNAL, NetworkScope.EXTERNAL, False
    ),
    "analyze_image": CapabilityContract(
        SideEffectLevel.NONE, DataSensitivity.SENSITIVE, NetworkScope.NONE, True
    ),
    "generate_slides": CapabilityContract(
        SideEffectLevel.MEDIUM, DataSensitivity.INTERNAL, NetworkScope.EXTERNAL, False
    ),
    "execute_code": CapabilityContract(
        SideEffectLevel.HIGH, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, False
    ),
    "sandbox_file": CapabilityContract(
        SideEffectLevel.HIGH, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, False
    ),
    "file_read": CapabilityContract(
        SideEffectLevel.NONE, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, True
    ),
    "file_write": CapabilityContract(
        SideEffectLevel.HIGH, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, False
    ),
    "file_str_replace": CapabilityContract(
        SideEffectLevel.MEDIUM, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, False
    ),
    "file_find_by_name": CapabilityContract(
        SideEffectLevel.NONE, DataSensitivity.INTERNAL, NetworkScope.SANDBOX_ONLY, True
    ),
    "file_find_in_content": CapabilityContract(
        SideEffectLevel.NONE, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, True
    ),
    "shell_exec": CapabilityContract(
        SideEffectLevel.HIGH, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, False
    ),
    "shell_view": CapabilityContract(
        SideEffectLevel.NONE, DataSensitivity.INTERNAL, NetworkScope.SANDBOX_ONLY, True
    ),
    "shell_wait": CapabilityContract(
        SideEffectLevel.NONE, DataSensitivity.INTERNAL, NetworkScope.SANDBOX_ONLY, True
    ),
    "shell_kill": CapabilityContract(
        SideEffectLevel.MEDIUM, DataSensitivity.INTERNAL, NetworkScope.SANDBOX_ONLY, False
    ),
    "invoke_skill": CapabilityContract(
        SideEffectLevel.MEDIUM, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, False
    ),
    "list_skills": CapabilityContract(
        SideEffectLevel.NONE, DataSensitivity.INTERNAL, NetworkScope.NONE, True
    ),
    "ask_user": CapabilityContract(SideEffectLevel.NONE, DataSensitivity.INTERNAL, NetworkScope.NONE, True),
    "deploy_expose_port": CapabilityContract(
        SideEffectLevel.HIGH, DataSensitivity.SENSITIVE, NetworkScope.EXTERNAL, False
    ),
    "deploy_get_url": CapabilityContract(
        SideEffectLevel.NONE, DataSensitivity.INTERNAL, NetworkScope.EXTERNAL, True
    ),
    "deploy_to_production": CapabilityContract(
        SideEffectLevel.HIGH, DataSensitivity.SENSITIVE, NetworkScope.EXTERNAL, False
    ),
    "http_request": CapabilityContract(
        SideEffectLevel.MEDIUM, DataSensitivity.INTERNAL, NetworkScope.EXTERNAL, False
    ),
    "execute_sql": CapabilityContract(
        SideEffectLevel.HIGH, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, False
    ),
    "send_notification": CapabilityContract(
        SideEffectLevel.MEDIUM, DataSensitivity.INTERNAL, NetworkScope.EXTERNAL, False
    ),
    "search_tools": CapabilityContract(
        SideEffectLevel.NONE, DataSensitivity.INTERNAL, NetworkScope.NONE, True
    ),
    "create_app_project": CapabilityContract(
        SideEffectLevel.HIGH, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, False
    ),
    "app_list_files": CapabilityContract(
        SideEffectLevel.NONE, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, True
    ),
    "app_read_file": CapabilityContract(
        SideEffectLevel.NONE, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, True
    ),
    "app_write_file": CapabilityContract(
        SideEffectLevel.HIGH, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, False
    ),
    "app_run_command": CapabilityContract(
        SideEffectLevel.HIGH, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, False
    ),
    "app_install_packages": CapabilityContract(
        SideEffectLevel.HIGH, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, False
    ),
    "app_start_server": CapabilityContract(
        SideEffectLevel.HIGH, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, False
    ),
    "app_stop_server": CapabilityContract(
        SideEffectLevel.MEDIUM, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, False
    ),
    "app_get_preview_url": CapabilityContract(
        SideEffectLevel.NONE, DataSensitivity.INTERNAL, NetworkScope.EXTERNAL, True
    ),
    "execute_script": CapabilityContract(
        SideEffectLevel.HIGH, DataSensitivity.SENSITIVE, NetworkScope.SANDBOX_ONLY, False
    ),
}

# Browser tools are all high-impact.
for _browser_tool_name in (
    "browser_navigate",
    "browser_screenshot",
    "browser_click",
    "browser_type",
    "browser_press_key",
    "browser_scroll",
    "browser_get_stream_url",
    "browser_console_exec",
    "browser_select_option",
    "browser_wait_for_element",
    "browser_dom_query",
    "browser_get_accessibility_tree",
):
    TOOL_CONTRACTS[_browser_tool_name] = CapabilityContract(
        SideEffectLevel.HIGH,
        DataSensitivity.SENSITIVE,
        NetworkScope.EXTERNAL,
        False,
    )


# Define which tool categories each agent type can access
AGENT_TOOL_MAPPING: dict[str, list[ToolCategory]] = {
    AgentType.TASK.value: [
        ToolCategory.SEARCH,
        ToolCategory.IMAGE,
        ToolCategory.SLIDES,
        ToolCategory.BROWSER,
        ToolCategory.CODE_EXEC,  # For execute_code tool
        ToolCategory.FILE_OPS,  # For file management (read, write, search, edit)
        ToolCategory.SHELL,  # For shell/process management
        ToolCategory.APP_BUILDER,  # For building and running apps
        ToolCategory.SKILL,
        ToolCategory.HANDOFF,
        ToolCategory.HITL,  # For asking user for input/decisions
        ToolCategory.DEPLOY,  # For exposing sandbox ports and production deployment
        ToolCategory.HTTP_CLIENT,  # For calling external APIs
        ToolCategory.DATABASE,  # For querying databases
        ToolCategory.NOTIFICATION,  # For sending notifications/webhooks
        ToolCategory.TOOL_SEARCH,  # For discovering tools on-demand
        ToolCategory.MCP,  # MCP tools from external servers
    ],
    AgentType.RESEARCH.value: [
        ToolCategory.SEARCH,
        ToolCategory.IMAGE,
        ToolCategory.BROWSER,
        ToolCategory.SKILL,
        ToolCategory.HANDOFF,
        ToolCategory.HITL,  # For asking user for input/decisions
    ],
}


def get_tools_by_category(category: ToolCategory) -> list[BaseTool]:
    """Get all tools in a specific category.

    Args:
        category: The tool category to retrieve

    Returns:
        List of tools in that category
    """
    return TOOL_CATALOG.get(category, []).copy()


def get_tools_for_agent(
    agent_type: str,
    include_handoffs: bool = True,
    execution_mode: str | None = None,
) -> list[BaseTool]:
    """Get all tools available to a specific agent type.

    This function retrieves tools based on the agent's allowed categories.
    All tools are enabled - the LLM decides when to use them.

    Args:
        agent_type: The agent type (e.g., "task", "research")
        include_handoffs: Whether to include handoff tools
        execution_mode: Optional execution mode; when "codeact", includes
            the CodeAct execute_script tool for the task agent

    Returns:
        List of tools available to the agent
    """
    allowed_categories = list(AGENT_TOOL_MAPPING.get(agent_type, []))

    # Add CodeAct category when execution_mode is "codeact" for the task agent
    if (
        execution_mode == "codeact"
        and agent_type == AgentType.TASK.value
        and ToolCategory.CODEACT not in allowed_categories
    ):
        allowed_categories.append(ToolCategory.CODEACT)

    tools: list[BaseTool] = []
    seen_names: set[str] = set()

    def add_tools(tool_list: list[BaseTool]) -> None:
        """Add tools to the result list, avoiding duplicates."""
        for tool in tool_list:
            if tool.name not in seen_names:
                tools.append(tool)
                seen_names.add(tool.name)

    # Add all tools from allowed categories
    for category in allowed_categories:
        # Skip handoff category (handled separately)
        if category == ToolCategory.HANDOFF:
            continue
        add_tools(TOOL_CATALOG.get(category, []))

    # Add handoff tools if enabled
    if include_handoffs and ToolCategory.HANDOFF in allowed_categories:
        add_tools(get_handoff_tools_for_agent(agent_type))

    logger.debug(
        "tools_retrieved_for_agent",
        agent=agent_type,
        tool_count=len(tools),
        tools=[t.name for t in tools],
    )

    return tools


def get_tool_names_for_agent(agent_type: str) -> list[str]:
    """Get names of all tools available to an agent.

    Useful for logging and debugging without instantiating tools.

    Args:
        agent_type: The agent type

    Returns:
        List of tool names
    """
    tools = get_tools_for_agent(agent_type, include_handoffs=False)
    return [tool.name for tool in tools]


def get_agent_categories(agent_type: str) -> list[ToolCategory]:
    """Get the tool categories available to an agent.

    Args:
        agent_type: The agent type

    Returns:
        List of tool categories the agent can access
    """
    return AGENT_TOOL_MAPPING.get(agent_type, []).copy()


def register_tool(category: ToolCategory, tool: BaseTool) -> None:
    """Register a new tool in a category.

    Use this to dynamically add tools (e.g., custom code execution tools)
    at runtime.

    Args:
        category: The category to register the tool in
        tool: The tool to register
    """
    if category not in TOOL_CATALOG:
        TOOL_CATALOG[category] = []

    # Avoid duplicates
    existing_names = {t.name for t in TOOL_CATALOG[category]}
    if tool.name not in existing_names:
        TOOL_CATALOG[category].append(tool)
        if tool.name not in TOOL_CONTRACTS:
            # Dynamic MCP tools default to medium-risk external contract unless explicitly set.
            TOOL_CONTRACTS[tool.name] = CapabilityContract(
                SideEffectLevel.MEDIUM,
                DataSensitivity.INTERNAL,
                NetworkScope.EXTERNAL,
                False,
            )
        logger.info(
            "tool_registered",
            category=category.value,
            tool=tool.name,
        )


def unregister_tool(category: ToolCategory, tool_name: str) -> bool:
    """Remove a tool from a category.

    Args:
        category: The category to remove from
        tool_name: Name of the tool to remove

    Returns:
        True if tool was removed, False if not found
    """
    if category not in TOOL_CATALOG:
        return False

    original_len = len(TOOL_CATALOG[category])
    TOOL_CATALOG[category] = [t for t in TOOL_CATALOG[category] if t.name != tool_name]

    if len(TOOL_CATALOG[category]) < original_len:
        logger.info(
            "tool_unregistered",
            category=category.value,
            tool=tool_name,
        )
        return True

    return False


def get_all_tools() -> list[BaseTool]:
    """Get all registered tools across all categories.

    Returns:
        List of all unique tools
    """
    seen_names: set[str] = set()
    tools: list[BaseTool] = []

    for category_tools in TOOL_CATALOG.values():
        for tool in category_tools:
            if tool.name not in seen_names:
                tools.append(tool)
                seen_names.add(tool.name)

    return tools


def get_tool_contract(tool_name: str) -> CapabilityContract | None:
    """Get capability contract metadata for a tool."""
    return TOOL_CONTRACTS.get(tool_name)


def validate_tool_contracts() -> None:
    """Fail loudly if any registered tool is missing a capability contract."""
    missing = sorted(
        tool.name
        for tool in get_all_tools()
        if tool.name not in TOOL_CONTRACTS
    )
    if missing:
        raise RuntimeError(f"Missing capability contracts for tools: {missing}")


# Soft-disabled tools: instead of removing from the schema (which
# invalidates the KV-cache prefix), we keep the tool definition and
# inject a system message telling the model not to use it.
_soft_disabled_tools: set[str] = set()


def soft_disable_tool(tool_name: str) -> None:
    """Mark a tool as soft-disabled.

    The tool schema remains in the prefix for KV-cache stability,
    but a system message is injected advising the model not to use it.

    Args:
        tool_name: Name of the tool to soft-disable
    """
    _soft_disabled_tools.add(tool_name)
    logger.info("tool_soft_disabled", tool=tool_name)


def soft_enable_tool(tool_name: str) -> None:
    """Re-enable a previously soft-disabled tool.

    Args:
        tool_name: Name of the tool to re-enable
    """
    _soft_disabled_tools.discard(tool_name)
    logger.info("tool_soft_enabled", tool=tool_name)


def get_soft_disabled_tools() -> set[str]:
    """Return the set of currently soft-disabled tool names."""
    return _soft_disabled_tools.copy()


def get_soft_disabled_message(disabled_tools: list[str] | set[str] | None = None) -> str | None:
    """Build a system message listing soft-disabled tools.

    Args:
        disabled_tools: Explicit list of tool names; defaults to _soft_disabled_tools

    Returns:
        A message string, or None if no tools are disabled
    """
    tools = disabled_tools if disabled_tools is not None else _soft_disabled_tools
    if not tools:
        return None
    tool_list = ", ".join(sorted(tools))
    return (
        f"[Tool Availability Notice] The following tools are currently unavailable "
        f"and must NOT be called: {tool_list}. "
        f"Use alternative tools or approaches instead."
    )


def get_tool_info() -> dict[str, Any]:
    """Get information about the tool registry for debugging.

    Returns:
        Dict with registry state information
    """
    return {
        "categories": {cat.value: [t.name for t in tools] for cat, tools in TOOL_CATALOG.items()},
        "agent_mappings": {
            agent: [cat.value for cat in cats] for agent, cats in AGENT_TOOL_MAPPING.items()
        },
        "total_tools": len(get_all_tools()),
    }
