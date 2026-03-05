"""Agent tools for the multi-agent system."""

from app.agents.tools.browser_use import (
    browser_click,
    browser_dom_query,
    browser_get_accessibility_tree,
    browser_get_stream_url,
    browser_navigate,
    browser_press_key,
    browser_screenshot,
    browser_scroll,
    browser_type,
)
from app.agents.tools.code_execution import execute_code, execute_code_with_context
from app.agents.tools.codeact import execute_script
from app.agents.tools.complete_step import complete_step_tool
from app.agents.tools.database import execute_sql
from app.agents.tools.deployment import deploy_to_production
from app.agents.tools.file_tools import (
    file_find_by_name,
    file_find_in_content,
    file_read,
    file_str_replace,
    file_write,
)
from app.agents.tools.handoff import (
    AGENT_DESCRIPTIONS,
    # Constants
    HANDOFF_MATRIX,
    MAX_HANDOFFS,
    SHARED_MEMORY_MIN_CHARS,
    SHARED_MEMORY_PRIORITIES,
    SHARED_MEMORY_TOTAL_BUDGET,
    # Types
    HandoffInfo,
    HandoffInput,
    SharedAgentMemory,
    build_query_with_context,
    # Handoff validation and routing
    can_handoff,
    # Tool creation
    create_handoff_tool,
    get_handoff_tools_for_agent,
    # Response handling
    is_handoff_response,
    parse_handoff_response,
    # Shared memory management
    truncate_shared_memory,
    update_handoff_history,
)
from app.agents.tools.http_client import http_request
from app.agents.tools.image_generation import generate_image
from app.agents.tools.notification import send_notification
from app.agents.tools.react_tool import (
    ReActLoopConfig,
    ReActLoopResult,
    ToolExecutionError,
    build_ai_message_from_chunks,
    deduplicate_tool_messages,
    estimate_message_tokens,
    execute_react_loop,
    execute_tool_calls,
    execute_tool_with_retry,
    execute_with_retry,
    get_react_config,
    is_transient_error,
    truncate_messages_to_budget,
    truncate_tool_result,
)
from app.agents.tools.registry import (
    AGENT_TOOL_MAPPING,
    TOOL_CATALOG,
    ToolCategory,
    get_agent_categories,
    get_all_tools,
    get_tool_info,
    get_tool_names_for_agent,
    get_tools_by_category,
    get_tools_for_agent,
    register_tool,
    unregister_tool,
)
from app.agents.tools.slide_generation import generate_slides
from app.agents.tools.tool_pipeline import (
    CanonicalToolHooks,
    ResearchToolHooks,
    TaskToolHooks,
    ToolExecutionContext,
    ToolExecutionHooks,
    ToolExecutionResult,
    execute_tool,
    execute_tools_batch,
)
from app.agents.tools.validators import (
    FileOperationOutput,
    ValidationResult,
    extract_images,
    extract_search_sources,
    get_error_message,
    is_tool_error_response,
    validate_code_execution,
    validate_file_operation,
    validate_image_generation,
    validate_search_results,
    validate_tool_output,
)
from app.agents.tools.vision import analyze_image
from app.agents.tools.web_search import parse_search_results, web_extract_structured, web_search
from app.sandbox import (
    DesktopSandboxManager,
    DesktopSandboxSession,
    ExecutionSandboxManager,
    ExecutionSandboxSession,
    get_desktop_sandbox_manager,
    get_execution_sandbox_manager,
    sandbox_file,
    sandbox_file_with_context,
)

__all__ = [
    # Core tools
    "web_search",
    "web_extract_structured",
    "parse_search_results",
    "generate_image",
    "generate_slides",
    "analyze_image",
    # Browser tools (E2B Desktop sandbox)
    "browser_navigate",
    "browser_screenshot",
    "browser_click",
    "browser_type",
    "browser_press_key",
    "browser_scroll",
    "browser_dom_query",
    "browser_get_stream_url",
    "browser_get_accessibility_tree",
    # Code execution and sandbox tools
    "execute_script",
    "execute_code",
    "execute_code_with_context",
    "sandbox_file",
    "sandbox_file_with_context",
    # File management tools
    "file_read",
    "file_write",
    "file_str_replace",
    "file_find_by_name",
    "file_find_in_content",
    # Execution sandbox management
    "ExecutionSandboxSession",
    "ExecutionSandboxManager",
    "get_execution_sandbox_manager",
    # Desktop sandbox management
    "DesktopSandboxSession",
    "DesktopSandboxManager",
    "get_desktop_sandbox_manager",
    # Handoff types
    "HandoffInfo",
    "SharedAgentMemory",
    "HandoffInput",
    # Handoff constants
    "HANDOFF_MATRIX",
    "MAX_HANDOFFS",
    "SHARED_MEMORY_TOTAL_BUDGET",
    "SHARED_MEMORY_PRIORITIES",
    "SHARED_MEMORY_MIN_CHARS",
    "AGENT_DESCRIPTIONS",
    # Handoff tool creation
    "create_handoff_tool",
    "get_handoff_tools_for_agent",
    # Handoff response handling
    "is_handoff_response",
    "parse_handoff_response",
    # Shared memory management
    "truncate_shared_memory",
    # Handoff validation and routing
    "can_handoff",
    "update_handoff_history",
    "build_query_with_context",
    # Registry
    "ToolCategory",
    "TOOL_CATALOG",
    "AGENT_TOOL_MAPPING",
    "get_tools_by_category",
    "get_tools_for_agent",
    "get_tool_names_for_agent",
    "get_agent_categories",
    "register_tool",
    "unregister_tool",
    "get_all_tools",
    "get_tool_info",
    # Validators
    "ValidationResult",
    "FileOperationOutput",
    "validate_tool_output",
    "validate_search_results",
    "validate_image_generation",
    "validate_code_execution",
    "validate_file_operation",
    "extract_images",
    "extract_search_sources",
    "is_tool_error_response",
    "get_error_message",
    # HTTP client
    "http_request",
    # Database
    "execute_sql",
    # Notifications
    "send_notification",
    # Production deployment
    "deploy_to_production",
    # Execution utilities
    "ToolExecutionError",
    "execute_with_retry",
    "execute_tool_with_retry",
    "execute_tool_calls",
    "is_transient_error",
    "build_ai_message_from_chunks",
    "deduplicate_tool_messages",
    # ReAct loop utilities
    "ReActLoopConfig",
    "ReActLoopResult",
    "execute_react_loop",
    "get_react_config",
    "estimate_message_tokens",
    "truncate_messages_to_budget",
    "truncate_tool_result",
    # Plan step tracking
    "complete_step_tool",
    # Tool pipeline (shared execution infrastructure)
    "ToolExecutionContext",
    "ToolExecutionResult",
    "ToolExecutionHooks",
    "execute_tool",
    "execute_tools_batch",
    "TaskToolHooks",
    "ResearchToolHooks",
    "CanonicalToolHooks",
]
