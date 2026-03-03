"""Utilities for ReAct-style tool invocation with streamed LLM responses.

This module provides:
- Tool call normalization from streamed chunks
- Retry logic with exponential backoff for transient failures
- Standardized tool execution helpers
- Unified ReAct loop implementation for all agents
"""

import asyncio
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.ai.llm import extract_text_from_content
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# Transient error types that should trigger retries
TRANSIENT_ERRORS = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)


class ErrorCategory(str, Enum):
    """Categories of tool errors for recovery strategy selection."""

    TRANSIENT = "transient"  # Network/timeout - auto-retry
    INPUT = "input"  # Bad arguments - modify and retry
    PERMISSION = "permission"  # Access denied - inform user
    RESOURCE = "resource"  # Not found - try alternative
    FATAL = "fatal"  # Unrecoverable - stop
    UNKNOWN = "unknown"  # Uncategorized


def classify_error(error_msg: str, tool_name: str = "") -> ErrorCategory:
    """Classify a tool error for recovery strategy selection.

    Args:
        error_msg: The error message string
        tool_name: Name of the tool that failed

    Returns:
        ErrorCategory indicating the type of error
    """
    if not error_msg:
        return ErrorCategory.UNKNOWN

    error_lower = error_msg.lower()

    # Transient errors (retry-worthy)
    transient_patterns = [
        "timeout",
        "timed out",
        "connection",
        "network",
        "temporarily unavailable",
        "rate limit",
        "429",
        "503",
        "econnrefused",
        "econnreset",
        "socket",
    ]
    if any(p in error_lower for p in transient_patterns):
        return ErrorCategory.TRANSIENT

    # Input errors (fix args and retry)
    input_patterns = [
        "invalid",
        "missing required",
        "validation error",
        "bad request",
        "400",
        "type error",
        "typeerror",
        "syntax error",
        "syntaxerror",
        "json",
        "parse error",
        "expected",
        "must be",
        "cannot be empty",
    ]
    if any(p in error_lower for p in input_patterns):
        return ErrorCategory.INPUT

    # Permission errors (inform user)
    permission_patterns = [
        "permission denied",
        "access denied",
        "forbidden",
        "403",
        "unauthorized",
        "401",
        "not allowed",
        "authentication",
    ]
    if any(p in error_lower for p in permission_patterns):
        return ErrorCategory.PERMISSION

    # Resource errors (try alternative)
    resource_patterns = [
        "not found",
        "404",
        "no such file",
        "does not exist",
        "filenotfounderror",
        "modulenotfounderror",
        "no module named",
        "command not found",
        "no results",
    ]
    if any(p in error_lower for p in resource_patterns):
        return ErrorCategory.RESOURCE

    # Fatal errors (stop)
    fatal_patterns = [
        "out of memory",
        "oom",
        "disk full",
        "quota exceeded",
        "segmentation fault",
        "killed",
        "fatal",
    ]
    if any(p in error_lower for p in fatal_patterns):
        return ErrorCategory.FATAL

    return ErrorCategory.UNKNOWN


class ToolExecutionError(Exception):
    """Error during tool execution."""

    def __init__(
        self,
        tool_name: str,
        message: str,
        is_transient: bool = False,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.tool_name = tool_name
        self.is_transient = is_transient
        self.original_error = original_error


async def execute_with_retry(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    **kwargs: Any,
) -> T:
    """Execute an async function with exponential backoff retry.

    Uses exponential backoff with optional jitter to handle transient failures.
    Only retries on transient errors (connection, timeout), not on validation
    or permission errors.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts (0 = no retries)
        base_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        **kwargs: Keyword arguments for func

    Returns:
        Result from successful function execution

    Raises:
        Exception: The last exception if all retries fail
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except TRANSIENT_ERRORS as e:
            last_exception = e

            if attempt >= max_retries:
                logger.warning(
                    "retry_exhausted",
                    func=func.__name__,
                    attempts=attempt + 1,
                    error=str(e),
                )
                raise

            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base ** attempt), max_delay)

            # Add jitter to prevent thundering herd
            if jitter:
                delay = delay * (0.5 + random.random())

            logger.info(
                "retrying_after_transient_error",
                func=func.__name__,
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=round(delay, 2),
                error=str(e),
            )

            await asyncio.sleep(delay)

        except Exception as e:
            # Non-transient errors are not retried
            logger.debug(
                "non_transient_error_not_retrying",
                func=func.__name__,
                error_type=type(e).__name__,
                error=str(e),
            )
            raise

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected retry loop exit")


async def execute_tool_with_retry(
    tool: BaseTool,
    tool_input: dict[str, Any],
    max_retries: int = 2,
    base_delay: float = 1.0,
) -> str:
    """Execute a LangChain tool with retry on transient failures.

    Args:
        tool: LangChain BaseTool to execute
        tool_input: Input dictionary for the tool
        max_retries: Maximum retry attempts
        base_delay: Base delay for exponential backoff

    Returns:
        Tool execution result as string

    Raises:
        ToolExecutionError: If execution fails after all retries
    """
    # Log tool invocation in debug mode
    logger.debug(
        "tool_invoked",
        tool_name=tool.name,
        tool_args={k: v for k, v in tool_input.items() if k not in ["user_id", "task_id"]},  # Exclude sensitive context
    )
    
    try:
        result = await execute_with_retry(
            tool.ainvoke,
            tool_input,
            max_retries=max_retries,
            base_delay=base_delay,
        )
        result_str = str(result) if result is not None else ""
        
        # Log tool completion in debug mode
        logger.debug(
            "tool_completed",
            tool_name=tool.name,
            result_length=len(result_str),
            result_preview=result_str[:200] if result_str else None,
        )
        
        return result_str

    except TRANSIENT_ERRORS as e:
        raise ToolExecutionError(
            tool_name=tool.name,
            message=f"Tool {tool.name} failed after {max_retries} retries: {e}",
            is_transient=True,
            original_error=e,
        )
    except Exception as e:
        raise ToolExecutionError(
            tool_name=tool.name,
            message=f"Tool {tool.name} execution error: {e}",
            is_transient=False,
            original_error=e,
        )


async def execute_tool_calls(
    ai_message: AIMessage,
    tools: list[BaseTool],
    max_retries: int = 2,
    on_tool_start: Callable[[str, dict], None] | None = None,
    on_tool_end: Callable[[str, str], None] | None = None,
    on_tool_error: Callable[[str, Exception], None] | None = None,
) -> list[ToolMessage]:
    """Execute all tool calls from an AI message.

    Processes tool calls sequentially, with retry logic for each.
    Provides hooks for monitoring tool execution progress.

    Args:
        ai_message: AIMessage containing tool_calls
        tools: List of available tools
        max_retries: Maximum retries per tool
        on_tool_start: Callback when tool execution starts (tool_name, args)
        on_tool_end: Callback when tool execution ends (tool_name, result)
        on_tool_error: Callback on tool error (tool_name, exception)

    Returns:
        List of ToolMessage results
    """
    if not ai_message.tool_calls:
        return []

    # Build tool lookup
    tool_map = {tool.name: tool for tool in tools}
    results: list[ToolMessage] = []

    for tool_call in ai_message.tool_calls:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")

        # Find tool
        tool = tool_map.get(tool_name)
        if not tool:
            error_msg = f"Tool not found: {tool_name}"
            logger.warning("tool_not_found", tool_name=tool_name)
            results.append(
                ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )
            continue

        # Notify start
        if on_tool_start:
            on_tool_start(tool_name, tool_args)

        try:
            # Execute with retry
            result = await execute_tool_with_retry(
                tool,
                tool_args,
                max_retries=max_retries,
            )

            # Notify end
            if on_tool_end:
                on_tool_end(tool_name, result[:500])

            results.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )

        except Exception as e:
            error_msg = f"Error invoking tool {tool_name}: {e}"
            is_transient = isinstance(e, ToolExecutionError) and e.is_transient
            log_event = "tool_execution_failed" if isinstance(e, ToolExecutionError) else "unexpected_tool_error"
            logger.error(log_event, tool=tool_name, error=str(e), is_transient=is_transient)

            if on_tool_error:
                on_tool_error(tool_name, e)

            results.append(
                ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )

    return results


def is_transient_error(error: Exception) -> bool:
    """Check if an error is transient and should trigger retry.

    Args:
        error: Exception to check

    Returns:
        True if the error is transient
    """
    return isinstance(error, TRANSIENT_ERRORS)


def _merge_tool_call_chunks(tool_calls: list[dict]) -> list[dict]:
    """Merge tool call chunks by index or ID.

    When streaming, tool calls arrive in multiple chunks that need to be merged.
    Each chunk may have partial data (e.g., first chunk has name, second has args).

    IMPORTANT: During streaming, args often arrive as string fragments (JSON pieces)
    that need to be concatenated and then parsed. This function handles both:
    - Dict args (non-streaming or already parsed)
    - String args (streaming JSON fragments)

    Args:
        tool_calls: List of tool call dicts from chunks

    Returns:
        List of merged tool call dicts
    """
    if not tool_calls:
        return []

    # Group by index first (streaming chunks use index), then by id
    merged: dict[int | str, dict] = {}
    # Track string args separately for concatenation
    string_args: dict[int | str, str] = {}
    # Track partial data for recovery
    partial_data: dict[int | str, list[dict]] = {}

    for tc in tool_calls:
        try:
            # Validate chunk structure
            if not isinstance(tc, dict):
                logger.warning(
                    "invalid_tool_call_chunk_type",
                    chunk_type=type(tc).__name__,
                )
                continue

            # Get the index or id for grouping
            index = tc.get("index")
            tc_id = tc.get("id") or tc.get("tool_call_id")

            # Use index as key if available (for streaming), otherwise use id
            key = index if index is not None else tc_id

            if key is None:
                # No way to merge, treat as standalone
                # Generate a unique key
                key = f"standalone_{len(merged)}"

            if key not in merged:
                merged[key] = {
                    "id": tc_id or "",
                    "name": tc.get("name") or tc.get("tool") or "",
                    "args": {},
                }
                string_args[key] = ""
                partial_data[key] = []

            # Store raw chunk for recovery
            partial_data[key].append(tc)

            # Merge fields
            existing = merged[key]

            # Update id if we have one
            if tc_id and not existing["id"]:
                existing["id"] = tc_id

            # Update name if we have one (support nested "function" format)
            func = tc.get("function") if isinstance(tc.get("function"), dict) else None
            name = tc.get("name") or tc.get("tool") or (func.get("name") if func else None)
            if name:
                existing["name"] = name

            # Merge args - handle both dict and string (streaming) formats
            args = tc.get("args")
            if args is None:
                args = tc.get("arguments")
            if args is None:
                args = tc.get("input")
            if args is None and func:
                args = func.get("arguments") or func.get("args") or func.get("input")
            if args is not None:
                if isinstance(args, dict):
                    # Dict args - merge directly
                    existing["args"].update(args)
                elif isinstance(args, str):
                    # String args (streaming) - concatenate
                    string_args[key] += args
                else:
                    logger.warning(
                        "unexpected_args_type",
                        args_type=type(args).__name__,
                        key=key,
                    )

        except Exception as e:
            logger.error(
                "tool_call_chunk_merge_error",
                error=str(e),
                chunk=str(tc)[:200] if tc else "None",
            )
            continue

    # Parse concatenated string args into dicts
    for key, args_str in string_args.items():
        if args_str:
            try:
                parsed_args = json.loads(args_str)
                if isinstance(parsed_args, dict):
                    merged[key]["args"].update(parsed_args)
                else:
                    logger.warning(
                        "parsed_args_not_dict",
                        args_str=args_str[:100],
                        parsed_type=type(parsed_args).__name__,
                    )
            except json.JSONDecodeError as e:
                # Try to recover partial JSON
                recovered_args = _recover_partial_json(args_str)
                if recovered_args:
                    merged[key]["args"].update(recovered_args)
                    logger.info(
                        "recovered_partial_json_args",
                        key=key,
                        recovered_keys=list(recovered_args.keys()),
                    )
                else:
                    logger.warning(
                        "failed_to_parse_args_json",
                        args_str=args_str[:200],
                        error=str(e),
                        chunk_count=len(partial_data.get(key, [])),
                    )

    result = list(merged.values())
    logger.debug(
        "tool_call_chunks_merged",
        input_count=len(tool_calls),
        output_count=len(result),
        tool_names=[tc.get("name", "") for tc in result],
    )
    return result


def _recover_partial_json(json_str: str) -> dict[str, Any] | None:
    """Attempt to recover a valid dict from partial/malformed JSON.

    Tries several strategies:
    1. Truncate at the last valid point
    2. Add missing closing braces
    3. Extract individual key-value pairs

    Args:
        json_str: Potentially malformed JSON string

    Returns:
        Recovered dict or None if recovery failed
    """
    if not json_str:
        return None

    # Strategy 1: Try adding missing closing braces
    test_str = json_str.rstrip()
    for _ in range(5):  # Max 5 closing braces
        test_str += "}"
        try:
            result = json.loads(test_str)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    # Strategy 2: Try truncating at the last valid comma or closing brace
    for end_char in [",", "}", '"']:
        last_pos = json_str.rfind(end_char)
        if last_pos > 0:
            test_str = json_str[:last_pos]
            if end_char == ",":
                test_str += "}"
            elif end_char == '"':
                test_str += "\"}"
            try:
                result = json.loads(test_str)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue

    # Strategy 3: Extract individual quoted strings that look like key-value pairs
    import re
    try:
        # Match patterns like "key": "value" or "key": 123
        pattern = r'"([^"]+)"\s*:\s*("([^"\\]|\\.)*"|[-\d.]+|true|false|null)'
        matches = re.findall(pattern, json_str)
        if matches:
            recovered = {}
            for match in matches:
                key = match[0]
                value_str = match[1]
                try:
                    recovered[key] = json.loads(value_str)
                except json.JSONDecodeError:
                    recovered[key] = value_str.strip('"')
            if recovered:
                return recovered
    except Exception:
        pass

    return None


def build_ai_message_from_chunks(response_chunks: list, query: str) -> AIMessage:
    """Build an AIMessage from streamed chunks with normalized tool calls.

    Handles both `tool_calls` and `tool_call_chunks` attributes:
    - `tool_calls`: Fully formed tool calls (non-streaming or parsed)
    - `tool_call_chunks`: Incremental streaming chunks with partial data
    """
    if not response_chunks:
        return AIMessage(content="")

    full_content = ""
    all_tool_calls: list[dict] = []
    # Track IDs from fully-formed tool_calls to avoid duplicates from tool_call_chunks
    seen_tool_call_ids: set[str] = set()

    for chunk in response_chunks:
        if hasattr(chunk, "content") and chunk.content:
            full_content += extract_text_from_content(chunk.content)

        # Handle fully formed tool_calls (preferred over tool_call_chunks)
        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            for tc in chunk.tool_calls:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    seen_tool_call_ids.add(tc_id)
                all_tool_calls.append(tc if isinstance(tc, dict) else tc.model_dump() if hasattr(tc, "model_dump") else tc)

        # Handle streaming tool_call_chunks (incremental format)
        # Skip if we already have a fully-formed tool_call with this ID
        if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
            for tc_chunk in chunk.tool_call_chunks:
                # Check if we already have this tool call from tool_calls
                tc_id = tc_chunk.get("id") if isinstance(tc_chunk, dict) else getattr(tc_chunk, "id", None)
                if tc_id and tc_id in seen_tool_call_ids:
                    continue  # Skip duplicate from tool_call_chunks

                # Convert ToolCallChunk to dict for unified processing
                if hasattr(tc_chunk, "model_dump"):
                    all_tool_calls.append(tc_chunk.model_dump())
                elif isinstance(tc_chunk, dict):
                    all_tool_calls.append(tc_chunk)
                else:
                    # Handle object-like chunks
                    all_tool_calls.append({
                        "index": getattr(tc_chunk, "index", None),
                        "id": getattr(tc_chunk, "id", None),
                        "name": getattr(tc_chunk, "name", None),
                        "args": getattr(tc_chunk, "args", None),
                    })

    # Log raw tool calls for debugging
    if all_tool_calls:
        logger.debug(
            "raw_tool_calls_collected",
            count=len(all_tool_calls),
            chunk_count=len(response_chunks),
        )

    # Merge tool call chunks (streaming sends partial data across chunks)
    merged_tool_calls = _merge_tool_call_chunks(all_tool_calls)

    normalized_tool_calls = []
    for tool_call in merged_tool_calls:
        func = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else None
        tool_name = tool_call.get("name") or tool_call.get("tool") or (func.get("name") if func else "") or ""
        if not tool_name:
            continue
        tool_args = tool_call.get("args")
        if tool_args is None:
            tool_args = tool_call.get("arguments")
        if tool_args is None:
            tool_args = tool_call.get("input")
        if tool_args is None and func:
            tool_args = func.get("arguments") or func.get("args") or func.get("input")
        if isinstance(tool_args, str):
            try:
                parsed_args = json.loads(tool_args)
                tool_args = parsed_args if isinstance(parsed_args, dict) else {}
            except json.JSONDecodeError:
                recovered_args = _recover_partial_json(tool_args)
                tool_args = recovered_args or {}
        if tool_args is None:
            tool_args = {}

        # Log tool calls with empty args for debugging
        if not tool_args and tool_name:
            logger.debug(
                "tool_call_with_empty_args",
                tool_name=tool_name,
                raw_tool_call=str(tool_call)[:200],
            )

        # Handle missing required args for specific tools
        if tool_name == "web_search" and not tool_args.get("query"):
            if query:
                tool_args = {**tool_args, "query": query}
            else:
                continue

        # Handle browser_navigate with missing url - extract from query
        if tool_name == "browser_navigate" and not tool_args.get("url"):
            import re
            # Try to extract URL from the query
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls = re.findall(url_pattern, query)
            if urls:
                # Strip trailing punctuation (including Chinese punctuation)
                extracted_url = urls[0].rstrip('.,;:!?，。；：！？、）】》')
                tool_args = {**tool_args, "url": extracted_url}
                logger.info(
                    "browser_navigate_url_extracted_from_query",
                    url=extracted_url[:50],
                )
            else:
                # Try to find domain-like patterns (e.g., "example.com")
                domain_pattern = r'\b([a-zA-Z0-9][-a-zA-Z0-9]*\.)+[a-zA-Z]{2,}\b'
                domains = re.findall(domain_pattern, query)
                if domains:
                    # Reconstruct full domain and add https://
                    full_domain_match = re.search(r'\b((?:[a-zA-Z0-9][-a-zA-Z0-9]*\.)+[a-zA-Z]{2,})\b', query)
                    if full_domain_match:
                        inferred_url = f"https://{full_domain_match.group(1)}"
                        tool_args = {**tool_args, "url": inferred_url}
                        logger.info(
                            "browser_navigate_url_inferred_from_domain",
                            url=inferred_url,
                        )
                    else:
                        logger.debug(
                            "browser_navigate_missing_url",
                            query=query[:50],
                        )
                        continue
                else:
                    logger.debug(
                        "browser_navigate_missing_url",
                        query=query[:50],
                    )
                    continue

        # Skip generate_image if prompt is missing (required field)
        if tool_name == "generate_image" and not tool_args.get("prompt"):
            if query:
                tool_args = {**tool_args, "prompt": query}
                logger.info(
                    "generate_image_prompt_fallback",
                    query=query[:100],
                )
            else:
                logger.debug(
                    "skipping_generate_image_missing_prompt",
                    tool_args=tool_args,
                    query=query[:100],
                )
                continue
        # Skip invoke_skill if required fields are missing
        if tool_name == "invoke_skill":
            params = tool_args.get("params")
            if isinstance(params, str):
                try:
                    parsed_params = json.loads(params)
                    if isinstance(parsed_params, dict):
                        tool_args["params"] = parsed_params
                        params = parsed_params
                except json.JSONDecodeError:
                    logger.debug(
                        "invoke_skill_params_parse_failed",
                        tool_args_preview=str(tool_args)[:200],
                    )
            if not tool_args.get("skill_id") or not isinstance(params, dict):
                logger.debug(
                    "skipping_invoke_skill_missing_args",
                    tool_args=tool_args,
                    query=query[:100],
                )
                continue

        # Skip execute_code if code is missing (required field)
        if tool_name == "execute_code" and not tool_args.get("code"):
            logger.debug(
                "skipping_execute_code_missing_code",
                tool_args=tool_args,
            )
            continue

        # Skip browser_click if coordinates are missing (required fields)
        if tool_name == "browser_click" and (
            tool_args.get("x") is None or tool_args.get("y") is None
        ):
            logger.debug(
                "skipping_browser_click_missing_coordinates",
                tool_args=tool_args,
            )
            continue

        # Skip browser_type if text is missing (required field)
        if tool_name == "browser_type" and not tool_args.get("text"):
            logger.debug(
                "skipping_browser_type_missing_text",
                tool_args=tool_args,
            )
            continue


        tool_call_id = tool_call.get("id") or tool_call.get("tool_call_id")
        if not tool_call_id:
            import uuid

            tool_call_id = str(uuid.uuid4())
        normalized_tool_calls.append(
            {
                **tool_call,
                "id": tool_call_id,
                "name": tool_name,
                "args": tool_args,
            }
        )

    # Deduplicate by tool_call_id to prevent "tool_use ids must be unique" errors
    seen_ids = set()
    deduplicated_tool_calls = []
    for tc in normalized_tool_calls:
        tc_id = tc.get("id")
        if tc_id not in seen_ids:
            seen_ids.add(tc_id)
            deduplicated_tool_calls.append(tc)
        else:
            logger.warning("duplicate_tool_call_id_removed", tool_call_id=tc_id, tool_name=tc.get("name"))

    # AIMessage requires tool_calls to be a list (not None)
    # Only include tool_calls if we have any
    if deduplicated_tool_calls:
        return AIMessage(content=full_content, tool_calls=deduplicated_tool_calls)
    return AIMessage(content=full_content)


@dataclass
class ReActLoopConfig:
    """Configuration for the ReAct loop.

    Attributes:
        max_iterations: Maximum number of tool-calling iterations
        max_retries_per_tool: Maximum retries per tool execution
        retry_base_delay: Base delay for exponential backoff on retries
        enable_streaming: Whether to stream LLM responses
        truncate_tool_results: Whether to truncate long tool results
        tool_result_max_chars: Maximum characters for tool results (when truncating)
        handoff_behavior: How to handle handoff requests ("immediate" or "deferred")
        max_consecutive_errors: Maximum consecutive tool errors before stopping
        max_message_tokens: Token budget for message history (approximate)
        preserve_recent_messages: Number of recent messages to always preserve
        parallel_tool_execution: Whether to execute independent tools in parallel
        max_parallel_tools: Maximum number of tools to execute in parallel (0 = unlimited)
    """
    max_iterations: int = 5
    max_retries_per_tool: int = 2
    retry_base_delay: float = 1.0
    enable_streaming: bool = True
    truncate_tool_results: bool = True
    tool_result_max_chars: int = 2000
    handoff_behavior: str = "immediate"  # "immediate" or "deferred"
    max_consecutive_errors: int = 2
    max_message_tokens: int = 100000  # Conservative token budget
    preserve_recent_messages: int = 4  # Keep last N messages when truncating
    parallel_tool_execution: bool = True  # Execute independent tools in parallel
    max_parallel_tools: int = 5  # Limit concurrent tool executions


def get_react_config(agent_type: str, tier=None) -> ReActLoopConfig:
    """Get the ReAct loop configuration for a specific agent type and tier.

    Uses ``get_quality_profile(tier)`` to derive iteration limits, parallelism,
    and truncation thresholds.  Agent-type overrides (e.g. handoff behavior,
    extra chars for research) are layered on top.

    Args:
        agent_type: The agent type (task, research, data)
        tier: Optional ModelTier enum, string, or None (defaults to PRO)

    Returns:
        ReActLoopConfig tuned for the given agent + tier combination
    """
    from app.ai.model_tiers import get_quality_profile

    profile = get_quality_profile(tier)

    # Base config from quality profile
    max_iterations = profile.react_max_iterations
    tool_result_max_chars = profile.react_tool_result_max_chars
    max_parallel_tools = profile.react_max_parallel_tools
    handoff_behavior = "immediate"

    # Agent-type overrides
    if agent_type == "research":
        handoff_behavior = "deferred"
        tool_result_max_chars = profile.react_tool_result_max_chars + 500
    elif agent_type == "data":
        handoff_behavior = "deferred"
        max_iterations = max(profile.data_analysis_max_iterations, profile.react_max_iterations)

    return ReActLoopConfig(
        max_iterations=max_iterations,
        handoff_behavior=handoff_behavior,
        truncate_tool_results=True,
        tool_result_max_chars=tool_result_max_chars,
        parallel_tool_execution=True,
        max_parallel_tools=max_parallel_tools,
    )


def estimate_message_tokens(message: BaseMessage) -> int:
    """Estimate token count for a message.

    Uses a simple heuristic: ~4 characters per token on average.

    Args:
        message: LangChain message to estimate

    Returns:
        Estimated token count
    """
    content = ""
    if isinstance(message.content, str):
        content = message.content
    elif isinstance(message.content, list):
        # Handle multimodal content
        for item in message.content:
            if isinstance(item, str):
                content += item
            elif isinstance(item, dict) and item.get("type") == "text":
                content += item.get("text", "")
    return len(content) // 4 + 1


def _snap_to_tool_pair_boundary(messages: list[BaseMessage], split_index: int) -> int:
    """Adjust a split index so it doesn't orphan ToolMessages from their AIMessage.

    A valid boundary must not:
    - Place an AIMessage with tool_calls in the dropped portion while its
      ToolMessage responses are in the kept portion (or vice versa).

    We scan forward from the proposed split to find a clean boundary where:
    - The message at split_index is NOT a ToolMessage (which would be orphaned
      from its AIMessage in the dropped portion)
    - The message just before split_index is NOT an AIMessage with tool_calls
      (whose ToolMessage responses would be in the kept portion)

    Args:
        messages: The message list being split
        split_index: Proposed split point (messages[:split_index] dropped,
                     messages[split_index:] kept)

    Returns:
        Adjusted split index that respects tool call/result pairing
    """
    if split_index <= 0 or split_index >= len(messages):
        return split_index

    # Scan backward: if the message at split_index is a ToolMessage, we need
    # to include its parent AIMessage in the kept portion too.
    while split_index > 0 and isinstance(messages[split_index], ToolMessage):
        split_index -= 1

    # Now check: if the message just before split_index is an AIMessage with
    # tool_calls, its ToolMessages would be in the kept portion but the AIMessage
    # would be dropped. Include it in the kept portion.
    if (
        split_index > 0
        and isinstance(messages[split_index - 1], AIMessage)
        and getattr(messages[split_index - 1], "tool_calls", None)
    ):
        split_index -= 1

    return split_index


def truncate_messages_to_budget(
    messages: list[BaseMessage],
    max_tokens: int = 100000,
    preserve_system: bool = True,
    preserve_recent: int = 4,
) -> tuple[list[BaseMessage], bool]:
    """Truncate old messages to stay within token budget.

    Preserves system messages (if preserve_system=True) and the most recent
    N messages (preserve_recent). Drops oldest non-system messages first.
    Respects tool call/result pairing so that AIMessages with tool_calls are
    never separated from their corresponding ToolMessages.

    Args:
        messages: List of messages to potentially truncate
        max_tokens: Maximum token budget
        preserve_system: Whether to always keep system messages
        preserve_recent: Number of recent messages to always keep

    Returns:
        Tuple of (truncated messages, was_truncated)
    """
    if not messages:
        return messages, False

    # Calculate current token estimate
    total_tokens = sum(estimate_message_tokens(m) for m in messages)

    if total_tokens <= max_tokens:
        return messages, False

    # Separate messages into categories
    system_messages = []
    other_messages = []

    for msg in messages:
        if preserve_system and isinstance(msg, SystemMessage):
            system_messages.append(msg)
        else:
            other_messages.append(msg)

    # Compute split point and snap to a tool-pair-safe boundary
    raw_split = len(other_messages) - preserve_recent
    if raw_split <= 0:
        return messages, False

    split_index = _snap_to_tool_pair_boundary(other_messages, raw_split)

    # Preserve the most recent messages (after snapping)
    if split_index < len(other_messages):
        preserved_recent = other_messages[split_index:]
    else:
        preserved_recent = other_messages
    droppable = other_messages[:split_index] if split_index > 0 else []

    # Calculate tokens for preserved messages
    preserved_tokens = sum(estimate_message_tokens(m) for m in system_messages + preserved_recent)

    if preserved_tokens >= max_tokens:
        # Even preserved messages exceed budget - return them anyway with warning
        logger.warning(
            "preserved_messages_exceed_budget",
            preserved_tokens=preserved_tokens,
            max_tokens=max_tokens,
        )
        return system_messages + preserved_recent, True

    # Add droppable messages from most recent, dropping oldest first
    remaining_budget = max_tokens - preserved_tokens
    kept_droppable = []

    for msg in reversed(droppable):
        msg_tokens = estimate_message_tokens(msg)
        if remaining_budget >= msg_tokens:
            kept_droppable.insert(0, msg)
            remaining_budget -= msg_tokens

    truncated = system_messages + kept_droppable + preserved_recent
    was_truncated = len(truncated) < len(messages)

    if was_truncated:
        logger.info(
            "messages_truncated",
            original_count=len(messages),
            truncated_count=len(truncated),
            dropped_count=len(messages) - len(truncated),
        )

    return truncated, was_truncated


def truncate_tool_result(content: str, max_chars: int = 2000) -> str:
    """Truncate a tool result to stay within character limit.

    For JSON content, attempts structural truncation (removing inner array
    elements) before falling back to character-level truncation.  For
    non-JSON content, preserves the beginning and end.

    Args:
        content: Tool result content to truncate
        max_chars: Maximum characters allowed

    Returns:
        Truncated content with indicator if truncated
    """
    if len(content) <= max_chars:
        return content

    # Attempt JSON-aware truncation for JSON content
    stripped = content.strip()
    if stripped.startswith(("{", "[")):
        try:
            data = json.loads(stripped)
            result = _truncate_json_structure(data, max_chars)
            if result is not None:
                return result
        except (json.JSONDecodeError, ValueError):
            pass  # Fall through to character-level truncation

    # Character-level fallback: keep first 60% and last 30%, 10% for msg
    head_budget = int(max_chars * 0.6)
    tail_budget = int(max_chars * 0.3)
    dropped = len(content) - head_budget - tail_budget
    truncation_msg = f"\n\n... [truncated {dropped} characters] ...\n\n"

    return content[:head_budget] + truncation_msg + content[-tail_budget:]


def _truncate_json_structure(data: Any, max_chars: int) -> str | None:
    """Attempt to truncate a JSON structure by removing inner array elements.

    Finds the largest array in the top-level structure and progressively
    removes elements from the middle until the result fits within max_chars.

    Args:
        data: Parsed JSON data
        max_chars: Maximum characters for the output

    Returns:
        Truncated JSON string, or None if structural truncation isn't feasible
    """
    # Find the target array to truncate
    target_array = None
    if isinstance(data, list):
        target_array = data
    elif isinstance(data, dict):
        # Find the largest list value in the top-level dict
        best_key = None
        best_len = 0
        for key, value in data.items():
            if isinstance(value, list) and len(value) > best_len:
                best_key = key
                best_len = len(value)
        if best_key is not None and best_len > 2:
            target_array = data[best_key]

    if target_array is None or len(target_array) <= 2:
        return None

    # Binary search for the number of elements we can keep
    original_len = len(target_array)
    low, high = 2, original_len
    best_result = None

    while low <= high:
        mid = (low + high) // 2
        # Keep first half and last half of the allowed count
        keep_head = mid // 2
        keep_tail = mid - keep_head
        truncated = (
            target_array[:keep_head]
            + [f"... ({original_len - mid} items truncated) ..."]
            + target_array[-keep_tail:]
        )

        # Temporarily replace and serialize
        original_contents = target_array[:]
        target_array[:] = truncated
        try:
            result = json.dumps(data, default=str, ensure_ascii=False)
        finally:
            target_array[:] = original_contents

        if len(result) <= max_chars:
            best_result = result
            low = mid + 1  # Try keeping more
        else:
            high = mid - 1  # Try keeping fewer

    return best_result


def deduplicate_tool_messages(messages: list) -> list:
    """Remove duplicate ToolMessages with the same tool_call_id.

    The Anthropic API requires exactly one tool_result per tool_use.
    This ensures we don't have duplicates which cause API errors.

    Args:
        messages: List of LangChain messages

    Returns:
        Deduplicated list of messages
    """
    seen_tool_call_ids: set[str] = set()
    result = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_call_id = msg.tool_call_id
            if tool_call_id in seen_tool_call_ids:
                logger.warning(
                    "duplicate_tool_message_removed",
                    tool_call_id=tool_call_id,
                    tool_name=getattr(msg, "name", "unknown"),
                )
                continue
            seen_tool_call_ids.add(tool_call_id)
        result.append(msg)

    return result


# Tools that require sequential execution due to side effects or special handling
SEQUENTIAL_TOOLS = {
    "browser_navigate",  # Needs pre-execution stream URL setup
    "browser_click",     # Browser state dependent
    "browser_type",      # Browser state dependent
    "browser_scroll",    # Browser state dependent
    "browser_press_key", # Browser state dependent
}


async def _execute_single_tool(
    tool: BaseTool,
    tool_name: str,
    tool_args: dict,
    tool_call_id: str,
    config: "ReActLoopConfig",
    events: list[dict],
    on_tool_call: Callable | None,
    on_tool_result: Callable | None,
    on_token: Callable | None,
) -> tuple[ToolMessage, bool]:
    """Execute a single tool and return the result message.

    Delegates to the shared tool execution pipeline via execute_tool()
    with CanonicalToolHooks for image/browser post-processing.

    Args:
        tool: The tool to execute
        tool_name: Name of the tool
        tool_args: Arguments for the tool
        tool_call_id: ID of the tool call
        config: ReAct loop configuration
        events: Events list to append to (for image events, etc.)
        on_tool_call: Callback when tool is called
        on_tool_result: Callback when tool returns
        on_token: Callback for tokens (for image placeholders)

    Returns:
        Tuple of (ToolMessage, is_error)
    """
    from app.agents.tools.tool_pipeline import (
        CanonicalToolHooks,
        ToolExecutionContext,
        execute_tool,
    )

    hooks = CanonicalToolHooks(
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
        on_token=on_token,
        all_events=events,
    )
    ctx = ToolExecutionContext(
        tool_name=tool_name,
        tool_args=tool_args,
        tool_call_id=tool_call_id,
        tool=tool,
    )
    result = await execute_tool(ctx, hooks=hooks, config=config, use_retry=True)
    return result.message, result.is_error


async def _execute_tools_parallel(
    tool_calls: list[dict],
    tool_map: dict[str, BaseTool],
    config: "ReActLoopConfig",
    events: list[dict],
    extra_tool_args: dict | None,
    on_tool_call: Callable | None,
    on_tool_result: Callable | None,
    on_token: Callable | None,
    on_browser_stream: Callable | None = None,
) -> tuple[list[ToolMessage], int]:
    """Execute tool calls in parallel where possible.

    Tools in SEQUENTIAL_TOOLS are executed sequentially.
    Other tools are executed in parallel with concurrency limit.

    Args:
        tool_calls: List of tool call dicts
        tool_map: Mapping of tool names to tools
        config: ReAct loop configuration
        events: Events list to append to
        extra_tool_args: Extra args to inject
        on_tool_call: Callback when tool is called
        on_tool_result: Callback when tool returns
        on_token: Callback for tokens
        on_browser_stream: Callback when browser stream is ready

    Returns:
        Tuple of (tool_results, error_count)
    """
    # Separate sequential and parallel tools
    sequential_calls = []
    parallel_calls = []

    for tc in tool_calls:
        tool_name = tc.get("name", "")
        if tool_name in SEQUENTIAL_TOOLS:
            sequential_calls.append(tc)
        else:
            parallel_calls.append(tc)

    results: list[tuple[str, ToolMessage, bool]] = []  # (tool_call_id, message, is_error)
    error_count = 0

    # Execute parallel tools concurrently
    if parallel_calls and config.parallel_tool_execution:
        semaphore = asyncio.Semaphore(config.max_parallel_tools or 5)

        async def execute_with_semaphore(tc: dict) -> tuple[str, ToolMessage, bool]:
            async with semaphore:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                tool_call_id = tc.get("id", "")

                if extra_tool_args:
                    tool_args = {**tool_args, **extra_tool_args}

                tool = tool_map.get(tool_name)
                if not tool:
                    return tool_call_id, ToolMessage(
                        content=f"Tool not found: {tool_name}",
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    ), True

                msg, is_err = await _execute_single_tool(
                    tool, tool_name, tool_args, tool_call_id,
                    config, events, on_tool_call, on_tool_result, on_token
                )
                return tool_call_id, msg, is_err

        parallel_results = await asyncio.gather(
            *[execute_with_semaphore(tc) for tc in parallel_calls],
            return_exceptions=True,
        )

        for i, res in enumerate(parallel_results):
            if isinstance(res, Exception):
                tc = parallel_calls[i]
                tool_call_id = tc.get("id", "")
                tool_name = tc.get("name", "")
                results.append((tool_call_id, ToolMessage(
                    content=f"Unexpected error: {res}",
                    tool_call_id=tool_call_id,
                    name=tool_name,
                ), True))
                error_count += 1
            else:
                tcid, msg, is_err = res
                results.append((tcid, msg, is_err))
                if is_err:
                    error_count += 1
    else:
        # Fall back to sequential for all parallel tools if disabled
        for tc in parallel_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})
            tool_call_id = tc.get("id", "")

            if extra_tool_args:
                tool_args = {**tool_args, **extra_tool_args}

            tool = tool_map.get(tool_name)
            if not tool:
                results.append((tool_call_id, ToolMessage(
                    content=f"Tool not found: {tool_name}",
                    tool_call_id=tool_call_id,
                    name=tool_name,
                ), True))
                error_count += 1
                continue

            msg, is_err = await _execute_single_tool(
                tool, tool_name, tool_args, tool_call_id,
                config, events, on_tool_call, on_tool_result, on_token
            )
            results.append((tool_call_id, msg, is_err))
            if is_err:
                error_count += 1

    # Execute sequential tools one by one (with browser pre-execution hooks)
    for tc in sequential_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool_call_id = tc.get("id", "")

        if extra_tool_args:
            tool_args = {**tool_args, **extra_tool_args}

        tool = tool_map.get(tool_name)
        if not tool:
            results.append((tool_call_id, ToolMessage(
                content=f"Tool not found: {tool_name}",
                tool_call_id=tool_call_id,
                name=tool_name,
            ), True))
            error_count += 1
            continue

        # Pre-execution: For browser_navigate, get stream URL first
        if tool_name == "browser_navigate" and on_browser_stream:
            try:
                from app.agents import events as agent_events
                from app.sandbox import get_desktop_sandbox_manager

                manager = get_desktop_sandbox_manager()
                user_id = tool_args.get("user_id")
                task_id = tool_args.get("task_id")

                session = await manager.get_or_create_sandbox(
                    user_id=user_id, task_id=task_id, launch_browser=True
                )

                try:
                    stream_url, auth_key = await session.executor.get_stream_url(require_auth=True)
                    stream_event = agent_events.browser_stream(
                        stream_url=stream_url, sandbox_id=session.sandbox_id, auth_key=auth_key
                    )
                    events.append(stream_event)
                    on_browser_stream(stream_url, session.sandbox_id, auth_key)
                    logger.info("browser_stream_event_emitted_early", sandbox_id=session.sandbox_id)
                except NotImplementedError:
                    # Provider doesn't support streaming (e.g., BoxLite) — skip
                    logger.debug("browser_stream_not_supported_by_provider")
                except Exception as stream_err:
                    if "already running" in str(stream_err).lower():
                        # Stream already running — try to get URL again
                        try:
                            stream_url, auth_key = await session.executor.get_stream_url(
                                require_auth=True
                            )
                            stream_event = agent_events.browser_stream(
                                stream_url=stream_url,
                                sandbox_id=session.sandbox_id,
                                auth_key=auth_key,
                            )
                            events.append(stream_event)
                            on_browser_stream(stream_url, session.sandbox_id, auth_key)
                        except Exception:
                            logger.warning("browser_stream_retry_failed", error=str(stream_err))
                    else:
                        logger.warning("browser_stream_early_failed", error=str(stream_err))
            except Exception as e:
                logger.warning("browser_pre_execution_failed", error=str(e))

        msg, is_err = await _execute_single_tool(
            tool, tool_name, tool_args, tool_call_id,
            config, events, on_tool_call, on_tool_result, on_token
        )
        results.append((tool_call_id, msg, is_err))
        if is_err:
            error_count += 1

    # Sort results by original tool_call order
    tool_call_order = {tc.get("id", ""): i for i, tc in enumerate(tool_calls)}
    results.sort(key=lambda x: tool_call_order.get(x[0], 999))

    return [msg for _, msg, _ in results], error_count


@dataclass
class ReActLoopResult:
    """Result from executing the ReAct loop.

    Attributes:
        messages: Updated message list including all tool interactions
        final_response: The final AI response content
        tool_iterations: Number of tool-calling iterations performed
        events: List of events generated during execution
        pending_handoff: Handoff info if a handoff was initiated
    """
    messages: list[BaseMessage]
    final_response: str = ""
    tool_iterations: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    pending_handoff: dict[str, Any] | None = None


async def execute_react_loop(
    llm_with_tools,
    messages: list[BaseMessage],
    tools: list[BaseTool],
    query: str,
    config: ReActLoopConfig | None = None,
    source_agent: str = "unknown",
    on_tool_call: Callable[[str, dict, str], None] | None = None,
    on_tool_result: Callable[[str, str, str], None] | None = None,
    on_handoff: Callable[[str, str, str], None] | None = None,
    on_token: Callable[[str], None] | None = None,
    extra_tool_args: dict[str, Any] | None = None,
    on_browser_stream: Callable[[str, str, str | None], None] | None = None,
    tool_hooks: Any | None = None,
) -> ReActLoopResult:
    """Execute a unified ReAct loop with tool calling and retry support.

    This is the canonical ReAct implementation that all agents should use.
    It handles:
    - Streaming LLM responses with token callbacks
    - Tool call detection and execution with retry
    - Handoff detection (immediate or deferred)
    - Iteration limits
    - Message truncation to stay within token budget
    - Tool result truncation

    Args:
        llm_with_tools: LLM instance with tools bound
        messages: Initial message list
        tools: List of available tools
        query: The user query (used for missing args fallback)
        config: Configuration for the loop
        source_agent: The name of the agent executing the loop (for handoff tracking)
        on_tool_call: Callback when a tool is called (tool_name, args, tool_call_id)
        on_tool_result: Callback when a tool returns (tool_name, result, tool_call_id)
        on_handoff: Callback when a handoff is detected (source, target, task)
        on_token: Callback for each streamed token (token_content)
        extra_tool_args: Additional arguments to inject into all tool calls (e.g., user_id, task_id)
        on_browser_stream: Callback when browser stream is ready (stream_url, sandbox_id, auth_key)
        tool_hooks: Optional ToolExecutionHooks instance for custom tool execution behavior.
            When provided, used instead of creating CanonicalToolHooks from flat callbacks.

    Returns:
        ReActLoopResult with updated messages and metadata
    """
    if config is None:
        config = ReActLoopConfig()

    messages = list(messages)  # Don't mutate input
    events: list[dict[str, Any]] = []
    tool_iterations = 0
    consecutive_errors = 0
    pending_handoff = None
    deferred_handoff = None

    # Build tool lookup for execution
    tool_map = {tool.name: tool for tool in tools}

    while tool_iterations < config.max_iterations:
        # Apply message truncation to stay within token budget
        messages, was_truncated = truncate_messages_to_budget(
            messages,
            max_tokens=config.max_message_tokens,
            preserve_recent=config.preserve_recent_messages,
        )

        if was_truncated:
            events.append({
                "type": "stage",
                "name": "context",
                "description": "Message history truncated to fit context window",
                "status": "completed",
            })

        # Stream LLM response
        response_chunks = []
        try:
            if config.enable_streaming:
                async for chunk in llm_with_tools.astream(messages):
                    response_chunks.append(chunk)
                    # Stream tokens to callback if provided
                    if on_token and hasattr(chunk, "content") and chunk.content:
                        content = extract_text_from_content(chunk.content)
                        if content:
                            on_token(content)
            else:
                response = await llm_with_tools.ainvoke(messages)
                response_chunks = [response]
        except Exception as e:
            # Providers with "thinking" mode (e.g. DeepSeek, Kimi) require
            # reasoning_content on every assistant message with tool_calls.
            # ThinkingAwareChatOpenAI patches the payload when its flag is
            # set. If not pre-configured, detect it here and enable the
            # patch, then retry.
            if "reasoning_content" in str(e):
                logger.info("react_loopthinking_mode")
                inner_llm = getattr(llm_with_tools, "bound", llm_with_tools)
                if hasattr(inner_llm, "thinking_mode"):
                    inner_llm.thinking_mode = True
                if config.enable_streaming:
                    async for chunk in llm_with_tools.astream(messages):
                        response_chunks.append(chunk)
                        if on_token and hasattr(chunk, "content") and chunk.content:
                            content = extract_text_from_content(chunk.content)
                            if content:
                                on_token(content)
                else:
                    response = await llm_with_tools.ainvoke(messages)
                    response_chunks = [response]
            else:
                logger.error("react_loop_llm_error", error=str(e), iteration=tool_iterations)
                raise

        # Build AI message from chunks
        ai_message = build_ai_message_from_chunks(response_chunks, query)

        # Check for tool calls
        if not ai_message.tool_calls:
            # No tool calls - we have the final response
            messages.append(ai_message)
            break

        # Separate handoff and regular tool calls
        handoff_call = None
        regular_tool_calls = []

        for tool_call in ai_message.tool_calls:
            tool_name = tool_call.get("name") or ""
            if tool_name.startswith("handoff_to_"):
                handoff_call = tool_call
            else:
                regular_tool_calls.append(tool_call)

        # Handle handoff based on config
        if handoff_call:
            target_agent = handoff_call.get("name", "").replace("handoff_to_", "")
            task_description = handoff_call.get("args", {}).get("task_description", "")
            context = handoff_call.get("args", {}).get("context", "")

            handoff_info = {
                "source_agent": source_agent,
                "target_agent": target_agent,
                "task_description": task_description,
                "context": context,
            }

            if config.handoff_behavior == "immediate" or not regular_tool_calls:
                # Return immediately with handoff
                pending_handoff = handoff_info

                if on_handoff:
                    on_handoff(source_agent, target_agent, task_description)

                logger.info(
                    "react_loop_handoff_immediate",
                    source=source_agent,
                    target=target_agent,
                    iteration=tool_iterations,
                )

                return ReActLoopResult(
                    messages=messages,
                    final_response="",
                    tool_iterations=tool_iterations,
                    events=events,
                    pending_handoff=pending_handoff,
                )
            else:
                # Deferred: execute regular tools first, then handoff
                deferred_handoff = handoff_info
                logger.info(
                    "react_loop_handoff_deferred",
                    source=source_agent,
                    target=target_agent,
                    pending_tools=[tc.get("name") for tc in regular_tool_calls],
                )

        # Add AI message with tool calls to history
        messages.append(ai_message)
        tool_iterations += 1

        # Execute tool calls (parallel where possible, sequential for browser tools)
        tool_results, iteration_errors = await _execute_tools_parallel(
            tool_calls=regular_tool_calls,
            tool_map=tool_map,
            config=config,
            events=events,
            extra_tool_args=extra_tool_args,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_token=on_token,
            on_browser_stream=on_browser_stream,
        )

        # Update consecutive errors
        if iteration_errors == len(regular_tool_calls) and regular_tool_calls:
            consecutive_errors += 1
        else:
            consecutive_errors = 0

        # Check for too many consecutive errors
        if consecutive_errors >= config.max_consecutive_errors:
            logger.warning(
                "react_loop_max_consecutive_errors",
                consecutive_errors=consecutive_errors,
                max=config.max_consecutive_errors,
            )
            events.append({
                "type": "stage",
                "name": "tool",
                "description": f"Too many consecutive tool errors ({consecutive_errors}); stopping.",
                "status": "completed",
            })
            messages.extend(tool_results)
            break

        # Add tool results to messages
        messages.extend(tool_results)

        logger.debug(
            "react_loop_iteration_completed",
            iteration=tool_iterations,
            tool_count=len(tool_results),
            error_count=iteration_errors,
        )

        # If we have a deferred handoff and just finished tools, return it
        if deferred_handoff and tool_iterations >= 1:
            pending_handoff = deferred_handoff

            if on_handoff:
                on_handoff(
                    deferred_handoff["source_agent"],
                    deferred_handoff["target_agent"],
                    deferred_handoff["task_description"],
                )

            logger.info(
                "react_loop_deferred_handoff_executed",
                source=deferred_handoff["source_agent"],
                target=deferred_handoff["target_agent"],
            )

            return ReActLoopResult(
                messages=messages,
                final_response="",
                tool_iterations=tool_iterations,
                events=events,
                pending_handoff=pending_handoff,
            )

    # Check if we hit the iteration limit
    if tool_iterations >= config.max_iterations:
        logger.warning(
            "react_loop_max_iterations",
            iterations=tool_iterations,
            max=config.max_iterations,
        )
        events.append({
            "type": "stage",
            "name": "tool",
            "description": (
                f"Tool limit reached ({config.max_iterations}); "
                "finishing without more tool calls."
            ),
            "status": "completed",
        })

    # Extract final response
    final_response = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            final_response = extract_text_from_content(msg.content)
            break

    return ReActLoopResult(
        messages=messages,
        final_response=final_response,
        tool_iterations=tool_iterations,
        events=events,
        pending_handoff=pending_handoff,
    )
