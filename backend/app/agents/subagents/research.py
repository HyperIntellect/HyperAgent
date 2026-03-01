"""Research subagent for multi-step deep research tasks with tool calling and handoff support."""

import threading
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
from app.agents.prompts import (
    get_analysis_prompt,
    get_report_prompt,
    get_search_system_prompt,
    get_synthesis_prompt,
)
from app.agents.scenarios import get_scenario_config
from app.agents.state import ResearchState
from app.agents.tools import (
    get_react_config,
    get_tools_for_agent,
)
from app.agents.tools.react_tool import truncate_messages_to_budget
from app.agents.tools.tool_pipeline import (
    ResearchToolHooks,
    ToolExecutionContext,
    execute_tool,
    execute_tools_batch,
)
from app.ai.llm import extract_text_from_content, llm_service
from app.core.logging import get_logger
from app.config import settings
from app.guardrails.scanners.output_scanner import output_scanner
from app.models.schemas import ResearchDepth, ResearchScenario
from app.services.search import SearchResult

logger = get_logger(__name__)

# Module-level caches for research tools (computed once, thread-safe)
_cached_research_tools: list | None = None
_research_cache_lock = threading.Lock()


def _get_cached_research_tools() -> list:
    """Get the research tools list, computing and caching on first call (thread-safe)."""
    global _cached_research_tools
    if _cached_research_tools is None:
        with _research_cache_lock:
            if _cached_research_tools is None:
                tools = get_tools_for_agent("research", include_handoffs=True)
                # Sort tools alphabetically for consistent KV-cache hits across requests
                _cached_research_tools = sorted(tools, key=lambda t: t.name)
    return _cached_research_tools

# Depth-based configuration
DEPTH_CONFIG = {
    ResearchDepth.FAST: {
        "analysis_detail": "brief",
        "skip_synthesis": True,
        "report_length": "concise",
        "max_searches": 1,
        "search_depth": "basic",
    },
    ResearchDepth.DEEP: {
        "analysis_detail": "in-depth with follow-up questions",
        "skip_synthesis": False,
        "report_length": "detailed and extensive",
        "max_searches": 5,
        "search_depth": "advanced",
    },
}


async def init_config_node(state: ResearchState) -> dict:
    """Initialize research configuration from scenario and depth.

    Args:
        state: Current research state

    Returns:
        Dict with configuration fields
    """
    depth = state.get("depth", ResearchDepth.FAST)
    scenario = state.get("scenario", ResearchScenario.ACADEMIC)

    config = get_scenario_config(scenario)
    depth_config = DEPTH_CONFIG.get(depth, DEPTH_CONFIG[ResearchDepth.FAST])

    # Build search system prompt
    search_prompt = get_search_system_prompt(
        scenario=config["name"],
        search_focus=", ".join(config.get("search_focus", [])),
        depth=depth.value if isinstance(depth, ResearchDepth) else depth,
        max_searches=depth_config["max_searches"],
        report_length=depth_config["report_length"],
    )

    logger.info(
        "research_config_initialized",
        depth=depth.value if isinstance(depth, ResearchDepth) else depth,
        scenario=scenario.value if isinstance(scenario, ResearchScenario) else scenario,
    )

    lc_messages = [
        SystemMessage(
            content=search_prompt,
            additional_kwargs={"cache_control": {"type": "ephemeral"}},
        ),
    ]

    # Inject persistent user memories if available
    user_id = state.get("user_id")
    if user_id:
        from app.services.memory_service import get_memory_store

        memory_text = get_memory_store().format_memories_for_prompt(user_id)
        if memory_text:
            lc_messages.append(SystemMessage(content=memory_text))
            logger.info("research_user_memories_injected", user_id=user_id)

    lc_messages.append(HumanMessage(content=f"Research topic: {state.get('query') or ''}"))

    return {
        "system_prompt": config["system_prompt"],
        "report_structure": config["report_structure"],
        "depth_config": depth_config,
        "lc_messages": lc_messages,
        "sources": [],
        "search_complete": False,
        "tool_iterations": 0,
        "events": [
            events.config(
                depth=depth.value if isinstance(depth, ResearchDepth) else depth,
                scenario=scenario.value if isinstance(scenario, ResearchScenario) else scenario,
            ),
            events.stage("search", "Searching for sources...", "running"),
        ],
    }


async def search_agent_node(state: ResearchState) -> dict:
    """ReAct agent node that decides whether to search or finish.

    Args:
        state: Current research state

    Returns:
        Dict with updated messages, events, and potential handoff
    """
    lc_messages = state.get("lc_messages") or []
    depth_config = state.get("depth_config") or {}
    tool_iterations = state.get("tool_iterations") or 0
    consecutive_errors = state.get("consecutive_errors") or 0
    max_searches = depth_config.get("max_searches", 5)

    # Circuit breaker: stop if too many consecutive tool errors
    react_config = get_react_config("research")
    if consecutive_errors >= react_config.max_consecutive_errors:
        logger.warning(
            "research_consecutive_errors_limit",
            consecutive_errors=consecutive_errors,
            max=react_config.max_consecutive_errors,
        )
        return {
            "lc_messages": lc_messages,
            "search_complete": True,
            "events": [
                events.stage(
                    "search",
                    f"Stopped after {consecutive_errors} consecutive errors. "
                    "Proceeding with available sources.",
                    "completed",
                )
            ],
        }

    deferred_handoff = state.get("deferred_handoff")
    if deferred_handoff:
        logger.info(
            "processing_deferred_handoff",
            target=deferred_handoff.get("target_agent"),
        )
        return {
            "lc_messages": lc_messages,
            "search_complete": True,
            "events": [events.handoff(
                source="research",
                target=deferred_handoff.get("target_agent", ""),
                task=deferred_handoff.get("task_description", ""),
            )],
            "pending_handoff": deferred_handoff,
            "deferred_handoff": None,  # Clear the deferred handoff
        }

    logger.info(
        "search_agent_processing",
        message_count=len(lc_messages),
        tool_iterations=tool_iterations,
        max_searches=max_searches,
    )

    # Enforce iteration limit to prevent infinite loops
    if tool_iterations >= max_searches:
        logger.warning(
            "max_searches_reached",
            count=tool_iterations,
            max=max_searches,
        )
        return {
            "lc_messages": lc_messages,
            "search_complete": True,
            "events": [
                events.stage(
                    "search",
                    f"Reached maximum searches ({max_searches}). Proceeding with available sources.",
                    "completed",
                )
            ],
        }

    event_list = []

    # Get all tools for research agent (cached, includes browser, search, image, handoffs)
    all_tools = _get_cached_research_tools()

    # Get LLM with tools bound
    provider = state.get("provider")
    tier = state.get("tier")
    model = state.get("model")
    llm = llm_service.choose_llm_for_task("research", provider=provider, tier_override=tier, model_override=model)
    llm_with_tools = llm.bind_tools(all_tools)

    # Truncate messages to stay within token budget
    react_config = get_react_config("research")
    lc_messages, was_truncated = truncate_messages_to_budget(
        lc_messages,
        max_tokens=react_config.max_message_tokens,
        preserve_recent=react_config.preserve_recent_messages,
    )
    if was_truncated:
        logger.info("research_messages_truncated_for_budget")

    try:
        response = await llm_with_tools.ainvoke(lc_messages)
        lc_messages = lc_messages + [response]

        # Track and log tool calls
        if response.tool_calls:
            # First pass: separate handoff and non-handoff tool calls
            handoff_call = None
            other_tool_calls = []

            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name") or ""
                if tool_name.startswith("handoff_to_"):
                    handoff_call = tool_call
                else:
                    other_tool_calls.append(tool_call)

            # If we have a handoff but also other tools, defer the handoff
            # to let the search tools execute first
            if handoff_call and other_tool_calls:
                target_agent = handoff_call.get("name", "").replace("handoff_to_", "")
                task_description = handoff_call.get("args", {}).get("task_description", "")
                context = handoff_call.get("args", {}).get("context", "")

                deferred_handoff_info = {
                    "source_agent": "research",
                    "target_agent": target_agent,
                    "task_description": task_description,
                    "context": context,
                }

                logger.info(
                    "handoff_deferred_for_tools",
                    target=target_agent,
                    pending_tools=[tc.get("name") for tc in other_tool_calls],
                )

                # Increment iteration count (one iteration, regardless of tool call count)
                tool_iterations += 1
                for tool_call in other_tool_calls:
                    valid_tool_names = [tool.name for tool in all_tools]
                    if tool_call["name"] not in valid_tool_names:
                        logger.warning(
                            "invalid_tool_call",
                            tool=tool_call["name"],
                            allowed=valid_tool_names,
                        )
                        continue
                    event_list.append(events.tool_call(
                        tool_call["name"],
                        tool_call["args"],
                        tool_call.get("id"),
                    ))

                logger.info(
                    "search_tool_calls",
                    tools=[tc["name"] for tc in other_tool_calls],
                    tool_iterations=tool_iterations,
                )

                return {
                    "lc_messages": lc_messages,
                    "tool_iterations": tool_iterations,
                    "events": event_list,
                    "deferred_handoff": deferred_handoff_info,
                }

            # If only handoff (no other tools), process it immediately
            if handoff_call and not other_tool_calls:
                target_agent = handoff_call.get("name", "").replace("handoff_to_", "")
                task_description = handoff_call.get("args", {}).get("task_description", "")
                context = handoff_call.get("args", {}).get("context", "")

                pending_handoff = {
                    "source_agent": "research",
                    "target_agent": target_agent,
                    "task_description": task_description,
                    "context": context,
                }

                event_list.append(events.handoff(
                    source="research",
                    target=target_agent,
                    task=task_description,
                ))

                logger.info("research_handoff_detected", target=target_agent)

                return {
                    "lc_messages": lc_messages,
                    "search_complete": True,
                    "events": event_list,
                    "pending_handoff": pending_handoff,
                }

            # No handoff - process all tool calls normally (one iteration)
            tool_iterations += 1
            for tool_call in response.tool_calls:
                # Validate tool name
                valid_tool_names = [tool.name for tool in all_tools]
                if tool_call["name"] not in valid_tool_names:
                    logger.warning(
                        "invalid_tool_call",
                        tool=tool_call["name"],
                        allowed=valid_tool_names,
                    )
                    continue

                event_list.append(events.tool_call(
                    tool_call["name"],
                    tool_call["args"],
                    tool_call.get("id"),
                ))
            logger.info(
                "search_tool_calls",
                tools=[tc["name"] for tc in response.tool_calls],
                tool_iterations=tool_iterations,
            )

        if not response.tool_calls:
            logger.info("search_phase_complete_no_tool_calls", tool_iterations=tool_iterations)
            return {
                "lc_messages": lc_messages,
                "search_complete": True,
                "events": event_list,
            }

        return {
            "lc_messages": lc_messages,
            "tool_iterations": tool_iterations,
            "events": event_list,
        }

    except Exception as e:
        logger.error("search_agent_failed", error=str(e), tool_iterations=tool_iterations)
        return {
            "lc_messages": lc_messages,
            "search_complete": True,
            "events": [
                events.error(
                    error_msg=str(e), name="search",
                    description=f"Search error: {str(e)}",
                )
            ],
        }


async def search_tools_node(state: ResearchState) -> dict:
    """Execute search tool calls and collect results.

    Uses the shared tool execution pipeline via execute_tools_batch
    with ResearchToolHooks for source parsing.

    Args:
        state: Current research state with pending tool calls

    Returns:
        Dict with tool results and collected sources
    """
    lc_messages = list(state.get("lc_messages", []))
    sources = list(state.get("sources", []))
    user_id = state.get("user_id")
    task_id = state.get("task_id")

    event_list: list[dict] = []

    # Get the last AI message with tool calls
    last_message = lc_messages[-1] if lc_messages else None
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"lc_messages": lc_messages, "events": event_list}

    # Build tool map from cached research tools
    all_tools = _get_cached_research_tools()
    tool_map = {tool.name: tool for tool in all_tools}
    react_config = get_react_config("research")

    from app.agents.hitl.tool_risk import requires_approval

    def hitl_check(tool_name: str) -> bool:
        # invoke_skill is routed through the HITL partition so per-skill checks
        # can run in hook-level logic.
        return tool_name == "invoke_skill" or requires_approval(
            tool_name,
            auto_approve_tools=state.get("auto_approve_tools", []),
            hitl_enabled=state.get("hitl_enabled", True),
            risk_threshold=settings.hitl_default_risk_threshold,
        )

    hooks = ResearchToolHooks(state=state)
    tool_messages, batch_events, error_count, pending_interrupt = await execute_tools_batch(
        tool_calls=last_message.tool_calls,
        tool_map=tool_map,
        config=react_config,
        hooks=hooks,
        user_id=user_id,
        task_id=task_id,
        run_id=state.get("run_id"),
        hitl_partition=True,
        hitl_check=hitl_check,
    )

    sources.extend(hooks.collected_sources)
    lc_messages.extend(tool_messages)
    event_list.extend(batch_events)

    # Circuit breaker: track consecutive errors
    consecutive_errors = state.get("consecutive_errors") or 0
    total_tool_count = len(last_message.tool_calls)
    if error_count == total_tool_count and total_tool_count > 0:
        consecutive_errors += 1
    else:
        consecutive_errors = 0

    logger.info(
        "search_tools_executed",
        new_sources=len(sources),
        consecutive_errors=consecutive_errors,
        pending_interrupt=bool(pending_interrupt),
    )

    result = {
        "lc_messages": lc_messages,
        "sources": sources,
        "events": event_list,
        "consecutive_errors": consecutive_errors,
    }
    if pending_interrupt:
        result["pending_interrupt"] = pending_interrupt
    return result


def should_wait_or_search(state: ResearchState) -> Literal["wait_interrupt", "search_agent"]:
    """Decide whether to wait for user approval or continue searching."""
    if state.get("pending_interrupt"):
        return "wait_interrupt"
    return "search_agent"


async def wait_interrupt_node(state: ResearchState) -> dict:
    """Wait for and process HITL responses for research tool approvals."""
    pending = state.get("pending_interrupt")
    if not pending:
        return {"pending_interrupt": None}

    interrupt_id = pending.get("interrupt_id")
    thread_id = pending.get("thread_id", "default")
    tool_call_id = pending.get("tool_call_id")
    tool_name = pending.get("tool_name", "")
    tool_args = dict(pending.get("tool_args", {}) or {})
    user_id = state.get("user_id")
    task_id = state.get("task_id")

    lc_messages = list(state.get("lc_messages", []))
    event_list: list[dict] = []
    all_tools = _get_cached_research_tools()
    tool_map = {tool.name: tool for tool in all_tools}
    auto_approve_tools = list(state.get("auto_approve_tools", []))
    interrupt_manager = get_interrupt_manager()

    try:
        response = await interrupt_manager.wait_for_response(
            thread_id=thread_id,
            interrupt_id=interrupt_id,
            timeout_seconds=settings.hitl_decision_timeout,
        )
        action = response.get("action", "deny")

        if action in ("approve", "approve_always"):
            if action == "approve_always" and tool_name not in auto_approve_tools:
                auto_approve_tools.append(tool_name)

            react_config = get_react_config("research")
            ctx = ToolExecutionContext(
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call_id,
                tool=tool_map.get(tool_name),
                user_id=user_id,
                task_id=task_id,
                run_id=state.get("run_id"),
            )
            hooks = ResearchToolHooks(state=state, skip_before_execution=True)
            exec_result = await execute_tool(ctx, hooks=hooks, config=react_config)
            if exec_result.message:
                result_str = exec_result.message.content
            else:
                result_str = "Tool execution returned no result."
            event_list.extend(exec_result.events)
        elif action == "deny":
            result_str = f"User denied execution of {tool_name}. The tool was not executed."
        else:
            result_str = f"Unsupported approval action: {action}. Tool not executed."
    except TimeoutError:
        result_str = "Approval timed out. Tool not executed."
    except Exception as e:
        logger.error("research_wait_interrupt_error", error=str(e), interrupt_id=interrupt_id)
        result_str = f"Error while waiting for approval: {e}"

    lc_messages.append(
        ToolMessage(
            content=result_str,
            tool_call_id=tool_call_id,
            name=tool_name,
        )
    )
    event_list.append(events.tool_result(tool_name, result_str, tool_id=tool_call_id))

    result = {
        "lc_messages": lc_messages,
        "events": event_list,
        "pending_interrupt": None,
    }
    if auto_approve_tools != state.get("auto_approve_tools", []):
        result["auto_approve_tools"] = auto_approve_tools
    return result


def should_continue_search(state: ResearchState) -> Literal["tools", "collect"]:
    """Determine whether to execute tools or finish search phase.

    Args:
        state: Current research state

    Returns:
        Next node: "tools" if tool calls pending, "collect" if done
    """
    # Check for pending handoff
    if state.get("pending_handoff"):
        return "collect"

    # Check if search is marked complete
    if state.get("search_complete", False):
        return "collect"

    lc_messages = state.get("lc_messages", [])
    if not lc_messages:
        return "collect"

    last_message = lc_messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "collect"


async def collect_sources_node(state: ResearchState) -> dict:
    """Finalize source collection and prepare for analysis.

    Args:
        state: Current research state

    Returns:
        Dict with finalized sources and events
    """
    sources = state.get("sources", [])

    event_list = [events.stage(
        "search",
        f"Found {len(sources)} sources",
        "completed",
    )]

    logger.info("sources_collected", count=len(sources))

    result = {"events": event_list}

    # Propagate handoff if present
    pending_handoff = state.get("pending_handoff")
    if pending_handoff:
        result["pending_handoff"] = pending_handoff

    return result


async def analyze_node(state: ResearchState) -> dict:
    """Analyze the search results.

    Args:
        state: Current research state with sources

    Returns:
        Dict with analysis and events
    """
    pending_handoff = state.get("pending_handoff")
    sources = state.get("sources") or []

    # Only skip if there's a pending handoff AND no sources to analyze
    # When handoff is pending but we have sources, run analysis first
    # to populate research_findings for the target agent
    if pending_handoff and not sources:
        logger.info("analyze_skipped_no_sources", has_handoff=True)
        return {"analysis": "", "events": [], "pending_handoff": pending_handoff}

    query = state.get("query") or ""
    system_prompt = state.get("system_prompt") or ""
    depth_config = state.get("depth_config") or {}

    event_list = [events.stage(
        "analyze",
        f"Analyzing sources ({depth_config.get('analysis_detail', 'thorough')})...",
        "running",
    )]

    provider = state.get("provider")
    tier = state.get("tier")
    model = state.get("model")
    llm = llm_service.choose_llm_for_task("research", provider=provider, tier_override=tier, model_override=model)
    sources_text = _format_sources(sources)

    analysis_prompt = get_analysis_prompt(
        query=query,
        sources_text=sources_text,
        analysis_detail=depth_config.get("analysis_detail", "thorough"),
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=analysis_prompt),
        ])
        analysis_text = extract_text_from_content(response.content)
        logger.info("analysis_completed", query=query[:50])
    except Exception as e:
        logger.error("analysis_failed", error=str(e))
        analysis_text = f"Analysis of '{query}' based on {len(sources)} sources."

    event_list.append(events.stage("analyze", "Source analysis complete", "completed"))

    result = {
        "analysis": analysis_text,
        "events": event_list,
    }

    # Preserve pending handoff if present
    if pending_handoff:
        result["pending_handoff"] = pending_handoff
        logger.info("analyze_completed_with_handoff", target=pending_handoff.get("target_agent"))

    return result


async def synthesize_node(state: ResearchState) -> dict:
    """Synthesize the analysis findings.

    Args:
        state: Current research state with analysis

    Returns:
        Dict with synthesis and events
    """
    # Skip if there's a pending handoff
    if state.get("pending_handoff"):
        return {"synthesis": "", "events": [], "pending_handoff": state.get("pending_handoff")}

    query = state.get("query") or ""
    analysis_text = state.get("analysis") or ""
    system_prompt = state.get("system_prompt") or ""

    event_list = [events.stage("synthesize", "Synthesizing findings...", "running")]

    provider = state.get("provider")
    tier = state.get("tier")
    model = state.get("model")
    llm = llm_service.choose_llm_for_task("research", provider=provider, tier_override=tier, model_override=model)
    synthesis_prompt = get_synthesis_prompt(
        query=query,
        analysis_text=analysis_text,
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=synthesis_prompt),
        ])
        synthesis_text = extract_text_from_content(response.content)
        logger.info("synthesis_completed", query=query[:50])
    except Exception as e:
        logger.error("synthesis_failed", error=str(e))
        synthesis_text = analysis_text

    event_list.append(events.stage("synthesize", "Synthesis complete", "completed"))

    return {
        "synthesis": synthesis_text,
        "events": event_list,
    }


async def write_node(state: ResearchState) -> dict:
    """Write the research report.

    Args:
        state: Current research state with analysis/synthesis

    Returns:
        Dict with report chunks, events, and potential handoff
    """
    # Check for pending handoff
    pending_handoff = state.get("pending_handoff")
    if pending_handoff:
        return {
            "report_chunks": [],
            "response": "",
            "events": [],
            "pending_handoff": pending_handoff,
        }

    query = state.get("query") or ""
    analysis = state.get("analysis") or ""
    synthesis = state.get("synthesis") or ""
    sources = state.get("sources") or []
    system_prompt = state.get("system_prompt") or ""
    report_structure = state.get("report_structure") or []
    depth_config = state.get("depth_config") or {}

    event_list = [events.stage("write", "Writing research report...", "running")]

    # Use synthesis if available, otherwise analysis
    combined_findings = synthesis if synthesis else analysis
    sources_text = _format_sources(sources)

    locale = state.get("locale") or "en"
    report_prompt = get_report_prompt(
        query=query,
        combined_findings=combined_findings,
        sources_text=sources_text,
        report_structure=report_structure,
        report_length=depth_config.get("report_length", "comprehensive"),
        locale=locale,
    )

    provider = state.get("provider")
    tier = state.get("tier")
    model = state.get("model")
    llm = llm_service.choose_llm_for_task("research", provider=provider, tier_override=tier, model_override=model)
    report_chunks = []

    try:
        async for chunk in llm.astream([
            SystemMessage(content=system_prompt),
            HumanMessage(content=report_prompt),
        ]):
            if chunk.content:
                content = extract_text_from_content(chunk.content)
                if content:
                    report_chunks.append(content)
                    event_list.append(events.token(content))

        logger.info("report_completed", query=query[:50])
    except Exception as e:
        logger.error("report_generation_failed", error=str(e))
        event_list.append(events.token(f"\n\nError generating report: {str(e)}"))

    # Apply output guardrails to the full report
    report_text = "".join(report_chunks)
    scan_result = await output_scanner.scan(report_text, query)
    if scan_result.blocked:
        logger.warning(
            "research_output_guardrail_blocked",
            violations=[v.value for v in scan_result.violations],
            reason=scan_result.reason,
        )
        report_text = (
            "I apologize, but the research report could not be delivered due to content policy. "
            "Please try a different research topic."
        )
    elif scan_result.sanitized_content:
        logger.info("research_output_guardrail_sanitized")
        report_text = scan_result.sanitized_content

    event_list.append(events.stage("write", "Report complete", "completed"))

    return {
        "report_chunks": report_chunks,
        "response": report_text,
        "events": event_list,
    }


def should_synthesize(state: ResearchState) -> str:
    """Determine whether to run synthesis step.

    Args:
        state: Current research state

    Returns:
        Next node name: "synthesize" or "write"
    """
    # Skip synthesis if there's a pending handoff
    if state.get("pending_handoff"):
        return "write"

    depth_config = state.get("depth_config", {})
    if depth_config.get("skip_synthesis", False):
        return "write"
    return "synthesize"


def _format_sources(results: list[SearchResult]) -> str:
    """Format search results for LLM prompts."""
    if not results:
        return "No sources available."

    formatted = []
    for i, result in enumerate(results, 1):
        score_str = f" (relevance: {result.relevance_score:.2f})" if result.relevance_score else ""
        formatted.append(f"{i}. [{result.title}]({result.url}){score_str}\n   {result.snippet}")
    return "\n\n".join(formatted)


def create_research_graph() -> StateGraph:
    """Create the research subagent graph with ReAct search pattern.

    Returns:
        Compiled research graph
    """
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("init_config", init_config_node)
    graph.add_node("search_agent", search_agent_node)
    graph.add_node("search_tools", search_tools_node)
    graph.add_node("wait_interrupt", wait_interrupt_node)
    graph.add_node("collect_sources", collect_sources_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("write", write_node)

    # Set entry point
    graph.set_entry_point("init_config")

    # Build graph
    graph.add_edge("init_config", "search_agent")

    # ReAct loop for search
    graph.add_conditional_edges(
        "search_agent",
        should_continue_search,
        {
            "tools": "search_tools",
            "collect": "collect_sources",
        },
    )
    graph.add_conditional_edges(
        "search_tools",
        should_wait_or_search,
        {
            "wait_interrupt": "wait_interrupt",
            "search_agent": "search_agent",
        },
    )
    graph.add_edge("wait_interrupt", "search_agent")

    # Analysis pipeline
    graph.add_edge("collect_sources", "analyze")

    # Conditional synthesis
    graph.add_conditional_edges(
        "analyze",
        should_synthesize,
        {
            "synthesize": "synthesize",
            "write": "write",
        },
    )

    graph.add_edge("synthesize", "write")
    graph.add_edge("write", END)

    return graph.compile()


# Compiled subgraph for use by supervisor
research_subgraph = create_research_graph()
