"""Deep Research Skill with ReAct loop for multi-step research.

Replaces the standalone research agent. Uses a ReAct loop where the LLM
decides when to search, browse, analyze, and write — instead of fixed stages.
"""

import hashlib
import json
import operator
import threading
from typing import Annotated, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.agents import events as agent_events
from app.agents.context_compression import (
    CompressionConfig,
    ContextCompressor,
    inject_summary_as_context,
)
from app.agents.prompts import (
    get_report_prompt,
)
from app.agents.scenarios import get_scenario_config
from app.agents.skills.skill_base import Skill, SkillMetadata, SkillParameter, SkillState
from app.agents.state import _override_reducer
from app.agents.tools import (
    get_react_config,
    get_tools_for_agent,
)
from app.agents.tools.react_tool import truncate_messages_to_budget
from app.agents.tools.tool_pipeline import (
    ResearchToolHooks,
    execute_tools_batch,
)
from app.ai.llm import extract_text_from_content, llm_service
from app.core.logging import get_logger
from app.guardrails.scanners.output_scanner import output_scanner
from app.models.schemas import ResearchScenario

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Depth configuration
# ---------------------------------------------------------------------------

DEPTH_CONFIG = {
    "fast": {
        "max_iterations": 5,
        "report_length": "concise",
        "analysis_detail": "brief",
        "search_depth": "basic",
        "system_guidance": "Be efficient. Focus on the most relevant 2-3 sources.",
    },
    "deep": {
        "max_iterations": 20,
        "report_length": "detailed and comprehensive",
        "analysis_detail": "in-depth with follow-up questions",
        "search_depth": "advanced",
        "system_guidance": (
            "Be thorough. Explore multiple angles, verify claims, cross-reference sources."
        ),
    },
}

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------


class DeepResearchSkillState(SkillState, total=False):
    """State for deep research skill execution."""

    # Research config
    query: str
    scenario: str
    depth: str
    depth_config: dict
    system_prompt: str
    report_structure: list[str]

    # ReAct loop state
    lc_messages: Annotated[list[BaseMessage], operator.add]
    tool_iterations: Annotated[int, _override_reducer]
    consecutive_errors: Annotated[int, _override_reducer]
    research_complete: Annotated[bool, _override_reducer]

    # Accumulated research data — uses operator.add so nodes return only NEW sources
    sources: Annotated[list[dict], operator.add]
    findings: str  # Summary from finish_research

    # Report output
    report_chunks: list[str]

    # Context passthrough
    locale: str
    provider: str | None
    model: str | None
    tier: Any | None
    messages: list[dict[str, Any]]  # Conversation history from parent
    # Uses operator.add so each node's events accumulate correctly
    pending_events: Annotated[list[dict[str, Any]], operator.add]

    # Anti-repetition: hashes of recent tool calls to detect stuck loops.
    # Uses override reducer so the node always sets the full list.
    last_tool_calls_hash: Annotated[list[str], _override_reducer]

    # Context compression summary from previous iterations
    context_summary: str | None


# ---------------------------------------------------------------------------
# finish_research signal tool
# ---------------------------------------------------------------------------


class FinishResearchInput(BaseModel):
    findings_summary: str = Field(
        description=(
            "Detailed, structured summary of ALL key findings from your research. "
            "Must include specific data points, statistics, and concrete facts — "
            "not just high-level conclusions. Organize by theme or topic area. "
            "This is the primary input for the report writer, so be thorough."
        )
    )
    key_data_points: str = Field(
        default="",
        description=(
            "Specific statistics, numbers, dates, and quantitative facts found "
            "during research. List each data point on its own line."
        ),
    )
    source_highlights: str = Field(
        default="",
        description=(
            "The most important claims or quotes from each key source. "
            "Format: 'Source Title: key claim or finding' — one per line."
        ),
    )
    confidence: str = Field(description="Overall confidence in findings: high, medium, or low")


@tool("finish_research", args_schema=FinishResearchInput)
def finish_research_tool(
    findings_summary: str,
    confidence: str,
    key_data_points: str = "",
    source_highlights: str = "",
) -> str:
    """Signal that research is complete and you are ready to write the final report.

    Call this when you have gathered enough information to write a comprehensive
    research report. You MUST provide a detailed, structured findings summary that
    includes specific data points, evidence, and per-source highlights — not just
    a brief paragraph. The report quality depends on the depth of your summary.
    """
    return json.dumps(
        {
            "status": "research_complete",
            "findings_summary": findings_summary,
            "key_data_points": key_data_points,
            "source_highlights": source_highlights,
            "confidence": confidence,
        }
    )


# ---------------------------------------------------------------------------
# Tool caching
# ---------------------------------------------------------------------------

_cached_tools: list[BaseTool] | None = None
_cache_lock = threading.Lock()


def _get_research_tools() -> list[BaseTool]:
    """Get tools for the research skill, cached thread-safely."""
    global _cached_tools
    if _cached_tools is None:
        with _cache_lock:
            if _cached_tools is None:
                # Get research agent tools (search, browser, image, skill, hitl)
                base_tools = get_tools_for_agent("research", include_handoffs=False)
                # Add execute_code and shell_exec for data-heavy research
                from app.agents.tools.code_execution import execute_code
                from app.agents.tools.shell_tools import shell_exec

                extra_tools = [execute_code, shell_exec]
                # Add the finish_research signal tool
                all_tools = base_tools + extra_tools + [finish_research_tool]
                # Remove handoff tools (skills handle this differently)
                all_tools = [t for t in all_tools if not t.name.startswith("handoff_to_")]
                # Sort for KV-cache consistency
                _cached_tools = sorted(all_tools, key=lambda t: t.name)
    return _cached_tools


def _clear_tool_cache() -> None:
    """Clear the cached tool list. Useful for testing and hot-reload."""
    global _cached_tools
    with _cache_lock:
        _cached_tools = None


# ---------------------------------------------------------------------------
# Source formatting helper
# ---------------------------------------------------------------------------


def _emit_usage_event(response: AIMessage, pending_events: list[dict]) -> None:
    """Extract usage_metadata from an LLM response and append a usage event."""
    usage_meta = getattr(response, "usage_metadata", None)
    if not usage_meta or not isinstance(usage_meta, dict):
        return

    input_tokens = usage_meta.get("input_tokens", 0)
    output_tokens = usage_meta.get("output_tokens", 0)
    if input_tokens == 0 and output_tokens == 0:
        return

    cached_tokens = 0
    input_detail = usage_meta.get("input_token_details") or {}
    cached_tokens = input_detail.get("cache_read", 0) or input_detail.get("cached", 0)

    resp_meta = getattr(response, "response_metadata", {}) or {}
    model_name = resp_meta.get("model_name", "")

    try:
        from app.services.usage_tracker import calculate_cost

        cost = calculate_cost(model_name, input_tokens, output_tokens, cached_tokens)
    except Exception:
        cost = 0.0

    pending_events.append(
        agent_events.usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            model=model_name,
            tier="research",
        )
    )


def _format_sources(sources: list[dict]) -> str:
    """Format source dicts with [N] reference numbers for inline citation matching.

    Output format:
        [1] Title — URL
            Key excerpt: "snippet text..."
            Relevance: 0.85
    """
    if not sources:
        return "No sources available."
    formatted = []
    for i, s in enumerate(sources, 1):
        title = s.get("title", "Untitled")
        url = s.get("url", "")
        snippet = s.get("snippet", "")
        score = s.get("relevance_score")
        lines = [f"[{i}] {title} — {url}"]
        if snippet:
            lines.append(f'    Key excerpt: "{snippet}"')
        if score:
            lines.append(f"    Relevance: {score:.2f}")
        formatted.append("\n".join(lines))
    return "\n\n".join(formatted)


def _extract_research_context(lc_messages: list[BaseMessage], max_chars: int = 8000) -> str:
    """Extract rich research data from tool results in message history.

    Scans lc_messages for ToolMessage content from web_search, browser_navigate,
    and browser tools, extracts key snippets, and builds structured research notes
    that the report writer can cite.

    Args:
        lc_messages: The accumulated LangChain messages from the research loop.
        max_chars: Maximum total characters for the extracted context.

    Returns:
        Structured research notes string, or empty string if nothing useful found.
    """
    notes: list[str] = []
    total_chars = 0
    # Map tool_call_id -> tool_name from AIMessage tool_calls for attribution
    tool_call_names: dict[str, str] = {}
    for msg in lc_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_call_names[tc["id"]] = tc["name"]

    research_tool_prefixes = ("web_search", "browser_navigate", "browser_screenshot")

    for msg in lc_messages:
        if not isinstance(msg, ToolMessage):
            continue
        tool_name = tool_call_names.get(msg.tool_call_id, "")
        if not any(tool_name.startswith(prefix) for prefix in research_tool_prefixes):
            continue

        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if not content or len(content) < 50:
            continue

        # Try to parse JSON tool results for structured extraction
        snippet = ""
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                # web_search results typically have a "results" list
                results = data.get("results", [])
                if results and isinstance(results, list):
                    parts = []
                    for r in results[:5]:
                        title = r.get("title", "")
                        url = r.get("url", "")
                        text = r.get("snippet", r.get("content", ""))[:300]
                        if title or text:
                            parts.append(f"  - {title} ({url}): {text}")
                    if parts:
                        snippet = f"[{tool_name}]\n" + "\n".join(parts)
                else:
                    # Single result or page content
                    text = data.get("content", data.get("text", data.get("markdown", "")))
                    if text:
                        snippet = f"[{tool_name}] {text[:500]}"
            elif isinstance(data, list):
                parts = []
                for item in data[:5]:
                    if isinstance(item, dict):
                        title = item.get("title", "")
                        text = item.get("snippet", item.get("content", ""))[:300]
                        if title or text:
                            parts.append(f"  - {title}: {text}")
                if parts:
                    snippet = f"[{tool_name}]\n" + "\n".join(parts)
        except (json.JSONDecodeError, TypeError):
            # Plain text content — take a meaningful excerpt
            snippet = f"[{tool_name}] {content[:500]}"

        if snippet:
            if total_chars + len(snippet) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    snippet = snippet[:remaining] + "..."
                    notes.append(snippet)
                break
            notes.append(snippet)
            total_chars += len(snippet)

    if not notes:
        return ""

    return "---\n".join(notes)


# ---------------------------------------------------------------------------
# DeepResearchSkill
# ---------------------------------------------------------------------------


class DeepResearchSkill(Skill):
    """Deep research skill with ReAct loop."""

    metadata = SkillMetadata(
        id="deep_research",
        name="Deep Research",
        version="1.0.0",
        description=(
            "Multi-step deep research with ReAct loop. Searches the web, "
            "browses pages, analyzes data, and writes comprehensive reports."
        ),
        category="research",
        parameters=[
            SkillParameter(
                name="query",
                type="string",
                description="The research question or topic",
                required=True,
            ),
            SkillParameter(
                name="scenario",
                type="string",
                description="Research scenario: academic, market, technical, news",
                required=False,
                default="academic",
            ),
            SkillParameter(
                name="depth",
                type="string",
                description="Research depth: fast or deep",
                required=False,
                default="deep",
            ),
        ],
        output_schema={
            "type": "object",
            "properties": {
                "report": {"type": "string", "description": "The research report"},
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "snippet": {"type": "string"},
                        },
                    },
                },
                "findings": {"type": "string", "description": "Key findings summary"},
            },
        },
        required_tools=["web_search"],
        risk_level="medium",
        side_effect_level="low",
        data_sensitivity="public",
        network_scope="external",
        max_execution_time_seconds=600,
        max_iterations=25,
        tags=["research", "analysis", "report", "deep_research"],
    )

    def create_graph(self):
        """Build the research ReAct graph."""
        graph = StateGraph(DeepResearchSkillState)

        # Cached LLM+tools binding — reused across react_loop iterations
        # to avoid redundant bind_tools() calls (same LLM + same tools every time).
        _llm_cache: dict[str, Any] = {"instance": None, "key": None}

        # ---- Node 1: init_config ----
        async def init_config(state: DeepResearchSkillState) -> dict:
            params = state.get("input_params", {})
            query = params.get("query", "")
            scenario_str = params.get("scenario", "academic")
            depth_str = params.get("depth", "deep")
            locale = params.get("locale", state.get("locale", "en"))
            provider = params.get("provider", state.get("provider"))
            model = params.get("model", state.get("model"))
            tier = params.get("tier", state.get("tier"))
            messages_from_params = params.get("messages", [])

            # Map scenario string to enum
            scenario_map = {
                "academic": ResearchScenario.ACADEMIC,
                "market": ResearchScenario.MARKET_ANALYSIS,
                "technical": ResearchScenario.TECHNICAL,
                "news": ResearchScenario.NEWS,
            }
            scenario_enum = scenario_map.get(scenario_str, ResearchScenario.ACADEMIC)

            config = get_scenario_config(scenario_enum)
            depth_cfg = dict(DEPTH_CONFIG.get(depth_str, DEPTH_CONFIG["deep"]))

            # Override max_iterations from tier quality profile
            from app.ai.model_tiers import get_quality_profile

            profile = get_quality_profile(state.get("tier"))
            if depth_str == "fast":
                depth_cfg["max_iterations"] = profile.deep_research_max_iters_fast
            else:
                depth_cfg["max_iterations"] = profile.deep_research_max_iters_deep

            # Build system prompt for the ReAct loop
            system_content = f"""You are a deep research agent conducting {config["name"]}.

{config["system_prompt"]}

## Research Guidelines
- {depth_cfg["system_guidance"]}
- Maximum iterations: {depth_cfg["max_iterations"]}
- Target report length: {depth_cfg["report_length"]}

## Available Actions
You have tools for web search, browsing pages, code execution, and more.
Use them strategically to gather comprehensive information.

## When to Finish
When you have gathered sufficient information \
(typically {3 if depth_str == "fast" else "8-15"} quality sources),
call the `finish_research` tool. The report quality depends entirely on your summary.

When calling `finish_research`, you MUST provide:
- `findings_summary`: A detailed, structured summary organized by theme. Include specific \
facts, statistics, and concrete evidence — not just high-level conclusions.
- `key_data_points`: List specific numbers, dates, statistics, and quantitative facts you found.
- `source_highlights`: For each important source, note its key claim or most valuable finding.
- `confidence`: Your overall confidence level.

## Important
- Search broadly first, then deep-dive into the most relevant results
- Use browser tools to read full articles when snippets aren't enough
- Use code execution if you need to analyze data or create charts
- Cross-reference information across multiple sources
- Track source quality and reliability
- When you find important data, note specific numbers and facts for your findings summary"""

            lc_messages = [
                SystemMessage(
                    content=system_content,
                    additional_kwargs={"cache_control": {"type": "ephemeral"}},
                ),
            ]

            # Inject user memories
            user_id = state.get("user_id")
            if user_id:
                try:
                    from app.services.memory_service import get_memory_store

                    memory_text = get_memory_store().format_memories_for_prompt(user_id)
                    if memory_text:
                        lc_messages.append(SystemMessage(content=memory_text))
                except Exception:
                    pass

            lc_messages.append(HumanMessage(content=f"Research topic: {query}"))

            pending_events = [
                agent_events.config(depth=depth_str, scenario=scenario_str),
                agent_events.stage("search", "Starting research...", "running"),
            ]

            logger.info(
                "deep_research_skill_init",
                query=query[:80],
                scenario=scenario_str,
                depth=depth_str,
            )

            return {
                "query": query,
                "scenario": scenario_str,
                "depth": depth_str,
                "depth_config": depth_cfg,
                "system_prompt": config["system_prompt"],
                "report_structure": config["report_structure"],
                "lc_messages": lc_messages,
                "sources": [],
                "tool_iterations": 0,
                "consecutive_errors": 0,
                "research_complete": False,
                "findings": "",
                "locale": locale,
                "provider": provider,
                "model": model,
                "tier": tier,
                "messages": messages_from_params,
                "pending_events": pending_events,
            }

        # ---- Node 2: react_loop ----
        async def react_loop(state: DeepResearchSkillState) -> dict:
            depth_cfg = state.get("depth_config") or DEPTH_CONFIG["deep"]
            max_iters = depth_cfg.get("max_iterations", 20)
            tool_iterations = state.get("tool_iterations", 0)
            consecutive_errors = state.get("consecutive_errors", 0)

            pending_events = []

            # Circuit breaker: too many consecutive errors
            if consecutive_errors >= 3:
                logger.warning("deep_research_circuit_breaker", errors=consecutive_errors)
                pending_events.append(
                    agent_events.stage("search", "Too many errors, finishing research", "completed")
                )
                return {
                    "research_complete": True,
                    "findings": _build_fallback_findings(state),
                    "pending_events": pending_events,
                }

            # Iteration limit reached
            if tool_iterations >= max_iters:
                logger.info("deep_research_max_iterations", iterations=tool_iterations)
                pending_events.append(
                    agent_events.stage("search", "Maximum iterations reached", "completed")
                )
                return {
                    "research_complete": True,
                    "findings": _build_fallback_findings(state),
                    "pending_events": pending_events,
                }

            # Get LLM and tools (cached across iterations)
            provider = state.get("provider")
            tier = state.get("tier")
            model_override = state.get("model")
            cache_key = f"{provider}:{tier}:{model_override}"

            if _llm_cache["key"] != cache_key or _llm_cache["instance"] is None:
                llm = llm_service.choose_llm_for_task(
                    "research", provider=provider, tier_override=tier, model_override=model_override
                )
                all_tools = _get_research_tools()
                _llm_cache["instance"] = llm.bind_tools(all_tools)
                _llm_cache["key"] = cache_key

            llm_with_tools = _llm_cache["instance"]

            # Context management: compression first, truncation as fallback
            lc_messages = list(state.get("lc_messages") or [])
            react_config = get_react_config("research")
            existing_summary = state.get("context_summary")
            new_context_summary = None

            # Attempt LLM-based compression before truncation (preserves URLs, findings)
            from app.config import settings

            compress_threshold = settings.context_compression_preserve_recent
            if settings.context_compression_enabled and len(lc_messages) > compress_threshold:
                compression_config = CompressionConfig(
                    token_threshold=settings.context_compression_token_threshold,
                    preserve_recent=settings.context_compression_preserve_recent,
                    enabled=True,
                )
                compressor = ContextCompressor(compression_config)
                try:
                    new_summary, lc_messages = await compressor.compress(
                        lc_messages,
                        existing_summary,
                        provider or "anthropic",
                        locale=state.get("locale", "en"),
                    )
                    if new_summary:
                        lc_messages = inject_summary_as_context(lc_messages, new_summary)
                        new_context_summary = new_summary
                        logger.info(
                            "deep_research_context_compressed",
                            summary_length=len(new_summary),
                        )
                except Exception as e:
                    logger.warning("deep_research_compression_skipped", error=str(e))

            # Fallback: hard truncation to stay within token budget
            lc_messages, was_truncated = truncate_messages_to_budget(
                lc_messages,
                max_tokens=react_config.max_message_tokens,
                preserve_recent=react_config.preserve_recent_messages,
            )
            if was_truncated:
                logger.info("deep_research_messages_truncated")

            # Invoke LLM
            try:
                response = await llm_with_tools.ainvoke(lc_messages)
            except Exception as e:
                logger.error("deep_research_llm_error", error=str(e))
                return {
                    "lc_messages": [AIMessage(content=f"Error: {str(e)}")],
                    "consecutive_errors": consecutive_errors + 1,
                    "pending_events": pending_events,
                }

            # Extract usage metadata for cost tracking
            _emit_usage_event(response, pending_events)

            new_messages = [response]

            # Check if LLM wants to call tools
            if not response.tool_calls:
                # No tool calls — LLM wants to finish without calling finish_research
                logger.info("deep_research_no_tool_calls_finishing")
                text_content = extract_text_from_content(response.content)
                pending_events.append(
                    agent_events.stage("search", "Research gathering complete", "completed")
                )
                return {
                    "lc_messages": new_messages,
                    "research_complete": True,
                    "findings": text_content or _build_fallback_findings(state),
                    "pending_events": pending_events,
                }

            # Check if finish_research was called
            for tc in response.tool_calls:
                if tc["name"] == "finish_research":
                    findings = tc["args"].get("findings_summary", "")
                    key_data_points = tc["args"].get("key_data_points", "")
                    source_highlights = tc["args"].get("source_highlights", "")
                    confidence = tc["args"].get("confidence", "medium")
                    # Enrich findings with structured data from the tool
                    if key_data_points:
                        findings += f"\n\n### Key Data Points\n{key_data_points}"
                    if source_highlights:
                        findings += f"\n\n### Source Highlights\n{source_highlights}"
                    logger.info(
                        "deep_research_finish_called",
                        confidence=confidence,
                        sources=len(state.get("sources") or []),
                    )
                    # Add tool response message
                    new_messages.append(
                        ToolMessage(
                            content=json.dumps(
                                {
                                    "status": "research_complete",
                                    "message": "Proceeding to write the report.",
                                }
                            ),
                            tool_call_id=tc["id"],
                        )
                    )
                    pending_events.append(
                        agent_events.stage(
                            "search",
                            f"Research complete ({confidence} confidence)",
                            "completed",
                        )
                    )
                    return {
                        "lc_messages": new_messages,
                        "research_complete": True,
                        "findings": findings,
                        "pending_events": pending_events,
                    }

            # Tool calls to execute (not finish_research) — proceed to execute_tools
            result = {
                "lc_messages": new_messages,
                "pending_events": pending_events,
            }
            if new_context_summary:
                result["context_summary"] = new_context_summary
            return result

        # ---- Node 3: execute_tools ----
        async def execute_tools(state: DeepResearchSkillState) -> dict:
            lc_messages = list(state.get("lc_messages") or [])
            tool_iterations = state.get("tool_iterations", 0)
            consecutive_errors = state.get("consecutive_errors", 0)

            pending_events = []
            new_sources = []  # Only new sources (delta) — operator.add handles accumulation
            all_tools = _get_research_tools()
            tool_map = {t.name: t for t in all_tools}
            react_config = get_react_config("research")

            # Get last AI message with tool calls
            last_message = lc_messages[-1] if lc_messages else None
            has_tool_calls = (
                last_message and hasattr(last_message, "tool_calls") and last_message.tool_calls
            )
            if not has_tool_calls:
                return {
                    "tool_iterations": tool_iterations + 1,
                    "pending_events": pending_events,
                }

            # Filter out finish_research from tool calls (handled in react_loop)
            tool_calls = [tc for tc in last_message.tool_calls if tc["name"] != "finish_research"]
            if not tool_calls:
                return {
                    "tool_iterations": tool_iterations + 1,
                    "pending_events": pending_events,
                }

            # Build hooks for source collection
            class SkillResearchHooks(ResearchToolHooks):
                """Inherits source collection from ResearchToolHooks.

                Parent's after_execution() parses web_search results and
                appends them to self.collected_sources automatically.
                """

                pass

            hooks = SkillResearchHooks()

            # Execute tool calls
            tool_messages, batch_events, error_count, _ = await execute_tools_batch(
                tool_calls=tool_calls,
                tool_map=tool_map,
                config=react_config,
                hooks=hooks,
                user_id=state.get("user_id"),
                task_id=state.get("task_id"),
            )

            # Collect new sources (SearchResult dataclasses → dicts)
            raw_sources = hooks.collected_sources
            if raw_sources:
                from dataclasses import asdict

                for src in raw_sources:
                    if isinstance(src, dict):
                        src_dict = src
                    elif hasattr(src, "__dataclass_fields__"):
                        src_dict = asdict(src)
                    else:
                        src_dict = {
                            "title": getattr(src, "title", ""),
                            "url": getattr(src, "url", ""),
                            "snippet": getattr(src, "snippet", ""),
                            "relevance_score": getattr(src, "relevance_score", None),
                        }
                    new_sources.append(src_dict)
                    pending_events.append(
                        agent_events.source(
                            title=src_dict.get("title", ""),
                            url=src_dict.get("url", ""),
                            snippet=src_dict.get("snippet", ""),
                        )
                    )

            # Add tool events
            pending_events.extend(batch_events)

            # Update consecutive errors
            if error_count == len(tool_calls):
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            # --- Anti-Repetition: Detect consecutive identical tool calls ---
            prev_hashes = list(state.get("last_tool_calls_hash", []) or [])
            for tc in tool_calls:
                tc_name = tc.get("name", "")
                tc_args = tc.get("args", {})
                try:
                    args_str = json.dumps(tc_args, sort_keys=True, default=str)
                except (TypeError, ValueError):
                    args_str = str(tc_args)
                call_hash = hashlib.md5(f"{tc_name}:{args_str}".encode()).hexdigest()[:12]
                prev_hashes.append(call_hash)

            # Keep only last 5 hashes
            if len(prev_hashes) > 5:
                prev_hashes = prev_hashes[-5:]

            # Count consecutive identical hashes from the end
            variation_message = None
            if len(prev_hashes) >= 2:
                last_hash = prev_hashes[-1]
                consecutive_identical = 0
                for h in reversed(prev_hashes):
                    if h == last_hash:
                        consecutive_identical += 1
                    else:
                        break

                if consecutive_identical >= 3:
                    variation_message = SystemMessage(
                        content=(
                            "[System: Repetition detected — you have called "
                            "the same tool with identical "
                            f"arguments {consecutive_identical} times. "
                            "The previous approach is NOT working. You MUST change strategy:\n"
                            "- Use DIFFERENT search terms or a different tool\n"
                            "- Try browsing a specific URL instead of searching\n"
                            "- Broaden or narrow your research angle\n"
                            "- If you have enough information, call finish_research\n"
                            "Do NOT retry the same call again."
                        )
                    )
                    logger.warning(
                        "deep_research_anti_repetition_force",
                        consecutive_count=consecutive_identical,
                    )
                elif consecutive_identical >= 2:
                    repeated_tool = (
                        tool_calls[-1].get("name", "unknown") if tool_calls else "unknown"
                    )
                    variation_message = SystemMessage(
                        content=(
                            f"[System: You have called {repeated_tool} with identical arguments "
                            f"{consecutive_identical} times. Try varying your approach:\n"
                            "- Use different search queries or keywords\n"
                            "- Try a different tool or browse a specific page\n"
                            "- Consider if you have enough info to call finish_research"
                        )
                    )
                    logger.info(
                        "deep_research_anti_repetition_hint",
                        consecutive_count=consecutive_identical,
                        tool=repeated_tool,
                    )

            result_messages = tool_messages
            if variation_message:
                result_messages = tool_messages + [variation_message]

            return {
                "lc_messages": result_messages,
                "sources": new_sources,  # Delta only — operator.add accumulates
                "tool_iterations": tool_iterations + 1,
                "consecutive_errors": consecutive_errors,
                "pending_events": pending_events,
                "last_tool_calls_hash": prev_hashes,
            }

        # ---- Node 4: write_report ----
        async def write_report(state: DeepResearchSkillState) -> dict:
            query = state.get("query", "")
            findings = state.get("findings", "")
            sources = state.get("sources") or []
            scenario_prompt = state.get("system_prompt", "")
            report_structure = state.get("report_structure") or []
            depth_cfg = state.get("depth_config") or {}
            locale = state.get("locale", "en")
            depth = state.get("depth", "deep")

            pending_events = [
                agent_events.stage("write", "Writing research report...", "running"),
            ]

            sources_text = _format_sources(sources)

            # Extract rich context from research tool results
            lc_messages = state.get("lc_messages") or []
            research_context = _extract_research_context(lc_messages)

            report_prompt = get_report_prompt(
                query=query,
                combined_findings=findings,
                sources_text=sources_text,
                report_structure=report_structure,
                report_length=depth_cfg.get("report_length", "comprehensive"),
                locale=locale,
                research_context=research_context,
            )

            # Dedicated report-writing system prompt (replaces the scenario research prompt)
            report_system_prompt = f"""You are an expert research report writer. Your task is to \
produce a professional, well-structured research report based on the provided findings and sources.

## Research Context
{scenario_prompt}

## Report Writing Guidelines
- Write in a clear, authoritative, and analytical tone
- Use professional markdown formatting: headers, paragraphs, tables, block quotes
- Provide substantive analysis in every section — avoid thin summaries or bullet-only sections
- Support claims with inline citations using [1], [2], etc. matching the numbered sources
- End with a ## References section listing all cited sources with their URLs
- {"800–1500 words (fast)" if depth == "fast" else "2000–4000 words (deep research)"}
- Compare perspectives across multiple sources where relevant
- Include specific data points, statistics, and evidence
- Note limitations, gaps, and areas for further research"""

            provider = state.get("provider")
            tier = state.get("tier")
            model_override = state.get("model")
            llm = llm_service.choose_llm_for_task(
                "research", provider=provider, tier_override=tier, model_override=model_override
            )

            report_chunks = []
            last_chunk = None
            try:
                async for chunk in llm.astream(
                    [
                        SystemMessage(content=report_system_prompt),
                        HumanMessage(content=report_prompt),
                    ]
                ):
                    last_chunk = chunk
                    if chunk.content:
                        content = extract_text_from_content(chunk.content)
                        if content:
                            report_chunks.append(content)
                            pending_events.append(agent_events.token(content))

                # Extract usage from the final streamed chunk
                if last_chunk is not None:
                    _emit_usage_event(last_chunk, pending_events)

                logger.info("deep_research_report_completed", query=query[:50])
            except Exception as e:
                logger.error("deep_research_report_failed", error=str(e))
                pending_events.append(agent_events.token(f"\n\nError generating report: {str(e)}"))

            # Apply output guardrails
            report_text = "".join(report_chunks)
            scan_result = await output_scanner.scan(report_text, query)
            if scan_result.blocked:
                logger.warning("deep_research_output_blocked")
                report_text = (
                    "I apologize, but the research report could not be delivered "
                    "due to content policy. Please try a different research topic."
                )
            elif scan_result.sanitized_content:
                report_text = scan_result.sanitized_content

            pending_events.append(agent_events.stage("write", "Report complete", "completed"))

            return {
                "output": {
                    "report": report_text,
                    "sources": sources,
                    "findings": findings,
                },
                "pending_events": pending_events,
            }

        # ---- Conditional edges ----
        def should_continue(state: DeepResearchSkillState) -> str:
            if state.get("research_complete"):
                return "write_report"
            return "execute_tools"

        # ---- Build graph ----
        graph.add_node("init_config", init_config)
        graph.add_node("react_loop", react_loop)
        graph.add_node("execute_tools", execute_tools)
        graph.add_node("write_report", write_report)

        graph.set_entry_point("init_config")
        graph.add_edge("init_config", "react_loop")
        graph.add_conditional_edges(
            "react_loop",
            should_continue,
            {
                "execute_tools": "execute_tools",
                "write_report": "write_report",
            },
        )
        graph.add_edge("execute_tools", "react_loop")
        graph.add_edge("write_report", END)

        return graph.compile()


def _build_fallback_findings(state: DeepResearchSkillState) -> str:
    """Build a findings summary from accumulated sources when finish_research wasn't called."""
    sources = state.get("sources") or []
    query = state.get("query", "")
    if sources:
        source_list = "\n".join(
            f"- {s.get('title', 'Untitled')}: {s.get('snippet', '')[:200]}" for s in sources[:10]
        )
        return f"Research on '{query}' found {len(sources)} sources:\n{source_list}"
    return f"Research on '{query}' — limited sources found."
