# Deep Research Skill — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert deep research from a standalone agent into a builtin skill with ReAct loop, eliminating dual-agent routing and enabling composability.

**Architecture:** DeepResearchSkill extends `Skill` with an internal LangGraph (init_config → react_loop ⇄ execute_tools → write_report). The task agent invokes it via `invoke_skill` like any other skill. Supervisor routing simplified to task-only.

**Tech Stack:** LangGraph StateGraph, LangChain tools, existing skill infrastructure (SkillExecutor, SkillRegistry)

**Design Doc:** `docs/plans/2026-03-03-deep-research-skill-design.md`

---

### Task 1: Create the DeepResearchSkill

**Files:**
- Create: `backend/app/agents/skills/builtin/deep_research_skill.py`

**Step 1: Create the skill file with state, metadata, and complete graph implementation**

Create `backend/app/agents/skills/builtin/deep_research_skill.py` with:

```python
"""Deep Research Skill with ReAct loop for multi-step research.

Replaces the standalone research agent. Uses a ReAct loop where the LLM
decides when to search, browse, analyze, and write — instead of fixed stages.
"""

import json
import operator
import threading
from typing import Any, Annotated

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
from app.agents.prompts import (
    get_report_prompt,
)
from app.agents.scenarios import get_scenario_config
from app.agents.skills.skill_base import Skill, SkillMetadata, SkillParameter, SkillState
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
from app.services.search import SearchResult

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
            "Be thorough. Explore multiple angles, verify claims, "
            "cross-reference sources."
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
    tool_iterations: int
    consecutive_errors: int
    research_complete: bool

    # Accumulated research data
    sources: list[dict]
    findings: str  # Summary from finish_research

    # Report output
    report_chunks: list[str]

    # Context passthrough
    locale: str
    provider: str | None
    model: str | None
    tier: Any | None
    messages: list[dict[str, Any]]  # Conversation history from parent
    pending_events: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# finish_research signal tool
# ---------------------------------------------------------------------------


class FinishResearchInput(BaseModel):
    findings_summary: str = Field(
        description="Comprehensive summary of all key findings from your research"
    )
    confidence: str = Field(
        description="Overall confidence in findings: high, medium, or low"
    )


@tool("finish_research", args_schema=FinishResearchInput)
def finish_research_tool(findings_summary: str, confidence: str) -> str:
    """Signal that research is complete and you are ready to write the final report.

    Call this when you have gathered enough information to write a comprehensive
    research report. Provide a thorough summary of your key findings.
    """
    return json.dumps({
        "status": "research_complete",
        "findings_summary": findings_summary,
        "confidence": confidence,
    })


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


# ---------------------------------------------------------------------------
# Source formatting helper
# ---------------------------------------------------------------------------


def _format_sources(sources: list[dict]) -> str:
    """Format source dicts for LLM prompts."""
    if not sources:
        return "No sources available."
    formatted = []
    for i, s in enumerate(sources, 1):
        title = s.get("title", "Untitled")
        url = s.get("url", "")
        snippet = s.get("snippet", "")
        score = s.get("relevance_score")
        score_str = f" (relevance: {score:.2f})" if score else ""
        formatted.append(f"{i}. [{title}]({url}){score_str}\n   {snippet}")
    return "\n\n".join(formatted)


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
            depth_cfg = DEPTH_CONFIG.get(depth_str, DEPTH_CONFIG["deep"])

            # Build system prompt for the ReAct loop
            system_content = f"""You are a deep research agent conducting {config['name']}.

{config['system_prompt']}

## Research Guidelines
- {depth_cfg['system_guidance']}
- Maximum iterations: {depth_cfg['max_iterations']}
- Target report length: {depth_cfg['report_length']}

## Available Actions
You have tools for web search, browsing pages, code execution, and more.
Use them strategically to gather comprehensive information.

## When to Finish
When you have gathered sufficient information (typically {3 if depth_str == 'fast' else '8-15'} quality sources),
call the `finish_research` tool with a thorough summary of your findings.
The system will then generate a structured report based on your findings.

## Important
- Search broadly first, then deep-dive into the most relevant results
- Use browser tools to read full articles when snippets aren't enough
- Use code execution if you need to analyze data or create charts
- Cross-reference information across multiple sources
- Track source quality and reliability"""

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

            # Get LLM and tools
            provider = state.get("provider")
            tier = state.get("tier")
            model_override = state.get("model")
            llm = llm_service.choose_llm_for_task(
                "research", provider=provider, tier_override=tier, model_override=model_override
            )

            all_tools = _get_research_tools()
            tool_map = {t.name: t for t in all_tools}
            llm_with_tools = llm.bind_tools(all_tools)

            # Truncate messages to budget
            lc_messages = list(state.get("lc_messages") or [])
            react_config = get_react_config("research")
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
                    confidence = tc["args"].get("confidence", "medium")
                    logger.info(
                        "deep_research_finish_called",
                        confidence=confidence,
                        sources=len(state.get("sources") or []),
                    )
                    # Add tool response message
                    new_messages.append(
                        ToolMessage(
                            content=json.dumps({
                                "status": "research_complete",
                                "message": "Proceeding to write the report.",
                            }),
                            tool_call_id=tc["id"],
                        )
                    )
                    pending_events.append(
                        agent_events.stage("search", f"Research complete ({confidence} confidence)", "completed")
                    )
                    return {
                        "lc_messages": new_messages,
                        "research_complete": True,
                        "findings": findings,
                        "pending_events": pending_events,
                    }

            # Tool calls to execute (not finish_research) — proceed to execute_tools
            return {
                "lc_messages": new_messages,
                "pending_events": pending_events,
            }

        # ---- Node 3: execute_tools ----
        async def execute_tools(state: DeepResearchSkillState) -> dict:
            lc_messages = list(state.get("lc_messages") or [])
            sources = list(state.get("sources") or [])
            tool_iterations = state.get("tool_iterations", 0)
            consecutive_errors = state.get("consecutive_errors", 0)

            pending_events = []
            all_tools = _get_research_tools()
            tool_map = {t.name: t for t in all_tools}
            react_config = get_react_config("research")

            # Get last AI message with tool calls
            last_message = lc_messages[-1] if lc_messages else None
            if not last_message or not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                return {
                    "tool_iterations": tool_iterations + 1,
                    "pending_events": pending_events,
                }

            # Filter out finish_research from tool calls (handled in react_loop)
            tool_calls = [
                tc for tc in last_message.tool_calls
                if tc["name"] != "finish_research"
            ]
            if not tool_calls:
                return {
                    "tool_iterations": tool_iterations + 1,
                    "pending_events": pending_events,
                }

            # Build hooks for source collection
            class SkillResearchHooks(ResearchToolHooks):
                def __init__(self):
                    super().__init__()
                    self.collected_sources = []

                async def after_execution(self, ctx, result):
                    result = await super().after_execution(ctx, result)
                    # Extract sources from search results
                    if ctx.tool_name == "web_search" and not result.is_error:
                        try:
                            from app.agents.tools.validators import extract_search_sources
                            parsed_sources = extract_search_sources(result.message.content)
                            for src in parsed_sources:
                                self.collected_sources.append({
                                    "title": src.get("title", ""),
                                    "url": src.get("url", ""),
                                    "snippet": src.get("snippet", ""),
                                    "relevance_score": src.get("relevance_score"),
                                })
                        except Exception:
                            pass
                    return result

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

            # Collect new sources
            new_sources = hooks.collected_sources
            if new_sources:
                sources.extend(new_sources)
                for src in new_sources:
                    pending_events.append(agent_events.source(
                        title=src.get("title", ""),
                        url=src.get("url", ""),
                        snippet=src.get("snippet", ""),
                    ))

            # Add tool events
            pending_events.extend(batch_events)

            # Update consecutive errors
            if error_count == len(tool_calls):
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            return {
                "lc_messages": tool_messages,
                "sources": sources,
                "tool_iterations": tool_iterations + 1,
                "consecutive_errors": consecutive_errors,
                "pending_events": pending_events,
            }

        # ---- Node 4: write_report ----
        async def write_report(state: DeepResearchSkillState) -> dict:
            query = state.get("query", "")
            findings = state.get("findings", "")
            sources = state.get("sources") or []
            system_prompt = state.get("system_prompt", "")
            report_structure = state.get("report_structure") or []
            depth_cfg = state.get("depth_config") or {}
            locale = state.get("locale", "en")

            pending_events = [
                agent_events.stage("write", "Writing research report...", "running"),
            ]

            sources_text = _format_sources(sources)

            report_prompt = get_report_prompt(
                query=query,
                combined_findings=findings,
                sources_text=sources_text,
                report_structure=report_structure,
                report_length=depth_cfg.get("report_length", "comprehensive"),
                locale=locale,
            )

            provider = state.get("provider")
            tier = state.get("tier")
            model_override = state.get("model")
            llm = llm_service.choose_llm_for_task(
                "research", provider=provider, tier_override=tier, model_override=model_override
            )

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
                            pending_events.append(agent_events.token(content))

                logger.info("deep_research_report_completed", query=query[:50])
            except Exception as e:
                logger.error("deep_research_report_failed", error=str(e))
                pending_events.append(
                    agent_events.token(f"\n\nError generating report: {str(e)}")
                )

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

            pending_events.append(
                agent_events.stage("write", "Report complete", "completed")
            )

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
            f"- {s.get('title', 'Untitled')}: {s.get('snippet', '')[:200]}"
            for s in sources[:10]
        )
        return f"Research on '{query}' found {len(sources)} sources:\n{source_list}"
    return f"Research on '{query}' — limited sources found."
```

**Step 2: Register in builtin/__init__.py**

Modify `backend/app/agents/skills/builtin/__init__.py` — add import and export:

```python
# Add import
from app.agents.skills.builtin.deep_research_skill import DeepResearchSkill

# Add to __all__
__all__ = [
    "WebResearchSkill",
    "DataAnalysisSkill",
    "ImageGenerationSkill",
    "CodeGenerationSkill",
    "TaskPlanningSkill",
    "AppBuilderSkill",
    "SlideGenerationSkill",
    "DeepResearchSkill",  # NEW
]
```

**Step 3: Verify skill loads**

Run: `cd /Users/feihe/Workspace/HyperAgent && python -c "from app.agents.skills.builtin.deep_research_skill import DeepResearchSkill; s = DeepResearchSkill(); print(f'Skill: {s.metadata.id}, params: {[p.name for p in s.metadata.parameters]}')"` from backend dir.

Expected: `Skill: deep_research, params: ['query', 'scenario', 'depth']`

---

### Task 2: Add Research to Task Agent Direct Skill Modes

**Files:**
- Modify: `backend/app/agents/subagents/task.py:316-353`

**Step 1: Add research to _DIRECT_SKILL_MODES**

In `backend/app/agents/subagents/task.py`, at line 333, add the research entry:

```python
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
    "research": {                          # NEW
        "skill_id": "deep_research",
        "param_key": "query",
    },
}
```

**Step 2: Add context passing for deep_research skill**

After the existing data_analysis context passing block (line 343-353), add a similar block for deep_research. Find the `if skill_spec["skill_id"] == "data_analysis":` block and add after it:

```python
# Pass scenario, depth, and context for deep_research skill
if skill_spec["skill_id"] == "deep_research":
    skill_params["scenario"] = state.get("scenario", "academic")
    depth = state.get("depth")
    if depth:
        # Handle both enum and string values
        skill_params["depth"] = depth.value if hasattr(depth, "value") else depth
    else:
        skill_params["depth"] = "deep"
    skill_params["locale"] = state.get("locale", "en")
    skill_params["provider"] = state.get("provider")
    skill_params["model"] = state.get("model")
    history = state.get("messages") or []
    if history:
        skill_params["messages"] = history[-6:]
```

---

### Task 3: Update API to Route Research Through Supervisor

**Files:**
- Modify: `backend/app/api/query.py:162-169, 582-634`

**Step 1: Add RESEARCH to _CHAT_MODES**

At line 162-168 in `query.py`, add `QueryMode.RESEARCH` to `_CHAT_MODES`:

```python
_CHAT_MODES = (
    QueryMode.TASK,
    QueryMode.APP,
    QueryMode.DATA,
    QueryMode.IMAGE,
    QueryMode.SLIDE,
    QueryMode.RESEARCH,  # NEW — research now flows through supervisor
)
```

**Step 2: Remove the worker queue else-branch for research**

Delete lines 582-634 (the `else:` block that enqueues to worker queue and returns `EventSourceResponse(research_stream_from_worker(task_id))`).

The research mode is now handled by the `if request.mode in _CHAT_MODES:` branch just like other modes, flowing through the supervisor → task agent → deep_research skill.

**Step 3: Ensure supervisor input includes scenario and depth**

In the supervisor input construction within the streaming block (around lines 210-260 where `agent_supervisor.run()` is called), verify that `scenario` and `depth` are passed through from the request. Check the existing supervisor input and add if missing:

```python
# In the run_input dict construction, ensure these are included:
"scenario": request.scenario,
"depth": request.depth,
```

---

### Task 4: Simplify Supervisor — Remove Research Routing

**Files:**
- Modify: `backend/app/agents/supervisor.py:247-290, 502-611`

**Step 1: Remove research_post_node function**

Delete the `research_post_node` function at lines 247-290.

**Step 2: Simplify create_supervisor_graph**

In `create_supervisor_graph()` (line 502+):

a) Remove the research_node inner function (lines 522-577)
b) Remove `graph.add_node("research", research_node)` (line 579)
c) Remove `graph.add_node("research_post", research_post_node)` (line 580)
d) Simplify the conditional edges from router:

**Before** (lines 588-595):
```python
graph.add_conditional_edges(
    "router",
    select_agent,
    {
        AgentType.TASK.value: "task",
        AgentType.RESEARCH.value: "research",
    },
)
```

**After:**
```python
# All queries route to task agent (research is now a skill)
graph.add_edge("router", "task")
```

e) Remove `graph.add_edge("research", "research_post")` (line 598)
f) Simplify the handoff checking loop (lines 601-609):

**Before:**
```python
for agent in ["task", "research_post"]:
    graph.add_conditional_edges(...)
```

**After:**
```python
graph.add_conditional_edges(
    "task",
    check_for_handoff,
    {
        "router": "router",
        END: END,
    },
)
```

**Step 3: Simplify router_node**

In the `router_node` function, since all queries now go to the task agent, the routing logic can be simplified. The router should always set `selected_agent` to `"task"`. However, keep the router_node structure for future extensibility — just ensure it defaults to task.

**Step 4: Remove research subgraph import**

At the top of supervisor.py, remove the import of `research_subgraph` and any imports from `app.agents.subagents.research`.

**Step 5: Remove select_agent function**

Since all queries route to task, the `select_agent` conditional function (lines 294-308) is no longer needed. Remove it.

---

### Task 5: Clean Up State and Research Agent

**Files:**
- Modify: `backend/app/agents/state.py:156-182`
- Delete: `backend/app/agents/subagents/research.py`

**Step 1: Add deprecation note to ResearchState**

In `backend/app/agents/state.py`, add a deprecation docstring to `ResearchState`:

```python
class ResearchState(SupervisorState, total=False):
    """DEPRECATED: Research is now handled by DeepResearchSkill.

    This state is kept for backward compatibility with existing DB records
    and migration scripts. Do not use for new code.
    """
    # ... existing fields ...
```

**Step 2: Delete research.py**

Delete `backend/app/agents/subagents/research.py`.

**Step 3: Remove research subgraph references**

Search for any remaining imports of `research_subgraph` or `create_research_graph` and remove them. Key places:
- `backend/app/agents/supervisor.py` (already handled in Task 4)
- Any `__init__.py` that re-exports research symbols

**Step 4: Update AGENT_TOOL_MAPPING**

In `backend/app/agents/tools/registry.py` (lines 303-310), the `AgentType.RESEARCH` mapping is no longer used for tool retrieval by the research agent directly. However, the deep_research skill uses `get_tools_for_agent("research")` internally, so **keep this mapping** — it now serves the skill instead of the agent.

---

### Task 6: Verify and Lint

**Step 1: Run linter**

Run: `make lint-backend`

Expected: No new errors (fix any import issues from deleted research.py)

**Step 2: Run type check**

Run: `cd backend && python -m mypy app/agents/skills/builtin/deep_research_skill.py --ignore-missing-imports`

Expected: No type errors

**Step 3: Run backend tests**

Run: `make test-backend`

Expected: Tests pass (some research-specific tests may need updating if they reference the old research agent directly)

---

### Task 7: Smoke Test End-to-End

**Step 1: Start backend**

Run: `make dev-backend`

Expected: Server starts without import errors

**Step 2: Verify skill registration**

Run: `curl http://localhost:8080/api/v1/skills | python -m json.tool | grep deep_research`

Expected: `deep_research` appears in the skills list

**Step 3: Test research mode streaming**

Run a test query via curl or the frontend with `mode: "research"`, `scenario: "technical"`, `depth: "fast"` to verify events stream correctly.
