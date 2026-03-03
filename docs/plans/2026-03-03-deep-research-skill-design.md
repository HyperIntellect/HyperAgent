# Deep Research Skill Design

**Date**: 2026-03-03
**Status**: Approved
**Goal**: Convert deep research from a standalone agent into a builtin skill, simplifying the architecture while enabling composability and a unified UX.

## Motivation

The current deep research is a **standalone agent** (`research.py`) — a separate LangGraph subgraph that the supervisor routes to alongside the task agent. This creates complexity:

1. **Dual-agent routing** — The supervisor must decide task vs research at every query
2. **Not composable** — Research can't be chained with other skills (data analysis, app building)
3. **Split frontend** — Research has a separate /task page flow, disconnected from chat

By converting research into a skill, we get:
- **Single agent** — Task agent is the only agent; research is just another skill it invokes
- **Composable** — Task agent can chain research → data analysis → app builder in one flow
- **Unified UX** — Research works inline in chat AND on the dedicated /task page

## Architecture

### Skill Definition

`DeepResearchSkill` extends `Skill` (not `ToolSkill`) with its own LangGraph implementing a ReAct loop.

```python
class DeepResearchSkill(Skill):
    metadata = SkillMetadata(
        id="deep_research",
        name="Deep Research",
        version="1.0.0",
        description="Multi-step deep research with ReAct loop",
        category="research",
        parameters=[
            SkillParameter(name="query", type="string", required=True),
            SkillParameter(name="scenario", type="string", required=False, default="academic"),
            SkillParameter(name="depth", type="string", required=False, default="deep"),
        ],
        required_tools=["web_search", "browser_interact"],
        risk_level="medium",
        side_effect_level="low",
        data_sensitivity="public",
        network_scope="external",
        max_execution_time_seconds=600,
        max_iterations=20,
        tags=["research", "analysis", "report"],
    )
```

### Graph Structure

```
init_config → react_loop ⇄ execute_tools → write_report → END
```

**Three main nodes:**

1. **`init_config`** — Initialize system prompt, depth config (fast/deep), scenario config (academic/market/technical/news). Reuses existing depth and scenario configurations.

2. **`react_loop`** — Core ReAct node. LLM with all tools bound decides what to do at each step: search, browse, analyze with code, ask the user, or call `finish_research` to signal completion.

3. **`execute_tools`** — Execute tool calls from the LLM using existing `execute_tools_batch`. Collects sources from search results. Loops back to `react_loop`.

4. **`write_report`** — Stream the final research report using accumulated sources and the findings summary from `finish_research`. Apply output guardrails.

### Key Design Decision: ReAct Loop

Unlike the current fixed pipeline (search → analyze → synthesize → write), the new design uses a single ReAct loop where the LLM organically decides when to:
- Search for more sources (`web_search`)
- Deep-dive into a page (`browser_interact`)
- Run code to analyze data (`execute_code`)
- Ask the user for clarification (`ask_user`)
- Declare research complete (`finish_research`)

This is more flexible and allows the LLM to adapt its research strategy based on what it finds.

## State

```python
class DeepResearchSkillState(SkillState):
    # Config
    query: str
    scenario: str                # academic | market | technical | news
    depth: str                   # fast | deep
    depth_config: dict           # max_iterations, report_length, system_guidance
    system_prompt: str

    # ReAct loop
    lc_messages: list            # LangChain message history
    tool_iterations: int         # Iteration counter
    consecutive_errors: int      # Circuit breaker
    research_complete: bool      # Set by finish_research tool

    # Research data
    sources: list[dict]          # {title, url, snippet, relevance_score}
    findings: str                # Summary from finish_research

    # Report
    report_chunks: list[str]     # Streamed report content

    # Context passthrough
    locale: str
    provider: str | None
    model: str | None
    tier: Any | None
    messages: list[dict]         # Last N messages from parent conversation
    pending_events: list[dict]
```

## Tools

| Tool | Source | Purpose |
|------|--------|---------|
| `web_search` | Existing | Search the web |
| `browser_interact` | Existing | Navigate/read web pages |
| `get_image` | Existing | Download/capture images |
| `execute_code` | Existing | Run code in sandbox |
| `shell_exec` | Existing | Shell commands for file processing |
| `invoke_skill` | Existing | Compose with other skills |
| `ask_user` | Existing | HITL user input |
| `finish_research` | **New** | Signal tool to end ReAct loop |

### finish_research Tool

```python
class FinishResearchInput(BaseModel):
    findings_summary: str = Field(description="Summary of key findings from research")
    confidence: str = Field(description="Confidence level: high, medium, low")
```

When the LLM calls this tool, the graph transitions from `react_loop` → `write_report`.

## Depth Configuration

```python
DEPTH_CONFIG = {
    "fast": {
        "max_iterations": 5,
        "report_length": "concise",
        "system_guidance": "Be efficient. Focus on the most relevant 2-3 sources.",
    },
    "deep": {
        "max_iterations": 20,
        "report_length": "detailed and comprehensive",
        "system_guidance": "Be thorough. Explore multiple angles, verify claims, cross-reference sources.",
    },
}
```

## Integration Changes

### Supervisor

Keep the supervisor for lifecycle management. Remove research-specific routing:

**Before:**
```
router → select_agent → {task_node OR research_node} → post_node → check_handoff
```

**After:**
```
router → select_agent → task_node → post_node → check_handoff
```

- Remove: `research_node`, `research_post_node`, research subgraph import
- Remove: Research agent routing in `router_node` and `select_agent`
- Keep: Supervisor lifecycle, error handling, event streaming, handoff infrastructure

### Task Agent

Add `"research"` to `_DIRECT_SKILL_MODES`:

```python
_DIRECT_SKILL_MODES = {
    "image": {"skill_id": "image_generation", "param_key": "prompt"},
    "app": {"skill_id": "app_builder", "param_key": "description"},
    "slide": {"skill_id": "slide_generation", "param_key": "topic"},
    "data": {"skill_id": "data_analysis", "param_key": "query"},
    "research": {"skill_id": "deep_research", "param_key": "query"},  # NEW
}
```

The task agent synthesizes an `invoke_skill` call for `deep_research` with query, scenario, and depth params — same pattern as data_analysis.

### API (`query.py`)

Research mode no longer uses the worker queue. It flows through the supervisor like other modes:

```
POST /api/v1/query/stream (mode=RESEARCH)
  → invoke supervisor with task agent
  → task agent auto-invokes deep_research skill
  → stream events from skill executor
```

### Frontend

**No changes needed.** Both entry points work:

1. **Chat inline** — Task agent invokes deep_research via LLM reasoning. Events stream through existing chat SSE flow.
2. **Dedicated /task page** — `mode: "research"` triggers direct skill invocation. Same SSE events, same UI components.

The skill emits the same event types (stage, source, tool_call, tool_result, token, error) that the current research agent emits, so the frontend event handlers work unchanged.

## Files Changed

### Created

| File | Purpose |
|------|---------|
| `backend/app/agents/skills/builtin/deep_research_skill.py` | New DeepResearchSkill |
| `backend/app/agents/skills/builtin/research_prompts.py` | Extracted scenario/report prompts |

### Modified

| File | Change |
|------|--------|
| `backend/app/agents/skills/builtin/__init__.py` | Register DeepResearchSkill |
| `backend/app/agents/supervisor.py` | Remove research routing, keep supervisor shell |
| `backend/app/agents/subagents/task.py` | Add `"research"` to `_DIRECT_SKILL_MODES` |
| `backend/app/agents/state.py` | Keep ResearchState (backward compat), add deprecation note |
| `backend/app/api/query.py` | Remove worker queue path for research; route through supervisor |
| `backend/app/models/schemas.py` | Ensure RESEARCH mode maps correctly |

### Deleted

| File | Reason |
|------|--------|
| `backend/app/agents/subagents/research.py` | Replaced by DeepResearchSkill |

## Event Flow

```
User query (mode=research)
  → API creates supervisor input
  → Supervisor routes to task_node
  → Task agent synthesizes invoke_skill("deep_research", {...})
  → SkillExecutor runs DeepResearchSkill graph
    → init_config: emit stage("config", "Initializing research")
    → react_loop: LLM decides actions
      → execute_tools: emit tool_call/tool_result/source events
      → loop back to react_loop
      → ... until finish_research called
    → write_report: emit stage("write"), stream tokens
  → Skill returns output with report + sources
  → Task agent returns skill output as response
  → SSE stream delivers events to frontend
```

## Risk & Mitigation

| Risk | Mitigation |
|------|-----------|
| ReAct loop runs too long | Depth-based max_iterations (5 for fast, 20 for deep) + overall timeout (600s) |
| LLM doesn't call finish_research | Force finish after max_iterations; fallback to writing report from accumulated sources |
| Context window fills up | Token budget truncation via existing `truncate_messages_to_budget` |
| Quality regression vs current pipeline | Current prompts carried over; scenario-specific system prompts preserved |
| Consecutive tool errors | Circuit breaker (3 consecutive errors → force finish) |
