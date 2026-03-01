# Tools and Skills Relationship

## Overview

HyperAgent has two complementary capability layers:

- **Tools** are atomic LangChain `BaseTool` instances that agents bind to their LLM and invoke via tool-calling. Each tool does one thing (search, execute code, navigate browser).
- **Skills** are multi-step LangGraph subgraphs that orchestrate complex workflows. Agents invoke skills through the `invoke_skill` tool, which bridges the two systems.

```
Agent (Task / Research)
  │
  │  LLM selects tool via tool-calling
  ▼
Tool Registry (TOOL_CATALOG)
  ├── Direct tools: web_search, execute_code, browser_*, generate_image, ...
  ├── CodeAct: execute_script (opt-in, with hyperagent helper library)
  ├── App builder tools: create_app_project, app_write_file, ...
  ├── Skill bridge: invoke_skill, list_skills
  ├── HITL: ask_user
  └── Handoff: handoff_to_research (dynamic, with artifact transfer)
        │
        │  invoke_skill("data_analysis", {...})
        ▼
Skill Executor → SkillRegistry → Skill.create_graph()
  │
  │  Skill nodes execute internally
  ▼
Backend Services: llm_service, search_service, sandbox_manager, ...
```

---

## A. Tool System

### Tool Catalog

All tools are registered in `backend/app/agents/tools/registry.py`:

| Category | Tools | Module |
|----------|-------|--------|
| **SEARCH** | `web_search` | `tools/web_search.py` |
| **IMAGE** | `generate_image`, `analyze_image` | `tools/image_generation.py`, `tools/vision.py` |
| **SLIDES** | `generate_slides` | `tools/slide_generation.py` |
| **BROWSER** | `browser_navigate`, `browser_screenshot`, `browser_click`, `browser_type`, `browser_press_key`, `browser_scroll`, `browser_get_stream_url` | `tools/browser_use.py` |
| **CODE_EXEC** | `execute_code` | `tools/code_execution.py` |
| **DATA** | `sandbox_file` | `sandbox/__init__.py` |
| **APP_BUILDER** | `create_app_project`, `app_start_server`, `app_run_command`, `app_install_packages`, `app_get_preview_url` | `tools/app_builder.py` |
| **SKILL** | `invoke_skill`, `list_skills` | `tools/skill_invocation.py` |
| **HITL** | `ask_user` | `tools/hitl_tool.py` |
| **CODEACT** | `execute_script` (opt-in via `execution_mode: "codeact"`) | `tools/codeact.py` |
| **HANDOFF** | `handoff_to_research` (dynamic per agent) | `tools/handoff.py` |

### Agent Tool Access

Each agent type gets a subset of categories via `AGENT_TOOL_MAPPING`:

| Agent | Categories |
|-------|-----------|
| **Task** | SEARCH, IMAGE, SLIDES, BROWSER, CODE_EXEC, CODEACT*, APP_BUILDER, SKILL, HANDOFF, HITL |
| **Research** | SEARCH, IMAGE, BROWSER, SKILL, HANDOFF, HITL |

*CODEACT tools are only included when `execution_mode` is set to `"codeact"` in configuration.

Key: Research agent has no CODE_EXEC, DATA, APP_BUILDER, or SLIDES access.

### Tool Execution Pipeline

All tool calls go through the unified pipeline in `tools/tool_pipeline.py`:

```
1. before_execution()     ← HITL approval, guardrail scan, ask_user handling
2. inject_tool_context()  ← Add user_id/task_id to tool args
3. on_tool_call()         ← Emit tool_call event to frontend
4. tool.ainvoke(args)     ← Actual tool execution (with retry on transient failure)
5. after_execution()      ← Extract events (images, skill output, app builder)
6. truncate_tool_result() ← Enforce max content length per tool
7. on_tool_result()       ← Emit tool_result event to frontend
```

The pipeline has three hook implementations:

| Hook Class | Used By | Capabilities |
|-----------|---------|-------------|
| `TaskToolHooks` | Task agent | Full HITL (tool + skill approval, ask_user), guardrails, event extraction |
| `ResearchToolHooks` | Research agent | HITL (tool + skill approval), guardrails, source parsing |
| `CanonicalToolHooks` | `execute_react_loop` | Legacy callback wrapping (on_tool_call/result/token) |

**`skip_before_execution` flag**: When a tool has already been HITL-approved, the approved execution re-enters the pipeline with `skip_before_execution=True`, bypassing HITL/guardrails but still getting context injection, event extraction, and truncation.

### Batch vs Single Execution

- **`execute_tools_batch()`**: Partitions tool calls into parallel (most tools, semaphore-limited to 5), sequential (browser tools — side effects), and HITL-requiring. Returns `(messages, events, error_count, pending_interrupt)`.
- **`execute_tool()`**: Single tool through the full pipeline. Returns `ToolExecutionResult(message, events, pending_interrupt)`.

### Context Injection

`tools/context_injection.py` injects session identifiers into tool args:

| Injection Type | Tools |
|---------------|-------|
| **user_id + task_id** | All browser tools, all app builder tools, `invoke_skill`, `execute_code`, `execute_script`, `sandbox_file` |
| **user_id only** | `generate_image`, `generate_slides` |

### Tool Soft-Disable

For KV-cache optimization, tools can be "soft-disabled" instead of removed from the tool schema:

```python
from app.agents.tools.registry import soft_disable_tool, get_soft_disabled_message

soft_disable_tool("execute_code")  # Tool schema remains, but system message notes it's unavailable
msg = get_soft_disabled_message()  # "The following tools are currently unavailable: execute_code"
```

This preserves the stable tool schema prefix across LLM iterations, improving cache hit rates.

---

## B. Skill System

### Skill Type Hierarchy

```python
Skill (abstract base)                    # Graph-based skills
├── create_graph() → StateGraph          # Must implement
├── validate_input(params) → (bool, str) # Input validation
└── metadata: SkillMetadata              # Skill metadata

ToolSkill(Skill)                         # Single-step skills
├── execute(params, context) → dict      # Must implement
└── create_graph()                       # Auto-generates 1-node graph wrapping execute()
```

- `Skill`: For multi-step workflows (web_research, app_builder, data_analysis, slide_generation)
- `ToolSkill`: For single-step operations (image_generation, code_generation, task_planning)

### SkillMetadata Fields

```python
class SkillMetadata(BaseModel):
    id: str                          # "web_research", "data_analysis"
    name: str                        # "Web Research", "Data Analysis"
    version: str = "1.0.0"
    description: str
    category: str                    # "research", "data", "creative", "automation", "code"
    parameters: list[SkillParameter] # Input parameter definitions
    output_schema: dict              # JSON schema for output
    required_tools: list[str]        # Tools this skill binds to its LLM (validated at invocation)
    risk_level: str | None           # "low", "medium", "high" — for HITL governance
    max_execution_time_seconds: int  # Timeout (default 300s)
    max_iterations: int              # Max graph iterations (default 10)
    author: str                      # "hyperagent"
    tags: list[str]                  # Searchable tags
    enabled: bool                    # Whether skill is active
```

### All Builtin Skills

| Skill ID | Type | Category | Graph Nodes | required_tools | Runtime Dependencies |
|----------|------|----------|-------------|----------------|---------------------|
| `web_research` | Skill | research | search → summarize | `[]` | `search_service`, `llm_service` |
| `data_analysis` | Skill | data | plan → code_loop → summarize | `["execute_code", "sandbox_file", "web_search", "generate_image", "analyze_image"]` | `llm_service`, `execute_react_loop` with 5 bound tools |
| `app_builder` | Skill | automation | plan → scaffold → generate → start → finalize | `[]` | `llm_service`, `app_sandbox_manager` |
| `slide_generation` | Skill | creative | research → outline → images → generate | `[]` | `search_service`, `llm_service`, `image_generation_service`, `pptx_generation_service` |
| `image_generation` | ToolSkill | creative | (single execute) | `[]` | `image_generation_service`, `file_storage_service` |
| `code_generation` | ToolSkill | code | (single execute) | `[]` | `llm_service` |
| `task_planning` | ToolSkill | automation | (single execute) | `[]` | `llm_service` |

### How Skills Use Tools vs Services

Most skills call backend services directly and declare `required_tools=[]`. There is one exception:

**`data_analysis` — the only skill that binds tools to an LLM:**

```python
# In _get_data_tools():
from app.agents.tools.code_execution import execute_code
from app.agents.tools.image_generation import generate_image
from app.agents.tools.vision import analyze_image
from app.agents.tools.web_search import web_search
from app.sandbox import sandbox_file

return [execute_code, sandbox_file, web_search, generate_image, analyze_image]

# In code_loop node:
llm_with_tools = llm.bind_tools(all_tools)
result = await execute_react_loop(llm_with_tools=llm_with_tools, tools=all_tools, ...)
```

`data_analysis` runs a full ReAct loop internally, where the LLM can call `execute_code`, `sandbox_file`, `web_search`, `generate_image`, and `analyze_image`. Its `required_tools` accurately lists all 5 tools.

All other skills call services directly:

| Skill | What it calls | Instead of tool |
|-------|--------------|-----------------|
| `web_research` | `search_service.search_raw()` | `web_search` |
| `app_builder` | `app_sandbox_manager.scaffold_project()`, `.write_file()`, `.install_dependencies()`, `.start_dev_server()` | `create_app_project`, `app_write_file`, etc. |
| `slide_generation` | `search_service.search_raw()`, `image_generation_service.generate_image()`, `pptx_generation_service.generate_pptx()` | `web_search`, `generate_image`, `generate_slides` |
| `image_generation` | `image_generation_service.generate_image()` | `generate_image` |

### required_tools Semantics

`required_tools` is a **runtime contract** — it lists exactly the tools a skill binds to its LLM via `.bind_tools()`. At invocation time, `invoke_skill` validates all declared tools exist in the registry:

```python
# In skill_invocation.py
if skill.metadata.required_tools:
    from app.agents.tools.registry import get_all_tools
    known = {t.name for t in get_all_tools()}
    missing = [t for t in skill.metadata.required_tools if t not in known]
    if missing:
        return error("Skill requires unavailable tools: {missing}")
```

Skills that call services directly (not tools) correctly declare `required_tools=[]`.

### Skill Registration & Loading

**Builtin skills**: Auto-discovered at startup in `SkillRegistry._register_builtin_skills()`. Stored in-memory only. Synced to DB via `sync_builtin_skills()` which updates: `name`, `version`, `description`, `category`, `author`, `metadata_json`.

**Dynamic skills**: Loaded from `skill_definitions` DB table. Source code validated by `skill_code_validator`, hash-verified, then executed in a restricted namespace with safe builtins and whitelisted imports.

**Skill composition**: Skills can invoke other skills via `SkillContext.invoke_skill()`, with a max depth limit (default 3) to prevent infinite recursion.

---

## C. Event Flow

Events flow from skills to the frontend through multiple paths. Understanding these paths is critical for avoiding duplication.

### Path 1: Real-time dispatch (primary for stage/terminal events)

```
Skill node emits event
  → skill_executor yields event via astream
  → invoke_skill iterates generator
  → dispatch_custom_event("skill_event", data=event)
  → stream_processor._handle_custom_event() yields to frontend
```

This is the **primary path** for `stage`, `terminal_command`, `terminal_output`, `terminal_error`, `terminal_complete`, `browser_stream`, and `workspace_update` events. They arrive at the frontend in real-time while the skill is still executing.

### Path 2: JSON extraction at tool_end (for skill_output)

```
invoke_skill completes → returns JSON with {skill_id, output, success}
  → stream_processor._handle_tool_end() parses JSON
  → Extracts and emits skill_output event (for preview URLs, download URLs)
```

This is the **only path** for `skill_output` events from `invoke_skill`. The skill_executor also yields a `skill_output` event, but `invoke_skill` captures it into the `output` variable rather than dispatching it — so it appears in the JSON return, not as a custom event.

### Path 3: State forwarding at chain_end (for subagent events)

```
Agent node completes → returns {events: [...]} in state
  → stream_processor._handle_chain_end() forwards events from output.events
  → Deduplication via _should_emit_subevent()
```

This catches events that were collected by `after_execution()` hooks (e.g., image events from `generate_image`, browser_stream from app_builder tools).

### Deduplication

The `StreamProcessor` maintains per-session dedup sets:

| Event Type | Dedup Key | Set |
|-----------|----------|-----|
| `stage` | `{name}:{status}` | `emitted_stage_keys` |
| `tool_call` | tool_call_id | `emitted_tool_call_ids` |
| `tool_result` | `result:{tool_call_id}` | `emitted_tool_call_ids` |
| `image` | index | `emitted_image_indices` |
| `skill_output` | `{skill_id}:{download_url\|preview_url}` | `emitted_skill_output_keys` |
| `terminal_*` | content-based key (first 200 chars) | `emitted_terminal_keys` |
| `workspace_update` | `ws:{operation}:{path}` | `emitted_terminal_keys` |
| `browser_stream` | `bs:{stream_url}` | `emitted_terminal_keys` |
| `interrupt` | interrupt_id | `emitted_interrupt_ids` |

### What was removed (Fix 2)

Previously, `invoke_skill` collected events in a `collected_events` list AND dispatched them in real-time via `dispatch_custom_event`. The JSON return included `"events": collected_events`, and both `stream_processor._handle_tool_end` and `event_extraction.extract_skill_events` extracted events from this JSON — creating dual emission. Now:

- `invoke_skill` no longer collects or returns events in JSON
- `stream_processor._handle_tool_end` no longer extracts `"events"` from invoke_skill JSON (only extracts `skill_output`)
- `event_extraction.extract_skill_events` no longer extracts `parsed.get("events")` (only handles image and app_builder browser_stream)
- Real-time dispatch via `dispatch_custom_event` is the single source for stage/terminal events

---

## D. HITL Integration

### Tool Risk Classification

`hitl/tool_risk.py` classifies tools into risk levels:

| Risk Level | Tools |
|-----------|-------|
| **HIGH** | browser_*, execute_code, sandbox_file, shell_command |
| **MEDIUM** | api_call, http_request, database_write |

### Skill Risk Determination

Skills are assessed via `get_skill_risk_level()`:
1. If `risk_level` is explicitly set in metadata, use it
2. Otherwise, infer from `required_tools` — if any tool is HIGH risk, the skill is HIGH risk

### Approval Flow

When `before_execution()` determines a tool needs approval:

```
1. Create approval interrupt → publish to Redis
2. Return ToolExecutionResult(message=None, pending_interrupt={...})
3. Agent stores pending_interrupt in state → graph pauses
4. Frontend shows approval dialog → user responds
5. wait_interrupt_node receives response
6. If approved: re-execute tool via execute_tool() with skip_before_execution=True
7. Tool goes through steps 2-7 of pipeline (context injection, execution, event extraction, truncation)
```

Both `TaskToolHooks` and `ResearchToolHooks` implement:
- Tool-level HITL approval (for high-risk direct tools)
- Skill-level HITL approval (for `invoke_skill` with high-risk skills)
- Guardrail scanning (tool_scanner)

`TaskToolHooks` additionally handles:
- `ask_user` tool (DECISION and INPUT interrupt types)

---

## E. Data Model

### SkillDefinition (DB)

| Column | Source | Updated by sync_builtin_skills? |
|--------|--------|-------------------------------|
| `id` | SkillMetadata.id | N/A (PK) |
| `name` | SkillMetadata.name | Yes |
| `version` | SkillMetadata.version | Yes |
| `description` | SkillMetadata.description | Yes |
| `category` | SkillMetadata.category | Yes |
| `author` | SkillMetadata.author | Yes |
| `metadata_json` | Full SkillMetadata JSON | Yes |
| `module_path` | Empty for builtins | No |
| `source_code` | For dynamic skills only | No |
| `source_code_hash` | For dynamic skills only | No |
| `enabled` | Default true | No (also duplicated in metadata_json) |
| `is_builtin` | true/false | No |
| `invocation_count` | Runtime counter | No |
| `last_invoked_at` | Runtime timestamp | No |

### SkillDefinition.to_dict()

Returns top-level columns plus fields parsed from `metadata_json`:
- `parameters`, `required_tools`, `risk_level`, `tags`, `output_schema`

---

## F. Key Architectural Decisions

### Why skills call services, not tools

1. **No circular dependencies**: Tools can invoke skills (via `invoke_skill`), so skills invoking tools would create cycles
2. **Performance**: Direct service calls avoid tool invocation overhead (context injection, HITL checks, event extraction)
3. **State isolation**: Skills manage their own internal state; the tool pipeline manages agent-level state

### Why data_analysis is the exception

`data_analysis` needs an LLM-driven iterative code execution loop — the LLM must decide what code to run, inspect results, and iterate. This requires `execute_react_loop` with bound tools, which is fundamentally different from the service-call pattern other skills use.

### Why required_tools validation matters

Even though most skills call services directly, `data_analysis` genuinely depends on tools being registered. The validation in `invoke_skill` prevents silent failures where a skill's ReAct loop would have a degraded or empty tool set.
