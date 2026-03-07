# Context Management

This document describes HyperAgent's context management system — how the agent constructs, compresses, caches, and enriches the LLM context window across multi-turn conversations and long-running task loops.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prompt Construction](#prompt-construction)
- [Context Compression](#context-compression)
- [KV-Cache Optimization](#kv-cache-optimization)
- [Context Engineering: Attention Manipulation](#context-engineering-attention-manipulation)
- [Persistent Memory](#persistent-memory)
- [Scratchpad (Context Offloading)](#scratchpad-context-offloading)
- [Tool Context Injection](#tool-context-injection)
- [Configuration Reference](#configuration-reference)
- [Key Files](#key-files)

---

## Overview

HyperAgent's context management operates across six layers, each addressing a different aspect of the context window:

| Layer | Purpose | Trigger |
|-------|---------|---------|
| **Prompt Construction** | Build the initial message list with stable prefix | Every request |
| **Context Compression** | Summarize old messages to free token budget | Token count > 60k |
| **KV-Cache Optimization** | Maximize LLM cache reuse via stable prefixes | Every reasoning step |
| **Attention Manipulation** | Inject todo/plan into recent window | Every ReAct iteration |
| **Persistent Memory** | Cross-session user knowledge | Per-user, if enabled |
| **Scratchpad** | Short-term context offloading | Agent-initiated |

---

## Architecture

```
                        User Query
                            |
                            v
                    +-----------------+
                    |   Supervisor    |
                    +--------+--------+
                             |
                             v
                +--------------------------+
                |  _build_initial_messages  |
                |  (Prompt Construction)    |
                +-----------+--------------+
                            |
          +-----------------+-----------------+
          |                 |                 |
          v                 v                 v
   [System Prompt]   [User Memories]   [Scratchpad]
   (cache_control)   (if enabled)      (if enabled)
          |                 |                 |
          +-----------------+-----------------+
                            |
                            v
                +------------------------+
                |  apply_context_policy  |
                |  (Compression Policy)  |
                +----------+-------------+
                           |
              +------------+------------+
              |                         |
              v                         v
     [LLM Summarization]      [Hard Truncation]
     (semantic, 60k trigger)   (budget fallback)
              |                         |
              +------------+------------+
                           |
                           v
                +------------------------+
                | Context Engineering    |
                | (Todo/Plan Injection)  |
                +-----------+------------+
                            |
                            v
                +------------------------+
                |  KV-Cache Optimization |
                |  (prefix_hash track)   |
                +-----------+------------+
                            |
                            v
                   [LLM Invocation]
```

---

## Prompt Construction

**File:** `backend/app/agents/subagents/task.py` — `_build_initial_messages()`

Every request begins by assembling a message list. The order matters for KV-cache efficiency — system messages form a **stable prefix** that the LLM can cache across iterations.

### Message Order

```
[0] SystemMessage: Task agent system prompt       <- immutable, cache_control: ephemeral
[1] SystemMessage: User memories                  <- if memory_enabled
[2] SystemMessage: Scratchpad context             <- if context_offloading_enabled
[3] SystemMessage: [Previous conversation summary]<- if context was compressed
[4] HumanMessage/AIMessage/ToolMessage: history   <- conversation turns
[5] SystemMessage: Plan step guidance             <- if executing a plan
[6] SystemMessage: Soft-disabled tools notice     <- if any tools are disabled
```

### System Prompt Caching

The core system prompt uses Anthropic's `cache_control` header to enable prompt caching:

```python
SystemMessage(
    content=system_prompt,
    additional_kwargs={"cache_control": {"type": "ephemeral"}},
)
```

For English locale, a pre-built `TASK_SYSTEM_MESSAGE` singleton is reused. Non-English locales append a language instruction block *after* the immutable core to avoid invalidating the cached prefix.

**File:** `backend/app/agents/prompts.py`

---

## Context Compression

Context compression prevents token overflow in long conversations by summarizing older messages while preserving semantic meaning and recoverable references.

### Two-Tier Strategy

```
Tier 1: LLM-Based Summarization (Primary)
  - Triggered at 60k tokens
  - Uses LITE-tier LLM for fast, cheap summarization
  - Preserves intent, decisions, key facts
  - Extracts recoverable references (files, URLs, tools, commands)
  - Keeps 10 most recent messages intact

Tier 2: Hard Truncation (Fallback)
  - Applied after compression or if compression fails
  - Hard budget: 100k tokens
  - Keeps system messages + last 4 recent messages
  - Respects tool call/result pairing
```

### Compression Pipeline

**Files:**
- `backend/app/agents/context_compression.py` — Core compression engine
- `backend/app/agents/context_policy.py` — Orchestration layer

```
reason_node()
    |
    v
apply_context_policy()
    |
    +-- 1. enforce_summary_singleton()
    |       Remove duplicate summary messages, keep only the latest
    |
    +-- 2. Compression Pass (if tokens > threshold)
    |       |
    |       +-- Estimate token count (tiktoken or heuristic: len/4)
    |       +-- Separate system messages (always preserved)
    |       +-- Split remaining into old (to summarize) + recent (to keep)
    |       +-- Snap to tool-pair boundary (never split AIMessage from ToolMessage)
    |       +-- Extract recoverable references via regex (pre-LLM, guaranteed)
    |       +-- Call LITE LLM to summarize old messages
    |       +-- Validate summary length (max 2000 tokens)
    |       +-- Inject summary as SystemMessage after main system prompt
    |
    +-- 3. Truncation Pass (hard budget)
            |
            +-- If still over 100k tokens, drop oldest non-system messages
            +-- Preserve last 4 messages + all system messages
            +-- Respect tool call/result pairing
```

### Recoverable Reference Extraction

Before LLM summarization, the system extracts references via regex patterns that are guaranteed to survive compression:

```python
refs = {
    "files":    set(),  # /path/to/file patterns
    "urls":     set(),  # http(s):// patterns
    "tools":    set(),  # Tool names from ToolMessage/AIMessage
    "commands": set(),  # Shell commands from shell_exec calls
}
```

These are appended to the summary regardless of what the LLM produces:

```
## Preserved References (automated)
- Files: /home/user/app.py, /tmp/output.csv
- URLs: https://example.com
- Tools: web_search, execute_code
- Commands: python script.py, npm install
```

This ensures the agent can always re-read files, re-visit URLs, and remember which tools were used, even after compression.

### Summary Injection

Summaries are injected as a `SystemMessage` with distinctive markers:

```
[Previous conversation summary]
<summary content>
[End of summary - recent messages follow]
```

The `enforce_summary_singleton` flag (default: true) ensures only one summary exists in the message list at any time. When a new summary is generated, older ones are removed.

### Tool-Pair Boundary Snapping

A critical invariant: tool calls and their results must never be separated. When splitting messages into "old" and "recent", the system snaps backward to avoid splitting:

- An `AIMessage` with `tool_calls` from its `ToolMessage` responses
- A `ToolMessage` from its parent `AIMessage`

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `context_compression_enabled` | `true` | Master toggle |
| `context_compression_token_threshold` | `60000` | Token count to trigger compression |
| `context_compression_preserve_recent` | `10` | Recent messages to always keep |
| `context_summary_singleton_enforced` | `true` | Keep only one summary in messages |

---

## KV-Cache Optimization

The system maximizes LLM KV-cache hit rates by maintaining a stable message prefix across reasoning iterations.

**File:** `backend/app/agents/context_compression.py` — `get_stable_prefix()`, `get_dynamic_suffix()`

### Stable Prefix vs Dynamic Suffix

Messages are conceptually split into two regions:

```
STABLE PREFIX (cached by LLM)          DYNAMIC SUFFIX (changes each iteration)
+----------------------------------+   +----------------------------------+
| [0] System: Agent instructions   |   | [N] Human: User query            |
| [1] System: User memories        |   | [N+1] AI: Reasoning + tool_calls |
| [2] System: Scratchpad context   |   | [N+2] Tool: Result               |
| [3] System: Conversation summary |   | [N+3] AI: More reasoning         |
+----------------------------------+   +----------------------------------+
  All leading SystemMessages             First non-SystemMessage onward
```

### Prefix Hash Tracking

**File:** `backend/app/agents/subagents/task.py` — `reason_node()` (lines 562-585)

Each iteration, the system computes an MD5 hash of the stable prefix content and compares it to the previous iteration's hash. If the prefix changed unexpectedly, a debug log is emitted — this signals potential cache invalidation.

```python
stable_prefix = get_stable_prefix(lc_messages)
prefix_content = "".join(str(m.content) for m in stable_prefix)
current_prefix_hash = hashlib.md5(prefix_content.encode(), usedforsecurity=False).hexdigest()

if prev_prefix_hash and prev_prefix_hash != current_prefix_hash:
    logger.debug("kv_cache_prefix_changed", ...)
```

The hash is stored in `TaskState.prefix_hash` and persisted across iterations.

### Alphabetical Tool Sorting

**File:** `backend/app/agents/subagents/task.py` — `_get_cached_task_tools()`

Tools are sorted alphabetically by name before binding to the LLM. This ensures consistent schema ordering across requests, preventing cache invalidation from tool reordering.

```python
_cached_task_tools = sorted(tools, key=lambda t: t.name)
```

Tool lists are cached with a `threading.Lock()` for thread safety and invalidated when tools are dynamically registered/unregistered (e.g., MCP connect/disconnect).

### Soft Tool Disabling

**File:** `backend/app/agents/tools/registry.py` — `soft_disable_tool()`, `get_soft_disabled_message()`

Instead of removing a tool's schema from the LLM (which changes the prefix and invalidates the cache), tools are **soft-disabled** by injecting a system message:

```
[Tool Availability Notice] The following tools are currently unavailable
and must NOT be called: tool_a, tool_b. Use alternative tools or approaches instead.
```

The tool schemas remain in the prefix, preserving cache stability.

---

## Context Engineering: Attention Manipulation

Long agent loops suffer from the "lost in the middle" problem — the global plan gets buried in old context and the LLM loses track. HyperAgent addresses this by re-injecting the execution plan into the model's recent attention window on every iteration.

**File:** `backend/app/agents/subagents/task.py` — `reason_node()` (lines 595-658)

### Planned Execution Injection

When the agent has an `execution_plan` (from the `task_planning` skill), a step guidance message is appended to the end of the message list on each iteration:

```
[Plan Execution — Step 3 of 7]
Progress:
  DONE: Step 1: Analyze requirements
  DONE: Step 2: Set up project structure
  >>> CURRENT: Step 3: Implement API endpoints
  pending: Step 4: Write tests
  pending: Step 5: Add error handling
  pending: Step 6: Create documentation
  pending: Step 7: Deploy

Current step: Implement API endpoints
Recommended tool: execute_code
Focus on completing this specific step.
```

### Todo.md Persistence

The execution plan is also written to `/home/user/.hyperagent/todo.md` inside the sandbox. This file persists across iterations and serves as a durable record of plan progress. After each successful tool execution, the todo is updated with the new step status.

The todo content (capped at 2000 characters) is injected as a `SystemMessage` on each `reason_node` call, ensuring the model always has the current plan state in its recent context.

---

## Persistent Memory

The persistent memory system stores cross-session user knowledge — preferences, facts, past experiences, and procedures — and injects them into the agent's context at the start of every request.

**Files:**
- `backend/app/services/memory_service.py` — Core store, extraction, formatting
- `docs/Agent-Memory-System.md` — Detailed documentation

### Memory Types

| Type | Purpose | Example |
|------|---------|---------|
| `preference` | User style, tool, language preferences | "Prefers Python with type hints" |
| `fact` | Facts about the user | "Senior engineer at Acme Corp" |
| `episodic` | Notable past interactions and outcomes | "Built dashboard with app_builder, completed in 45s" |
| `procedural` | Workflows and procedures | "For data analysis: CSV -> pandas -> chart" |

### Loading (Injection)

Memories are fetched from PostgreSQL (or in-memory fallback) and formatted as an XML block injected as a `SystemMessage`:

```xml
<user_memories>
Remembered from previous conversations:
<preferences>
<!-- Apply these preferences to tailor your responses (language, style, tools). -->
- Prefers Python with type hints
</preferences>
<facts>
<!-- Use these facts as context. Do not re-ask questions already answered here. -->
- Senior engineer at Acme Corp
</facts>
<past_experiences>
<!-- Reference relevant past experiences. Reuse successful approaches. -->
- Built dashboard using app_builder (tools: app_builder; outcome: completed; took 45s)
</past_experiences>
<procedures>
<!-- Follow these known procedures/tool sequences when the task matches. -->
- For data analysis: upload CSV -> run pandas script -> generate chart
</procedures>
Use these to personalize responses. Do not mention them unless asked.
</user_memories>
```

### Extraction (Saving)

After each conversation, the supervisor fires a background task that:

1. Takes the last 20 messages from the conversation
2. Sends them to a LITE-tier LLM with an extraction prompt
3. Extracts `{type, content}` memory objects
4. Runs safety checks (quarantines prompt injection attempts)
5. Deduplicates against existing memories (case-insensitive)
6. Persists to PostgreSQL

### Safety

Memories pass through two safety gates:

1. **Injection detection** — Patterns like "ignore instruction", "override safety", "jailbreak" cause the memory to be quarantined and never rendered into prompts.
2. **Render sanitization** — When formatting for prompts, lines starting with `important:`, `instruction:`, `system:`, or `you must` are stripped. Angle brackets are removed to prevent XML injection.

---

## Scratchpad (Context Offloading)

The scratchpad allows the agent to explicitly offload context — notes, intermediate results, working memory — outside the main message list, and retrieve it later.

**Files:**
- `backend/app/services/scratchpad_service.py` — Storage service
- `backend/app/agents/tools/scratchpad.py` — `write_scratchpad` and `read_scratchpad` tools

### Scopes

| Scope | Lifetime | Storage |
|-------|----------|---------|
| `session` | Current task/conversation | In-memory dict keyed by `(user_id, task_id, namespace)` |
| `persistent` | Cross-session | PostgreSQL `Memory` table (falls back to in-memory) |

### Prompt Injection

At the start of each request, if `context_offloading_enabled` is true, scratchpad notes are fetched and injected as a `SystemMessage`:

```python
scratchpad_context = await get_scratchpad_service().get_compact_context(
    user_id=user_id, task_id=task_id, max_chars=1200
)
if scratchpad_context:
    lc_messages.append(SystemMessage(content=scratchpad_context))
```

The injected block looks like:

```
[Scratchpad Context]
Session scratchpad:
<agent's notes>

Persistent scratchpad:
<agent's persistent notes>
```

Content is capped at 1200 characters. Individual notes are capped at 4000 characters on read and 12000 on write.

---

## Tool Context Injection

**File:** `backend/app/agents/tools/context_injection.py`

Tools that need session awareness (browser, app builder, code execution, file operations, scratchpad) have `user_id` and `task_id` injected into their arguments automatically. This is done immutably — a new dict is returned, never mutating the original args.

| Category | Injected Fields |
|----------|----------------|
| Session tools (browser, app builder, code exec, file ops, scratchpad, skills) | `user_id` + `task_id` |
| User-only tools (image generation, slide generation) | `user_id` only |

---

## Configuration Reference

### Environment Variables / Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `context_compression_enabled` | `true` | Enable LLM-based context compression |
| `context_compression_token_threshold` | `60000` | Token count to trigger compression |
| `context_compression_preserve_recent` | `10` | Recent messages preserved during compression |
| `context_summary_singleton_enforced` | `true` | Keep only one summary in message list |
| `context_offloading_enabled` | `false` | Enable scratchpad tools and prompt injection |
| `context_offloading_persistent_enabled` | `false` | Enable DB-backed persistent scratchpad |
| `memory_enabled` | `true` | Enable persistent user memory (per-request toggle) |

### Internal Constants

| Constant | Value | Location |
|----------|-------|----------|
| Truncation max tokens | `100000` | `react_tool.py` |
| Truncation preserve recent | `4` | `react_tool.py` |
| Max summary tokens | `2000` | `context_compression.py` |
| Min messages to compress | `5` | `context_compression.py` |
| Scratchpad max read chars | `4000` | `scratchpad_service.py` |
| Scratchpad max write chars | `12000` | `scratchpad_service.py` |
| Scratchpad compact context max | `1200` | `scratchpad_service.py` |
| Todo injection max chars | `2000` | `task.py` |
| Memory extraction messages | `20` | `memory_service.py` |

---

## Key Files

| File | Role |
|------|------|
| `backend/app/agents/subagents/task.py` | Message construction, compression integration, KV-cache tracking, attention manipulation |
| `backend/app/agents/context_compression.py` | LLM-based summarization, token estimation, prefix/suffix extraction, summary injection |
| `backend/app/agents/context_policy.py` | Orchestrates compression + truncation in sequence |
| `backend/app/agents/memory.py` | `ConversationMemory` class with windowing and `window_messages()` utility |
| `backend/app/agents/prompts.py` | System prompt definitions with cache control and locale support |
| `backend/app/agents/tools/registry.py` | Tool catalog, alphabetical sorting, soft-disable mechanism, cache invalidation |
| `backend/app/agents/tools/react_tool.py` | `truncate_messages_to_budget()`, tool message deduplication |
| `backend/app/agents/tools/context_injection.py` | `user_id`/`task_id` injection into tool args |
| `backend/app/agents/state.py` | `TaskState` with `context_summary`, `prefix_hash`, `active_todo`, `execution_plan` |
| `backend/app/agents/events.py` | Event schema for compression/truncation stage events |
| `backend/app/services/memory_service.py` | Persistent memory store, extraction, formatting, safety |
| `backend/app/services/scratchpad_service.py` | Scratchpad storage with session and persistent scopes |
| `backend/app/agents/tools/scratchpad.py` | `write_scratchpad` and `read_scratchpad` tool definitions |
