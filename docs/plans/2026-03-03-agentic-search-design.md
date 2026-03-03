# Agentic Search System Design

**Date:** 2026-03-03
**Status:** Approved
**Replaces:** `web_research_skill.py`, `deep_research_skill.py`

## 1. Goals

Replace the current web search skills (web_research + deep_research) with a single **Agentic Search** skill that:

- **Decomposes** complex queries into targeted sub-queries
- **Searches in parallel** across multiple providers
- **Evaluates** result coverage and identifies gaps
- **Refines** searches iteratively until confidence is sufficient
- **Returns structured findings** (not a report) for flexible downstream use
- **Fast-paths** trivial queries through single-shot search (tiered approach)

## 2. Architecture Overview

### Graph Structure (LangGraph State Machine)

```
                          ┌──────────────┐
                          │   classify   │  LITE model
                          └──────┬───────┘
                                 │
                    ┌────────────┴────────────┐
                    │ trivial                  │ complex
                    ▼                          ▼
            ┌──────────────┐         ┌──────────────┐
            │ quick_search │         │  plan_search  │  PRO model
            └──────┬───────┘         └──────┬───────┘
                   │                        │
                   ▼                        ▼
            ┌──────────────┐         ┌──────────────┐
            │  synthesize  │         │execute_search │  parallel, no LLM
            └──────┬───────┘         └──────┬───────┘
                   │                        │
                   ▼                        ▼
                  END                ┌──────────────┐
                                    │   evaluate    │  PRO model
                                    └──────┬───────┘
                                           │
                                  ┌────────┴────────┐
                                  │ sufficient       │ gaps
                                  ▼                  ▼
                           ┌──────────────┐   ┌──────────────┐
                           │  synthesize  │   │    refine     │  PRO model
                           └──────┬───────┘   └──────┬───────┘
                                  │                  │
                                  ▼                  ▼
                                 END          execute_search
                                              (loop, max 3 rounds)
```

### Node Responsibilities

| Node | Model Tier | Purpose |
|------|-----------|---------|
| `classify` | LITE | Classify complexity: `trivial` or `complex`. Extract intent signals (factual, comparative, temporal, exploratory, statistical). |
| `quick_search` | None | Single search via best-fit provider. No LLM needed. |
| `plan_search` | PRO | Decompose into 2-6 sub-queries with type, provider preference, and priority. |
| `execute_search` | None | Fan out sub-queries via `asyncio.gather` to search providers. Populate knowledge state. |
| `evaluate` | PRO | Score coverage per sub-query (0-1). Identify gaps, contradictions. Decide: sufficient → synthesize, or gaps → refine. |
| `refine` | PRO | Generate follow-up queries for unanswered/low-coverage sub-queries. Update plan. |
| `synthesize` | PRO | Build structured output: facts with citations, confidence, unanswered questions. |

### LLM Call Budget

- **Trivial query:** 1 LLM call (classify) + tool execution. Could be 0 if classifier is confident.
- **Complex query:** 3-5 LLM calls (classify + plan + evaluate + optional refine + synthesize).
- **Max refinement rounds:** 3.

## 3. State Model

### Core Knowledge Accumulator

```python
class SearchKnowledge(TypedDict):
    """Structured knowledge accumulated during agentic search."""

    # Sub-query tracking
    sub_queries: list[SubQuery]

    # Accumulated facts
    facts: list[ExtractedFact]

    # Source registry (deduplicated by URL)
    sources: dict[str, SourceEntry]

    # Contradiction tracking
    contradictions: list[Contradiction]

    # Coverage metrics
    coverage_scores: dict[str, float]   # sub_query_id -> coverage (0.0-1.0)
    overall_confidence: float            # weighted average
    refinement_round: int                # 0-3


class SubQuery(TypedDict):
    id: str
    query: str
    type: str          # factual | comparative | temporal | statistical | exploratory
    provider: str      # preferred search provider
    priority: int      # 1-3 (3 = highest)
    status: str        # pending | searched | sufficient | gap


class ExtractedFact(TypedDict):
    claim: str
    source_urls: list[str]
    sub_query_id: str
    confidence: float   # 0.0-1.0


class SourceEntry(TypedDict):
    title: str
    url: str
    snippet: str
    content: str | None
    domain: str
    credibility: float  # domain-based 0.0-1.0
    used_for: list[str] # sub_query_ids


class Contradiction(TypedDict):
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    sub_query_id: str
```

### Full Skill State

```python
class AgenticSearchState(SkillState, total=False):
    # Input
    query: str
    complexity: str                     # trivial | complex
    intent_signals: list[str]

    # Knowledge accumulator
    knowledge: SearchKnowledge

    # Search plan
    search_plan: list[SubQuery]

    # LLM conversation for evaluation/refinement
    lc_messages: Annotated[list[BaseMessage], operator.add]

    # Loop control
    max_refinements: int                # default 3
    current_refinement: int

    # Output
    output: dict

    # Events + context passthrough
    pending_events: Annotated[list[dict], operator.add]
    locale: str
    provider: str | None
    model: str | None
    tier: Any | None
```

## 4. Multi-Provider Search Architecture

### Provider Interface

```python
class SearchProvider(ABC):
    """Abstract search provider."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def supported_types(self) -> list[str]: ...

    @abstractmethod
    async def search(
        self, query: str, max_results: int = 5, **kwargs
    ) -> list[SearchResult]: ...


class SearchProviderRegistry:
    """Registry of available search providers."""

    def register(self, provider: SearchProvider) -> None: ...

    def get_provider_for_type(self, query_type: str) -> SearchProvider:
        """Select best provider for query type. Falls back to Tavily."""
        ...

    async def search(
        self, query: str, query_type: str, **kwargs
    ) -> list[SearchResult]:
        """Route to appropriate provider."""
        ...
```

### Providers

| Provider | API Key Env | Query Types | Notes |
|----------|------------|-------------|-------|
| **Tavily** | `TAVILY_API_KEY` | factual, comparative, exploratory, technical | Default/fallback for all types |
| **Serper** | `SERPER_API_KEY` | news, temporal, recent | Google Search via Serper API |
| **Jina Reader** | `JINA_API_KEY` | deep_read | Page content extraction, not search. Used for follow-up deep reading. |

Providers are auto-registered on startup based on available API keys. If only Tavily is configured, it handles all query types (current behavior).

### Caching

Existing `SearchCache` (5-min TTL, 1000 entries) is preserved and shared across all providers. Cache key includes provider name.

## 5. Output Schema

```python
class AgenticSearchOutput(TypedDict):
    """Structured output from agentic search."""

    # Core findings
    facts: list[dict]               # [{claim, sources, confidence, sub_query}]
    sources: list[dict]             # [{title, url, snippet, credibility}]

    # Quality metrics
    overall_confidence: float       # 0.0-1.0
    coverage: dict[str, float]      # sub_query_id -> coverage score
    unanswered: list[str]           # sub-queries not sufficiently answered
    contradictions: list[dict]      # conflicting claims across sources

    # Metadata
    complexity: str                 # trivial | complex
    sub_queries_count: int
    sources_count: int
    refinement_rounds: int
    provider_usage: dict[str, int]  # provider_name -> num queries
```

### Task Agent Integration

The task agent invokes `agentic_search` via `invoke_skill` like current `deep_research`. Key difference:

- **Current:** `deep_research` returns `{report, sources, findings}` — agent forwards the report.
- **New:** `agentic_search` returns structured facts — agent composes its own response, cites sources, or feeds into other tools.

The `web_search` tool remains available to the task agent for simple ad-hoc searches that don't need the agentic pipeline.

## 6. Event Streaming

### New Event Types

| Event | Payload | When |
|-------|---------|------|
| `search_plan` | `{sub_queries: [{id, query, type, provider}]}` | After plan_search |
| `sub_query_status` | `{id, status, coverage?}` | Per sub-query state change |
| `knowledge_update` | `{facts_count, sources_count}` | After execute_search |
| `confidence_update` | `{confidence, coverage_summary}` | After evaluate |
| `refinement_start` | `{round, follow_up_queries}` | When entering refinement |

### Event Flow (Complex Query)

1. `stage: "classifying"` → `stage: "planning"`
2. `search_plan: {sub_queries: [...]}`
3. `sub_query_status: {id: "sq1", status: "searching"}` (×N parallel)
4. `source` events as results arrive (reuse existing)
5. `sub_query_status: {id: "sq1", status: "done", coverage: 0.8}`
6. `knowledge_update: {facts_count: 12, sources_count: 8}`
7. `stage: "evaluating"` → `confidence_update: {confidence: 0.6}`
8. `refinement_start: {round: 1, follow_up_queries: [...]}`
9. ... repeat search + evaluate ...
10. `confidence_update: {confidence: 0.85}` → `stage: "synthesizing"`
11. `skill_output` with final structured results

### Event Flow (Trivial Query)

1. `stage: "classifying"` → `stage: "searching"`
2. `source` events
3. `stage: "synthesizing"` → `skill_output`

## 7. File Structure

```
backend/app/
├── agents/skills/builtin/
│   ├── agentic_search_skill.py      # NEW — main skill
│   ├── deep_research_skill.py       # REMOVE (after migration)
│   └── web_research_skill.py        # REMOVE (after migration)
├── services/
│   ├── search.py                    # REFACTOR — thin wrapper + cache
│   └── search_providers/            # NEW
│       ├── __init__.py              # Registry init + auto-registration
│       ├── base.py                  # SearchProvider ABC, SearchProviderRegistry
│       ├── tavily_provider.py       # Extract from current search.py
│       ├── serper_provider.py       # NEW — Google via Serper
│       └── jina_provider.py         # NEW — Jina Reader
├── agents/tools/
│   └── web_search.py               # KEPT — ad-hoc single searches
```

## 8. Configuration

New environment variables:

```env
# Search providers (Tavily already exists)
SERPER_API_KEY=           # Optional — enables Google search via Serper
JINA_API_KEY=             # Optional — enables Jina Reader for deep content

# Agentic search tuning
AGENTIC_SEARCH_MAX_REFINEMENTS=3        # Max refinement rounds
AGENTIC_SEARCH_CONFIDENCE_THRESHOLD=0.7 # Target confidence to stop
AGENTIC_SEARCH_MAX_SUB_QUERIES=6        # Max sub-queries per plan
```

## 9. Migration Plan

1. Build `agentic_search_skill.py` and `search_providers/` as new files.
2. Register `agentic_search` skill in skill registry.
3. Update `_DIRECT_SKILL_MODES` in task agent: `"research"` → `"agentic_search"`.
4. Keep `deep_research` and `web_research` as deprecated aliases that redirect to `agentic_search`.
5. After validation, remove old skill files and aliases.

## 10. Domain Credibility Scoring

Simple heuristic-based scoring for source credibility:

| Domain Pattern | Score | Rationale |
|----------------|-------|-----------|
| `.gov`, `.edu` | 0.9 | Institutional authority |
| Major news (nytimes, bbc, reuters, etc.) | 0.85 | Editorial standards |
| Wikipedia | 0.7 | Community-edited, generally reliable |
| Tech docs (docs.*, developer.*) | 0.8 | Official documentation |
| Stack Overflow, GitHub | 0.7 | Community-vetted |
| Unknown domains | 0.5 | Neutral default |
| Known low-quality domains | 0.2 | Configurable blocklist |

Scores are used by the evaluator to weight source reliability when assessing coverage.
