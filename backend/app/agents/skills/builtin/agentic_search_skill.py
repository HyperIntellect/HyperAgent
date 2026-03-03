"""Web Research Skill — unified tier-aware search with query decomposition and evaluation.

LITE tier: fast single search + summarize.
PRO tier: LLM classifier decides trivial vs complex.
MAX tier: full agentic pipeline with decomposition, parallel search, and evaluation.
"""

import asyncio
import json
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.agents import events as agent_events
from app.agents.skills.skill_base import Skill, SkillMetadata, SkillParameter, SkillState
from app.agents.state import _override_reducer
from app.ai.llm import extract_text_from_content, llm_service
from app.ai.model_tiers import ModelTier
from app.config import settings
from app.core.logging import get_logger
from app.services.search import SearchResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# State types
# ---------------------------------------------------------------------------


class SubQuery(TypedDict):
    id: str
    query: str
    type: str       # factual|comparative|temporal|statistical|exploratory
    provider: str   # preferred search provider name
    priority: int   # 1-3 (3 = highest)
    status: str     # pending|searched|sufficient|gap


class ExtractedFact(TypedDict):
    claim: str
    source_urls: list[str]
    sub_query_id: str
    confidence: float


class SourceEntry(TypedDict):
    title: str
    url: str
    snippet: str
    content: str | None
    domain: str
    credibility: float
    used_for: list[str]


class Contradiction(TypedDict):
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    sub_query_id: str


class SearchKnowledge(TypedDict, total=False):
    sub_queries: list[SubQuery]
    facts: list[ExtractedFact]
    sources: dict[str, SourceEntry]
    contradictions: list[Contradiction]
    coverage_scores: dict[str, float]
    overall_confidence: float
    refinement_round: int


class AgenticSearchState(SkillState, total=False):
    # Input
    query: str
    complexity: Annotated[str, _override_reducer]
    intent_signals: list[str]

    # Knowledge
    knowledge: Annotated[dict, _override_reducer]

    # Loop control
    max_refinements: Annotated[int, _override_reducer]
    current_refinement: Annotated[int, _override_reducer]

    # Events
    pending_events: Annotated[list[dict[str, Any]], operator.add]

    # Context passthrough
    locale: str
    provider: str | None
    model: str | None
    tier: Any | None

    # Internal evaluation state
    _gap_analysis: Annotated[str, _override_reducer]
    _sufficient: Annotated[bool, _override_reducer]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = (
    "You are a search query classifier. Given a user query, determine:\n\n"
    '1. **complexity**: Is this "trivial" (single web search) '
    'or "complex" (multiple searches, comparison, analysis)?\n\n'
    "Trivial: capital of France, python list sort, weather in Tokyo\n"
    "Complex: compare React vs Vue, economic impacts of AI, "
    "quantum computing developments\n\n"
    "2. **intent_signals**: Pick 1-3 from: factual, comparative, "
    "temporal, statistical, exploratory\n\n"
    "Respond in JSON only:\n"
    '{"complexity": "trivial"|"complex", '
    '"intent_signals": ["factual", ...]}'
)


PLAN_PROMPT = (
    "You are a search strategist. Decompose the query into "
    "2-{max_sub_queries} focused sub-queries.\n\n"
    "For each sub-query, specify:\n"
    '- "query": search query string\n'
    '- "type": factual|comparative|temporal|statistical|'
    "exploratory|news|technical\n"
    '- "provider": "tavily" (general) or "serper" (news)\n'
    '- "priority": 1-3 (3 = most important)\n\n'
    "Respond in JSON only:\n"
    '{{"sub_queries": [{{"query": "...", "type": "...", '
    '"provider": "...", "priority": N}}, ...]}}'
)


EVALUATE_PROMPT = (
    "You are a research evaluator. Assess the search results:\n\n"
    "1. Rate coverage per sub-query from 0.0 to 1.0.\n"
    "2. Identify contradictions between sources.\n"
    "3. Decide: is information sufficient?\n\n"
    "Original query: {query}\n\n"
    "Sub-queries and results:\n{sub_query_summary}\n\n"
    "Respond in JSON only:\n"
    '{{\n  "coverage": {{"<sq_id>": <score>, ...}},\n'
    '  "contradictions": [{{"claim_a": "...", '
    '"source_a": "url", "claim_b": "...", '
    '"source_b": "url", "sub_query_id": "..."}}],\n'
    '  "sufficient": true|false,\n'
    '  "gap_analysis": "missing info or empty string"\n}}'
)


REFINE_PROMPT = (
    "You are a search strategist refining a research plan.\n\n"
    "Original query: {query}\n"
    "Gap analysis: {gap_analysis}\n\n"
    "Current sub-queries and coverage:\n{coverage_summary}\n\n"
    "Generate 1-3 follow-up sub-queries to fill gaps. "
    "Use different search terms.\n\n"
    "Respond in JSON only:\n"
    '{{"follow_up_queries": [{{"query": "...", "type": "...", '
    '"provider": "...", "priority": N}}, ...]}}'
)


SYNTHESIZE_PROMPT = (
    "You are a research synthesizer. Given accumulated sources, "
    "produce a structured summary.\n\n"
    "Original query: {query}\n\n"
    "Sources:\n{sources_summary}\n\n"
    "Contradictions:\n{contradictions_summary}\n\n"
    "Respond in JSON only:\n"
    '{{\n  "facts": [\n'
    '    {{"claim": "...", "sources": ["url1"], '
    '"confidence": 0.0-1.0, "sub_query": "sq_id"}}\n'
    "  ],\n"
    '  "summary": "synthesis with citations [1], [2]",\n'
    '  "unanswered": ["unanswered aspects"],\n'
    '  "confidence": 0.0-1.0\n}}'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (code block markers)
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# AgenticSearchSkill
# ---------------------------------------------------------------------------


class AgenticSearchSkill(Skill):
    """Agentic search with query decomposition, parallel search, and evaluation."""

    metadata = SkillMetadata(
        id="web_research",
        name="Agentic Search",
        version="2.0.0",
        description=(
            "Unified web research skill with tier-aware execution. "
            "LITE: fast single search + summarize. PRO: LLM classifier decides. "
            "MAX: full agentic pipeline with query decomposition and evaluation."
        ),
        category="research",
        parameters=[
            SkillParameter(
                name="query",
                type="string",
                description="The search query or research question",
                required=True,
            ),
            SkillParameter(
                name="topic",
                type="string",
                description="Alias for query (backward compatibility)",
                required=False,
            ),
            SkillParameter(
                name="max_sources",
                type="number",
                description="Maximum sources per sub-query (1-10)",
                required=False,
                default=5,
            ),
            SkillParameter(
                name="locale",
                type="string",
                description="Locale for the search",
                required=False,
                default="en",
            ),
            SkillParameter(
                name="provider",
                type="string",
                description="LLM provider to use",
                required=False,
            ),
        ],
        output_schema={
            "type": "object",
            "properties": {
                "facts": {"type": "array"},
                "sources": {"type": "array"},
                "overall_confidence": {"type": "number"},
                "coverage": {"type": "object"},
                "unanswered": {"type": "array"},
                "contradictions": {"type": "array"},
                "complexity": {"type": "string"},
                "summary": {"type": "string"},
            },
        },
        required_tools=["web_search"],
        risk_level="medium",
        side_effect_level="low",
        data_sensitivity="public",
        network_scope="external",
        max_execution_time_seconds=300,
        max_iterations=20,
        tags=["search", "research", "web", "agentic"],
    )

    def create_graph(self):
        """Build the agentic search LangGraph."""
        graph = StateGraph(AgenticSearchState)

        # ---- Node: classify ----
        async def classify(state: AgenticSearchState) -> dict:
            params = state.get("input_params", {})
            query = params.get("query") or params.get("topic", "")
            locale = params.get("locale", state.get("locale", "en"))
            provider = params.get("provider", state.get("provider"))
            model = params.get("model", state.get("model"))
            tier = params.get("tier", state.get("tier"))

            pending_events: list[dict] = [
                agent_events.stage("classifying", "Analyzing query complexity...", "running"),
            ]

            # Resolve tier for tier-aware branching
            resolved_tier: ModelTier | None = None
            if isinstance(tier, ModelTier):
                resolved_tier = tier
            elif isinstance(tier, str):
                try:
                    resolved_tier = ModelTier(tier.lower())
                except ValueError:
                    resolved_tier = None

            if resolved_tier == ModelTier.LITE:
                # LITE: always fast path — skip LLM classify
                complexity = "trivial"
                intent_signals = ["factual"]
            elif resolved_tier == ModelTier.MAX:
                # MAX: always full agentic pipeline — skip LLM classify
                complexity = "complex"
                intent_signals = ["exploratory"]
            else:
                # PRO or unknown: use LLM classifier (existing behavior)
                try:
                    llm = llm_service.get_llm_for_tier(ModelTier.LITE, provider=provider)
                    resp = await llm.ainvoke([
                        SystemMessage(content=CLASSIFY_PROMPT),
                        HumanMessage(content=f"Query: {query}"),
                    ])
                    text = extract_text_from_content(resp.content) or ""
                    result = _parse_json_response(text)
                    complexity = result.get("complexity", "complex")
                    intent_signals = result.get("intent_signals", ["factual"])
                except Exception as e:
                    logger.warning("classify_failed_defaulting_complex", error=str(e))
                    complexity = "complex"
                    intent_signals = ["exploratory"]

            pending_events.append(
                agent_events.stage("classifying", f"Query classified as {complexity}", "completed"),
            )

            knowledge: dict = {
                "sub_queries": [],
                "facts": [],
                "sources": {},
                "contradictions": [],
                "coverage_scores": {},
                "overall_confidence": 0.0,
                "refinement_round": 0,
            }

            return {
                "query": query,
                "complexity": complexity,
                "intent_signals": intent_signals,
                "knowledge": knowledge,
                "max_refinements": settings.web_research_max_refinements,
                "current_refinement": 0,
                "locale": locale,
                "provider": provider,
                "model": model,
                "tier": tier,
                "pending_events": pending_events,
                "_gap_analysis": "",
                "_sufficient": False,
            }

        # ---- Node: quick_search ----
        async def quick_search(state: AgenticSearchState) -> dict:
            """Fast path for trivial queries — single search, no LLM planning."""
            query = state.get("query", "")
            intent_signals = state.get("intent_signals", ["factual"])
            pending_events: list[dict] = [
                agent_events.stage("searching", "Quick search...", "running"),
            ]

            from app.services.search_providers import get_search_registry
            from app.services.search_providers.credibility import score_domain

            registry = get_search_registry()
            query_type = intent_signals[0] if intent_signals else "factual"

            try:
                results = await registry.search(query, query_type=query_type, max_results=5)
            except Exception as e:
                logger.error("quick_search_failed", error=str(e))
                results = []

            knowledge = dict(state.get("knowledge", {}))
            sources = dict(knowledge.get("sources", {}))

            for r in results:
                url = r.url
                if url not in sources:
                    sources[url] = {
                        "title": r.title,
                        "url": url,
                        "snippet": r.snippet,
                        "content": r.content,
                        "domain": url.split("/")[2] if len(url.split("/")) > 2 else "",
                        "credibility": score_domain(url),
                        "used_for": ["quick"],
                    }
                    pending_events.append(
                        agent_events.source(
                            title=r.title, url=url, snippet=r.snippet,
                            relevance_score=r.relevance_score,
                        )
                    )

            knowledge["sources"] = sources
            knowledge["overall_confidence"] = 0.7 if results else 0.0

            pending_events.append(
                agent_events.stage("searching", "Quick search complete", "completed"),
            )

            return {
                "knowledge": knowledge,
                "pending_events": pending_events,
            }

        # ---- Node: plan_search ----
        async def plan_search(state: AgenticSearchState) -> dict:
            """Decompose complex query into sub-queries."""
            query = state.get("query", "")
            provider = state.get("provider")
            pending_events: list[dict] = [
                agent_events.stage("planning", "Decomposing query into sub-queries...", "running"),
            ]

            max_sq = settings.web_research_max_sub_queries

            try:
                llm = llm_service.get_llm_for_tier(ModelTier.PRO, provider=provider)
                prompt = PLAN_PROMPT.replace("{max_sub_queries}", str(max_sq))
                resp = await llm.ainvoke([
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"Query: {query}"),
                ])
                text = extract_text_from_content(resp.content) or ""
                plan = _parse_json_response(text)
                raw_sqs = plan.get("sub_queries", [])
            except Exception as e:
                logger.warning("plan_search_failed", error=str(e))
                raw_sqs = [{
                    "query": query, "type": "exploratory",
                    "provider": "tavily", "priority": 3,
                }]

            sub_queries: list[dict] = []
            for i, sq in enumerate(raw_sqs[:max_sq]):
                sub_queries.append({
                    "id": f"sq{i + 1}",
                    "query": sq.get("query", query),
                    "type": sq.get("type", "factual"),
                    "provider": sq.get("provider", "tavily"),
                    "priority": sq.get("priority", 2),
                    "status": "pending",
                })

            knowledge = dict(state.get("knowledge", {}))
            knowledge["sub_queries"] = sub_queries

            pending_events.append(
                agent_events.search_plan(
                    sub_queries=[
                        {
                            "id": sq["id"],
                            "query": sq["query"],
                            "type": sq["type"],
                            "provider": sq["provider"],
                        }
                        for sq in sub_queries
                    ]
                )
            )
            pending_events.append(
                agent_events.stage(
                    "planning",
                    f"Planned {len(sub_queries)} sub-queries",
                    "completed",
                ),
            )

            return {
                "knowledge": knowledge,
                "pending_events": pending_events,
            }

        # ---- Node: execute_search ----
        async def execute_search(state: AgenticSearchState) -> dict:
            """Execute pending sub-queries in parallel."""
            knowledge = dict(state.get("knowledge", {}))
            sub_queries = list(knowledge.get("sub_queries", []))
            pending_events: list[dict] = [
                agent_events.stage("searching", "Executing searches in parallel...", "running"),
            ]

            from app.services.search_providers import get_search_registry
            from app.services.search_providers.credibility import score_domain

            registry = get_search_registry()
            max_sources = int(state.get("input_params", {}).get("max_sources", 5))

            pending_sqs = [sq for sq in sub_queries if sq["status"] == "pending"]

            async def _search_one(sq: dict) -> tuple[str, list[SearchResult]]:
                try:
                    results = await registry.search(
                        sq["query"],
                        query_type=sq["type"],
                        max_results=max_sources,
                    )
                    return sq["id"], results
                except Exception as e:
                    logger.error("sub_query_search_failed", sq_id=sq["id"], error=str(e))
                    return sq["id"], []

            # Emit searching status for all pending
            for sq in pending_sqs:
                pending_events.append(
                    agent_events.sub_query_status(id=sq["id"], status="searching")
                )

            # Fan out searches in parallel
            tasks = [_search_one(sq) for sq in pending_sqs]
            results_by_sq = await asyncio.gather(*tasks)

            sources = dict(knowledge.get("sources", {}))
            for sq_id, results in results_by_sq:
                for r in results:
                    url = r.url
                    if url not in sources:
                        sources[url] = {
                            "title": r.title,
                            "url": url,
                            "snippet": r.snippet,
                            "content": r.content,
                            "domain": url.split("/")[2] if len(url.split("/")) > 2 else "",
                            "credibility": score_domain(url),
                            "used_for": [sq_id],
                        }
                        pending_events.append(
                            agent_events.source(
                                title=r.title, url=url, snippet=r.snippet,
                                relevance_score=r.relevance_score,
                            )
                        )
                    else:
                        if sq_id not in sources[url]["used_for"]:
                            sources[url]["used_for"].append(sq_id)

                # Mark sub-query as searched
                for sq in sub_queries:
                    if sq["id"] == sq_id:
                        sq["status"] = "searched"
                        pending_events.append(
                            agent_events.sub_query_status(id=sq_id, status="done")
                        )

            knowledge["sources"] = sources
            knowledge["sub_queries"] = sub_queries

            pending_events.append(
                agent_events.knowledge_update(
                    facts_count=len(knowledge.get("facts", [])),
                    sources_count=len(sources),
                )
            )
            pending_events.append(
                agent_events.stage("searching", f"Found {len(sources)} sources", "completed"),
            )

            return {
                "knowledge": knowledge,
                "pending_events": pending_events,
            }

        # ---- Node: evaluate ----
        async def evaluate(state: AgenticSearchState) -> dict:
            """Evaluate search coverage and decide if refinement is needed."""
            query = state.get("query", "")
            knowledge = dict(state.get("knowledge", {}))
            provider = state.get("provider")
            pending_events: list[dict] = [
                agent_events.stage("evaluating", "Evaluating search coverage...", "running"),
            ]

            sub_queries = knowledge.get("sub_queries", [])
            sources = knowledge.get("sources", {})

            # Build summary for LLM
            summary_parts = []
            for sq in sub_queries:
                sq_sources = [
                    f"  - {s['title']}: {s['snippet'][:150]}"
                    for s in sources.values()
                    if sq["id"] in s.get("used_for", [])
                ]
                summary_parts.append(
                    f"[{sq['id']}] {sq['query']} (type: {sq['type']}, status: {sq['status']})\n"
                    + ("\n".join(sq_sources) if sq_sources else "  No results found.")
                )
            sub_query_summary = "\n\n".join(summary_parts)

            try:
                llm = llm_service.get_llm_for_tier(ModelTier.PRO, provider=provider)
                prompt = EVALUATE_PROMPT.format(
                    query=query, sub_query_summary=sub_query_summary
                )
                resp = await llm.ainvoke([
                    SystemMessage(content="You are a research evaluator. Respond in JSON only."),
                    HumanMessage(content=prompt),
                ])
                text = extract_text_from_content(resp.content) or ""
                evaluation = _parse_json_response(text)
            except Exception as e:
                logger.warning("evaluate_failed", error=str(e))
                evaluation = {
                    "coverage": {},
                    "contradictions": [],
                    "sufficient": True,
                    "gap_analysis": "",
                }

            coverage = evaluation.get("coverage", {})
            knowledge["coverage_scores"] = coverage
            knowledge["contradictions"] = evaluation.get("contradictions", [])

            # Calculate weighted confidence
            total_weight = 0.0
            weighted_sum = 0.0
            for sq in sub_queries:
                sq_coverage = coverage.get(sq["id"], 0.5)
                weight = sq.get("priority", 2)
                weighted_sum += sq_coverage * weight
                total_weight += weight
                sq["status"] = "sufficient" if sq_coverage >= 0.6 else "gap"

            overall = weighted_sum / total_weight if total_weight > 0 else 0.0
            knowledge["overall_confidence"] = overall
            knowledge["sub_queries"] = sub_queries

            sufficient = evaluation.get("sufficient", True)
            gap_analysis = evaluation.get("gap_analysis", "")

            pending_events.append(
                agent_events.confidence_update(
                    confidence=overall, coverage_summary=coverage
                )
            )
            pending_events.append(
                agent_events.stage("evaluating", f"Confidence: {overall:.0%}", "completed"),
            )

            return {
                "knowledge": knowledge,
                "_gap_analysis": gap_analysis,
                "_sufficient": sufficient,
                "pending_events": pending_events,
            }

        # ---- Node: refine ----
        async def refine(state: AgenticSearchState) -> dict:
            """Generate follow-up queries for gaps."""
            query = state.get("query", "")
            knowledge = dict(state.get("knowledge", {}))
            provider = state.get("provider")
            current_refinement = state.get("current_refinement", 0)
            gap_analysis = state.get("_gap_analysis", "")

            pending_events: list[dict] = []

            sub_queries = list(knowledge.get("sub_queries", []))
            coverage = knowledge.get("coverage_scores", {})

            coverage_summary = "\n".join(
                f"  [{sq['id']}] {sq['query']} — coverage: {coverage.get(sq['id'], 0):.1%}"
                for sq in sub_queries
            )

            try:
                llm = llm_service.get_llm_for_tier(ModelTier.PRO, provider=provider)
                prompt = REFINE_PROMPT.format(
                    query=query,
                    gap_analysis=gap_analysis,
                    coverage_summary=coverage_summary,
                )
                resp = await llm.ainvoke([
                    SystemMessage(content="You are a search strategist. Respond in JSON only."),
                    HumanMessage(content=prompt),
                ])
                text = extract_text_from_content(resp.content) or ""
                result = _parse_json_response(text)
                follow_ups = result.get("follow_up_queries", [])
            except Exception as e:
                logger.warning("refine_failed", error=str(e))
                follow_ups = []

            existing_count = len(sub_queries)
            for i, sq in enumerate(follow_ups[:3]):
                sub_queries.append({
                    "id": f"sq{existing_count + i + 1}",
                    "query": sq.get("query", ""),
                    "type": sq.get("type", "factual"),
                    "provider": sq.get("provider", "tavily"),
                    "priority": sq.get("priority", 2),
                    "status": "pending",
                })

            knowledge["sub_queries"] = sub_queries
            knowledge["refinement_round"] = current_refinement + 1

            pending_events.append(
                agent_events.refinement_start(
                    round=current_refinement + 1,
                    follow_up_queries=[sq.get("query", "") for sq in follow_ups[:3]],
                )
            )

            return {
                "knowledge": knowledge,
                "current_refinement": current_refinement + 1,
                "pending_events": pending_events,
            }

        # ---- Node: synthesize ----
        async def synthesize(state: AgenticSearchState) -> dict:
            """Build structured output from accumulated knowledge."""
            query = state.get("query", "")
            knowledge = dict(state.get("knowledge", {}))
            complexity = state.get("complexity", "trivial")
            provider = state.get("provider")

            pending_events: list[dict] = [
                agent_events.stage("synthesizing", "Building structured findings...", "running"),
            ]

            sources = knowledge.get("sources", {})
            sub_queries = knowledge.get("sub_queries", [])
            contradictions = knowledge.get("contradictions", [])
            overall_confidence = knowledge.get("overall_confidence", 0.0)

            source_list = [
                {
                    "title": s["title"],
                    "url": s["url"],
                    "snippet": s["snippet"],
                    "credibility": s["credibility"],
                }
                for s in sources.values()
            ]

            # Provider usage stats
            provider_usage: dict[str, int] = {}
            for sq in sub_queries:
                p = sq.get("provider", "tavily")
                provider_usage[p] = provider_usage.get(p, 0) + 1

            # For trivial queries, skip LLM synthesis
            if complexity == "trivial":
                pending_events.append(
                    agent_events.stage("synthesizing", "Complete", "completed"),
                )
                return {
                    "output": {
                        "facts": [],
                        "sources": source_list,
                        "overall_confidence": overall_confidence,
                        "coverage": knowledge.get("coverage_scores", {}),
                        "unanswered": [],
                        "contradictions": [],
                        "complexity": complexity,
                        "summary": "",
                        "sub_queries_count": 0,
                        "sources_count": len(source_list),
                        "refinement_rounds": 0,
                        "provider_usage": {},
                    },
                    "pending_events": pending_events,
                }

            # Complex: LLM synthesis
            sources_summary = "\n".join(
                f"[{i + 1}] {s['title']} — {s['url']} "
                f"(credibility: {s['credibility']:.1f})\n"
                f"    {s['snippet'][:200]}"
                for i, s in enumerate(sources.values())
            )
            contradictions_summary = (
                "\n".join(
                    f"- {c.get('claim_a', '')} vs {c.get('claim_b', '')} "
                    f"({c.get('source_a', '')} vs {c.get('source_b', '')})"
                    for c in contradictions
                )
                if contradictions
                else "No contradictions detected."
            )

            try:
                llm = llm_service.get_llm_for_tier(ModelTier.PRO, provider=provider)
                prompt = SYNTHESIZE_PROMPT.format(
                    query=query,
                    sources_summary=sources_summary,
                    contradictions_summary=contradictions_summary,
                )
                resp = await llm.ainvoke([
                    SystemMessage(content="You are a research synthesizer. Respond in JSON only."),
                    HumanMessage(content=prompt),
                ])
                text = extract_text_from_content(resp.content) or ""
                synthesis = _parse_json_response(text)
            except Exception as e:
                logger.warning("synthesize_failed", error=str(e))
                synthesis = {
                    "facts": [],
                    "summary": "",
                    "unanswered": [],
                    "confidence": overall_confidence,
                }

            unanswered = list(synthesis.get("unanswered", []))
            for sq in sub_queries:
                if sq["status"] == "gap":
                    unanswered.append(sq["query"])

            pending_events.append(
                agent_events.stage("synthesizing", "Complete", "completed"),
            )

            return {
                "output": {
                    "facts": synthesis.get("facts", []),
                    "sources": source_list,
                    "overall_confidence": synthesis.get("confidence", overall_confidence),
                    "coverage": knowledge.get("coverage_scores", {}),
                    "unanswered": unanswered,
                    "contradictions": [
                        {
                            "claim_a": c.get("claim_a"),
                            "claim_b": c.get("claim_b"),
                            "source_a": c.get("source_a"),
                            "source_b": c.get("source_b"),
                        }
                        for c in contradictions
                    ],
                    "complexity": complexity,
                    "summary": synthesis.get("summary", ""),
                    "sub_queries_count": len(sub_queries),
                    "sources_count": len(source_list),
                    "refinement_rounds": knowledge.get("refinement_round", 0),
                    "provider_usage": provider_usage,
                },
                "pending_events": pending_events,
            }

        # ---- Conditional edges ----
        def after_classify(state: AgenticSearchState) -> str:
            if state.get("complexity") == "trivial":
                return "quick_search"
            return "plan_search"

        def after_evaluate(state: AgenticSearchState) -> str:
            current_ref = state.get("current_refinement", 0)
            max_ref = state.get("max_refinements", 3)
            threshold = settings.web_research_confidence_threshold
            sufficient = state.get("_sufficient", True)
            knowledge = state.get("knowledge", {})
            confidence = knowledge.get("overall_confidence", 0.0)

            if sufficient or confidence >= threshold or current_ref >= max_ref:
                return "synthesize"
            return "refine"

        # ---- Build graph ----
        graph.add_node("classify", classify)
        graph.add_node("quick_search", quick_search)
        graph.add_node("plan_search", plan_search)
        graph.add_node("execute_search", execute_search)
        graph.add_node("evaluate", evaluate)
        graph.add_node("refine", refine)
        graph.add_node("synthesize", synthesize)

        graph.set_entry_point("classify")
        graph.add_conditional_edges(
            "classify",
            after_classify,
            {"quick_search": "quick_search", "plan_search": "plan_search"},
        )
        graph.add_edge("quick_search", "synthesize")
        graph.add_edge("plan_search", "execute_search")
        graph.add_edge("execute_search", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            after_evaluate,
            {"synthesize": "synthesize", "refine": "refine"},
        )
        graph.add_edge("refine", "execute_search")
        graph.add_edge("synthesize", END)

        return graph.compile()
