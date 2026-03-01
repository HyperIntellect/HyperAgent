"""Web Research Skill for focused web research."""

from langgraph.graph import StateGraph, END

from app.ai.llm import llm_service
from app.ai.model_tiers import ModelTier
from app.core.logging import get_logger
from app.services.search import search_service
from app.agents.skills.skill_base import Skill, SkillMetadata, SkillParameter, SkillState

logger = get_logger(__name__)


class WebResearchSkill(Skill):
    """Performs focused web research on a topic and provides summarized findings."""

    metadata = SkillMetadata(
        id="web_research",
        name="Web Research",
        version="2.0.0",
        description="Performs focused web research on a topic and provides summarized findings with sources",
        category="research",
        parameters=[
            SkillParameter(
                name="topic",
                type="string",
                description="Research topic or question to investigate",
                required=True,
            ),
            SkillParameter(
                name="max_sources",
                type="number",
                description="Maximum number of sources to gather (1-10)",
                required=False,
                default=5,
            ),
        ],
        output_schema={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Research summary"},
                "sources": {
                    "type": "array",
                    "description": "List of source documents",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "snippet": {"type": "string"},
                        },
                    },
                },
                "key_findings": {
                    "type": "array",
                    "description": "Key findings from research",
                    "items": {"type": "string"},
                },
            },
        },
        required_tools=[],
        risk_level="low",
        side_effect_level="none",
        data_sensitivity="public",
        network_scope="external",
        idempotency_hint=True,
        max_iterations=3,
        tags=["research", "web", "search"],
    )

    def create_graph(self) -> StateGraph:
        """Create the LangGraph subgraph for web research."""
        graph = StateGraph(SkillState)
        
        from pydantic import BaseModel, Field

        class ResearchSummaryResponse(BaseModel):
            """Structured response for research summary."""
            summary: str = Field(description="Comprehensive summary of the research findings based on sources")
            key_findings: list[str] = Field(description="List of 3-5 key insights or findings as bullet points")

        async def search_node(state: SkillState) -> dict:
            """Search for information on the topic."""
            topic = state["input_params"]["topic"]
            max_sources = int(state["input_params"].get("max_sources", 5))

            logger.info("web_research_skill_searching", topic=topic, max_sources=max_sources)

            try:
                # Perform web search
                results = await search_service.search_raw(
                    query=topic,
                    max_results=min(max_sources, 10),
                    search_depth="advanced",
                )

                if not results:
                    return {
                        "output": {
                            "summary": f"No results found for: {topic}",
                            "sources": [],
                            "key_findings": [],
                        },
                        "iterations": state["iterations"] + 1,
                    }

                # Format sources
                sources = [
                    {
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet or "",
                        "content": r.content or "",
                    }
                    for r in results
                ]

                return {
                    "output": {"sources": sources},
                    "iterations": state["iterations"] + 1,
                }

            except Exception as e:
                logger.error("web_research_skill_search_failed", error=str(e))
                return {
                    "error": f"Search failed: {str(e)}",
                    "iterations": state["iterations"] + 1,
                }

        async def summarize_node(state: SkillState) -> dict:
            """Summarize research findings."""
            topic = state["input_params"]["topic"]
            sources = state["output"].get("sources", [])

            if not sources:
                return {
                    "output": {
                        "summary": "No sources to summarize",
                        "sources": [],
                        "key_findings": [],
                    },
                    "iterations": state["iterations"] + 1,
                }

            logger.info("web_research_skill_summarizing", source_count=len(sources))

            try:
                # Build context from sources
                source_context = "\n\n".join([
                    f"**{s['title']}**\n{s['url']}\n{s.get('snippet', '')}"
                    for s in sources
                ])

                # Create prompt for summarization
                prompt = f"""Research Topic: {topic}

Sources:
{source_context}

Based on the above sources, provide:
1. A comprehensive summary of the findings
2. 3-5 key insights or findings"""

                # Get LLM for summarization
                llm = llm_service.get_llm_for_tier(ModelTier.PRO)
                
                # Use structured output
                structured_llm = llm.with_structured_output(ResearchSummaryResponse)
                
                # Generate summary
                result: ResearchSummaryResponse = await structured_llm.ainvoke(prompt)

                return {
                    "output": {
                        "summary": result.summary,
                        "sources": [
                            {"title": s["title"], "url": s["url"], "snippet": s.get("snippet", "")}
                            for s in sources
                        ],
                        "key_findings": result.key_findings,
                    },
                    "iterations": state["iterations"] + 1,
                }

            except Exception as e:
                logger.error("web_research_skill_summarize_failed", error=str(e))
                return {
                    "error": f"Summarization failed: {str(e)}",
                    "iterations": state["iterations"] + 1,
                }

        # Build graph
        graph.add_node("search", search_node)
        graph.add_node("summarize", summarize_node)
        graph.set_entry_point("search")
        graph.add_edge("search", "summarize")
        graph.add_edge("summarize", END)

        return graph.compile()
