"""Slide Generation Skill for creating PPTX presentations."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.skills.skill_base import Skill, SkillMetadata, SkillParameter, SkillState
from app.ai.image import image_generation_service
from app.ai.llm import llm_service
from app.services.pptx_gen import (
    SlideDeck,
    SlideElement,
    SlideSpec,
    SlideTheme,
    pptx_generation_service,
)
from app.core.logging import get_logger
from app.services.file_storage import file_storage_service
from app.services.search import search_service

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------

STYLE_THEMES: dict[str, SlideTheme] = {
    "professional": SlideTheme(
        primary_color="#1B365D",
        secondary_color="#4A90D9",
        accent_color="#E8792F",
        font_heading="Calibri",
        font_body="Calibri",
    ),
    "creative": SlideTheme(
        primary_color="#6C3483",
        secondary_color="#F39C12",
        accent_color="#1ABC9C",
        font_heading="Georgia",
        font_body="Calibri",
    ),
    "minimal": SlideTheme(
        primary_color="#2C3E50",
        secondary_color="#7F8C8D",
        accent_color="#3498DB",
        font_heading="Calibri Light",
        font_body="Calibri Light",
    ),
    "bold": SlideTheme(
        primary_color="#C0392B",
        secondary_color="#2C3E50",
        accent_color="#F39C12",
        font_heading="Arial Black",
        font_body="Arial",
    ),
}


class SlideGenerationSkill(Skill):
    """Generates PPTX presentations from a topic description."""

    metadata = SkillMetadata(
        id="slide_generation",
        name="Slide Generation",
        version="1.0.0",
        description=(
            "Creates professional PPTX presentations by researching a topic, "
            "generating a structured outline with an LLM, and producing a "
            "downloadable PowerPoint file."
        ),
        category="creative",
        parameters=[
            SkillParameter(
                name="topic",
                type="string",
                description="Topic or description for the presentation",
                required=True,
            ),
            SkillParameter(
                name="num_slides",
                type="number",
                description="Number of slides (4-20)",
                required=False,
                default=8,
            ),
            SkillParameter(
                name="style",
                type="string",
                description="Visual style: professional, creative, minimal, bold",
                required=False,
                default="professional",
            ),
            SkillParameter(
                name="include_images",
                type="boolean",
                description="Whether to generate images for slides",
                required=False,
                default=False,
            ),
            SkillParameter(
                name="additional_context",
                type="string",
                description="Additional context or requirements",
                required=False,
                default=None,
            ),
        ],
        output_schema={
            "type": "object",
            "properties": {
                "download_url": {
                    "type": "string",
                    "description": "URL to download the generated PPTX",
                },
                "storage_key": {
                    "type": "string",
                    "description": "Storage key for the PPTX file",
                },
                "slide_count": {
                    "type": "number",
                    "description": "Number of slides in the presentation",
                },
                "title": {
                    "type": "string",
                    "description": "Title of the presentation",
                },
            },
        },
        required_tools=[],
        risk_level="medium",
        side_effect_level="medium",
        data_sensitivity="internal",
        network_scope="external",
        idempotency_hint=False,
        max_execution_time_seconds=300,
        max_iterations=5,
        tags=["slides", "presentation", "pptx", "creative"],
    )

    def create_graph(self) -> StateGraph:
        """Create the LangGraph subgraph for slide generation."""
        graph = StateGraph(SkillState)

        # ------------------------------------------------------------------
        # Node 1: Research
        # ------------------------------------------------------------------

        async def research_node(state: SkillState) -> dict:
            """Research the topic using web search."""
            topic = state["input_params"]["topic"]

            logger.info("slide_generation_research_started", topic=topic[:100])

            events = list(state.get("pending_events", []))
            events.append({"stage": "Researching topic..."})

            try:
                results = await search_service.search_raw(
                    query=topic,
                    max_results=8,
                    search_depth="advanced",
                )

                context_parts: list[str] = []
                sources: list[dict] = []
                for r in results:
                    context_parts.append(f"- {r.title}: {r.snippet}")
                    sources.append({"title": r.title, "url": r.url})

                research_context = "\n".join(context_parts) if context_parts else ""

                logger.info(
                    "slide_generation_research_completed",
                    topic=topic[:80],
                    source_count=len(sources),
                )

                return {
                    "output": {
                        "research_context": research_context,
                        "sources": sources,
                    },
                    "pending_events": events,
                    "iterations": state.get("iterations", 0) + 1,
                }

            except Exception as e:
                logger.warning("slide_generation_research_failed", error=str(e))
                # Continue without research context rather than failing entirely
                return {
                    "output": {
                        "research_context": "",
                        "sources": [],
                    },
                    "pending_events": events,
                    "iterations": state.get("iterations", 0) + 1,
                }

        # ------------------------------------------------------------------
        # Node 2: Outline
        # ------------------------------------------------------------------

        async def outline_node(state: SkillState) -> dict:
            """Generate a structured slide outline using the LLM."""
            params = state["input_params"]
            topic = params["topic"]
            num_slides = int(params.get("num_slides", 8))
            style = params.get("style", "professional")
            additional_context = params.get("additional_context", "")
            research_context = state.get("output", {}).get("research_context", "")
            sources = state.get("output", {}).get("sources", [])

            events = list(state.get("pending_events", []))
            events.append({"stage": "Creating slide outline..."})

            logger.info(
                "slide_generation_outline_started",
                topic=topic[:80],
                num_slides=num_slides,
                style=style,
            )

            # Build the LLM prompt
            research_section = ""
            if research_context:
                research_section = (
                    "Research context (use this to ground your "
                    "content with real data):\n" + research_context
                )

            prompt = f"""Create a complete, content-rich presentation for: "{topic}"

Number of slides: {num_slides}
Style: {style}

{f"Additional requirements: {additional_context}" if additional_context else ""}

{research_section}

Available layouts: title_slide, section_header, content, two_column, blank
- Start with a title_slide
- Use section_header to divide major sections
- Use content for detailed slides (3-6 bullet points, 1-2 sentences each)
- Use two_column for comparisons
- End with a content slide for summary/conclusion or a title_slide for a closing message
- Write substantive, presentation-ready content — not just outlines
- Include notes with speaker talking points on every content slide
- Include image_prompt on slides where a visual would add value
- Ensure exactly {num_slides} slides total"""

            try:
                from app.agents.tools.slide_generation import SlideOutlineSchema

                from app.ai.model_tiers import ModelTier

                llm = llm_service.get_llm_for_tier(ModelTier.PRO)
                structured_llm = llm.with_structured_output(SlideOutlineSchema)
                outline: SlideOutlineSchema = await structured_llm.ainvoke(prompt)

                deck_title = outline.title or topic
                raw_slides = outline.slides

                # Convert to SlideDeck, preserving image_prompt per slide
                theme = STYLE_THEMES.get(style, STYLE_THEMES["professional"])
                slides: list[SlideSpec] = []
                image_prompts: dict[int, str] = {}
                for idx, s in enumerate(raw_slides):
                    elements = []
                    for el in s.elements:
                        elements.append(
                            SlideElement(
                                type=el.type or "text",
                                content=el.content or "",
                            )
                        )
                    slides.append(
                        SlideSpec(
                            layout=s.layout or "content",
                            title=s.title or "",
                            subtitle=s.subtitle,
                            elements=elements,
                            notes=s.notes,
                        )
                    )
                    if s.image_prompt:
                        image_prompts[idx] = s.image_prompt

                deck = SlideDeck(
                    title=deck_title,
                    theme=theme,
                    slides=slides,
                )

                logger.info(
                    "slide_generation_outline_completed",
                    title=deck_title,
                    slide_count=len(slides),
                )

                return {
                    "output": {
                        "deck": deck.model_dump(),
                        "sources": sources,
                        "image_prompts": image_prompts,
                        "include_images": bool(params.get("include_images", False)),
                    },
                    "pending_events": events,
                    "iterations": state.get("iterations", 0) + 1,
                }

            except (KeyError, TypeError, ValueError) as e:
                logger.warning("slide_outline_structured_output_failed", error=str(e))
                # Fallback: create a basic deck
                theme = STYLE_THEMES.get(style, STYLE_THEMES["professional"])
                fallback_deck = SlideDeck(
                    title=topic,
                    theme=theme,
                    slides=[
                        SlideSpec(
                            layout="title_slide",
                            title=topic,
                            subtitle="Generated Presentation",
                        ),
                        SlideSpec(
                            layout="content",
                            title="Overview",
                            elements=[
                                SlideElement(
                                    type="text",
                                    content=f"An overview of {topic}",
                                ),
                            ],
                        ),
                        SlideSpec(
                            layout="content",
                            title="Summary",
                            elements=[
                                SlideElement(
                                    type="text",
                                    content="Thank you for your attention.",
                                ),
                            ],
                        ),
                    ],
                )
                return {
                    "output": {
                        "deck": fallback_deck.model_dump(),
                        "sources": sources,
                    },
                    "pending_events": events,
                    "iterations": state.get("iterations", 0) + 1,
                }

            except Exception as e:
                logger.error("slide_outline_failed", error=str(e))
                return {
                    "error": f"Failed to create slide outline: {e}",
                    "pending_events": events,
                    "iterations": state.get("iterations", 0) + 1,
                }

        # ------------------------------------------------------------------
        # Node 3: Image Generation
        # ------------------------------------------------------------------

        async def images_node(state: SkillState) -> dict:
            """Generate images for slides that have image_prompt set."""
            output = state.get("output", {})
            deck_data = output.get("deck")
            image_prompts: dict = output.get("image_prompts", {})
            include_images = output.get("include_images", False)

            events = list(state.get("pending_events", []))

            if not deck_data or not include_images or not image_prompts:
                # Nothing to do — pass through
                return {
                    "pending_events": events,
                    "iterations": state.get("iterations", 0) + 1,
                }

            events.append({"stage": "Generating images for slides..."})

            deck = SlideDeck(**deck_data)
            # Limit to max 4 images to keep generation time reasonable
            prompts_to_generate = list(image_prompts.items())[:4]
            generated_count = 0

            for slide_idx_str, prompt_text in prompts_to_generate:
                slide_idx = int(slide_idx_str) if isinstance(slide_idx_str, str) else slide_idx_str
                if slide_idx < 0 or slide_idx >= len(deck.slides):
                    continue

                try:
                    results = await image_generation_service.generate_image(
                        prompt=prompt_text,
                        size="1792x1024",
                        n=1,
                    )
                    if results and results[0].base64_data:
                        data_uri = f"data:image/png;base64,{results[0].base64_data}"

                        # Save to storage for preview URL
                        preview_url = ""
                        user_id = state.get("user_id")
                        if user_id:
                            try:
                                import base64 as b64mod

                                image_bytes = b64mod.b64decode(results[0].base64_data)
                                storage_result = await file_storage_service.save_generated_image(
                                    image_data=image_bytes,
                                    user_id=user_id,
                                    content_type="image/png",
                                    metadata={"type": "slide_image", "slide_index": slide_idx},
                                )
                                preview_url = storage_result["url"]
                            except Exception as e:
                                logger.warning("slide_image_storage_failed", error=str(e))

                        deck.slides[slide_idx].elements.append(
                            SlideElement(
                                type="image",
                                content=data_uri,
                                style={"preview_url": preview_url},
                            )
                        )
                        generated_count += 1
                        logger.info(
                            "slide_image_generated",
                            slide_index=slide_idx,
                            prompt=prompt_text[:60],
                        )
                except Exception as e:
                    logger.warning(
                        "slide_image_generation_failed",
                        slide_index=slide_idx,
                        error=str(e),
                    )

            logger.info(
                "slide_images_node_completed",
                generated=generated_count,
                requested=len(prompts_to_generate),
            )

            return {
                "output": {
                    **output,
                    "deck": deck.model_dump(),
                },
                "pending_events": events,
                "iterations": state.get("iterations", 0) + 1,
            }

        # ------------------------------------------------------------------
        # Node 4: Generate PPTX
        # ------------------------------------------------------------------

        async def generate_node(state: SkillState) -> dict:
            """Generate the PPTX file from the outline."""
            user_id = state.get("user_id")
            output = state.get("output", {})
            deck_data = output.get("deck")
            sources = output.get("sources", [])

            events = list(state.get("pending_events", []))
            events.append({"stage": "Generating presentation..."})

            if not deck_data:
                return {
                    "error": "No slide deck data available",
                    "pending_events": events,
                    "iterations": state.get("iterations", 0) + 1,
                }

            try:
                deck = SlideDeck(**deck_data)
                pptx_bytes = pptx_generation_service.generate_pptx(deck)

                logger.info(
                    "slide_generation_pptx_created",
                    title=deck.title,
                    slide_count=len(deck.slides),
                    size_bytes=len(pptx_bytes),
                )

                # Build slide outline for frontend preview
                slide_outline = [
                    {
                        "layout": s.layout,
                        "title": s.title,
                        "subtitle": s.subtitle,
                        "elements": [
                            {
                                "type": el.type,
                                "content": (
                                    el.style.get("preview_url", "")
                                    if el.type == "image"
                                    else el.content
                                ),
                            }
                            for el in s.elements
                            if el.type == "text"
                            or (el.type == "image" and el.style.get("preview_url"))
                        ],
                        "notes": s.notes,
                    }
                    for s in deck.slides
                ]

                # Save to file storage
                result_output: dict = {
                    "title": deck.title,
                    "slide_count": len(deck.slides),
                    "sources": sources,
                    "slide_outline": slide_outline,
                }

                if user_id:
                    try:
                        storage_result = await file_storage_service.save_generated_image(
                            image_data=pptx_bytes,
                            user_id=user_id,
                            content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            metadata={
                                "type": "pptx",
                                "title": deck.title,
                                "slide_count": len(deck.slides),
                            },
                        )
                        result_output["download_url"] = storage_result["url"]
                        result_output["storage_key"] = storage_result["storage_key"]

                        logger.info(
                            "slide_generation_saved",
                            storage_key=storage_result["storage_key"],
                            user_id=user_id,
                        )
                    except Exception as e:
                        logger.warning("slide_generation_storage_failed", error=str(e))
                else:
                    logger.warning("slide_generation_no_user_id")

                logger.info(
                    "slide_generation_skill_completed",
                    title=deck.title,
                    slide_count=len(deck.slides),
                )

                return {
                    "output": result_output,
                    "pending_events": events,
                    "iterations": state.get("iterations", 0) + 1,
                }

            except Exception as e:
                logger.error("slide_generation_failed", error=str(e))
                return {
                    "error": f"PPTX generation failed: {e}",
                    "pending_events": events,
                    "iterations": state.get("iterations", 0) + 1,
                }

        # ------------------------------------------------------------------
        # Build graph: research -> outline -> images -> generate -> END
        # ------------------------------------------------------------------

        graph.add_node("research", research_node)
        graph.add_node("outline", outline_node)
        graph.add_node("images", images_node)
        graph.add_node("generate", generate_node)

        graph.set_entry_point("research")
        graph.add_edge("research", "outline")
        graph.add_edge("outline", "images")
        graph.add_edge("images", "generate")
        graph.add_edge("generate", END)

        return graph.compile()
