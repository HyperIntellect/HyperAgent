"""PlannerAgent: Decomposes complex queries into executable plan steps.

This subgraph takes a user query and conversation context, then uses an LLM
to produce a structured list of PlanStep objects. It can also revise plans
when given revision_context from failed steps.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.agents import events
from app.agents.state import PlannerState, PlanStep
from app.ai.llm import extract_text_from_content, llm_service
from app.core.logging import get_logger

logger = get_logger(__name__)

PLANNER_PROMPT = """You are a task planner. Decompose the user's request into a sequence of concrete, executable steps.

For each step, provide:
- step_number: Sequential number starting from 1
- goal: Clear, specific description of what this step should accomplish
- tools_hint: List of tool/skill names that would be useful (from: web_search, execute_code, app_write_file, app_install_packages, app_start_server, browser_navigate, generate_image, invoke_skill, sandbox_file, generate_slides)
- expected_output: What success looks like for this step

Rules:
- Each step should be independently executable
- Steps should be ordered by dependency (prerequisites first)
- Keep steps focused — one clear goal per step
- Typically 2-6 steps for most tasks
- Don't create steps for things the LLM can answer directly

Respond with ONLY a JSON array of step objects. No other text.

Example:
```json
[
    {"step_number": 1, "goal": "Create the backend API with FastAPI", "tools_hint": ["execute_code", "app_write_file"], "expected_output": "FastAPI app.py with endpoints"},
    {"step_number": 2, "goal": "Build the React frontend", "tools_hint": ["app_write_file", "app_install_packages"], "expected_output": "React components rendering"},
    {"step_number": 3, "goal": "Start and test the application", "tools_hint": ["app_start_server"], "expected_output": "App running and accessible"}
]
```"""

REVISION_PROMPT_TEMPLATE = """The previous plan failed. Here is what happened:

{revision_context}

Create a revised plan that works around the failures. You may:
- Skip failed steps if they're not essential
- Use alternative approaches
- Break complex steps into simpler sub-steps
- Remove steps that are no longer needed

Respond with ONLY a JSON array of step objects."""


def _parse_plan_steps(content: str) -> list[PlanStep]:
    """Parse LLM response into structured PlanStep list.

    Handles both raw JSON arrays and markdown-wrapped JSON blocks.

    Args:
        content: Raw LLM response text containing a JSON array of steps.

    Returns:
        List of PlanStep dicts.

    Raises:
        ValueError: If content is not a valid JSON array or is empty.
    """
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        raise ValueError("Expected JSON array of steps")
    if not parsed:
        raise ValueError("Empty plan")

    steps: list[PlanStep] = []
    for i, item in enumerate(parsed):
        steps.append(
            PlanStep(
                step_number=item.get("step_number", i + 1),
                goal=item.get("goal", f"Step {i + 1}"),
                tools_hint=item.get("tools_hint", []),
                expected_output=item.get("expected_output", ""),
            )
        )
    return steps


async def planner_node(state: PlannerState) -> dict:
    """Decompose query into plan steps using LLM.

    Args:
        state: PlannerState with at least query and provider fields.

    Returns:
        Dict with plan_steps, complexity_assessment, and events.
    """
    query = state.get("query", "")
    provider = state.get("provider")
    tier = state.get("tier")
    revision_context = state.get("revision_context")
    event_list: list[dict] = []

    messages = [SystemMessage(content=PLANNER_PROMPT)]
    if revision_context:
        messages.append(
            HumanMessage(
                content=REVISION_PROMPT_TEMPLATE.format(
                    revision_context=revision_context,
                )
            )
        )
        messages.append(HumanMessage(content=f"Original request: {query}"))
    else:
        messages.append(HumanMessage(content=query))

    llm = llm_service.choose_llm_for_task(
        task_type="task",
        provider=provider,
        tier_override=tier or "pro",
    )

    try:
        response = await llm.ainvoke(messages)
        content = extract_text_from_content(response.content)
        plan_steps = _parse_plan_steps(content)

        logger.info(
            "planner_produced_plan",
            step_count=len(plan_steps),
            is_revision=bool(revision_context),
        )
        event_list.append(
            events.reasoning(
                thinking=f"Decomposed task into {len(plan_steps)} steps",
                confidence=0.85,
                context="planning",
            )
        )
        return {
            "plan_steps": plan_steps,
            "complexity_assessment": f"{len(plan_steps)} steps planned",
            "events": event_list,
        }
    except Exception as e:
        logger.error("planner_failed", error=str(e))
        fallback_step = PlanStep(
            step_number=1,
            goal=query,
            tools_hint=[],
            expected_output="Task completed",
        )
        event_list.append(
            events.reasoning(
                thinking=f"Planning failed ({e}), falling back to single-step execution",
                confidence=0.3,
                context="planning",
            )
        )
        return {
            "plan_steps": [fallback_step],
            "complexity_assessment": "fallback_single_step",
            "events": event_list,
        }


def create_planner_graph():
    """Create the PlannerAgent subgraph.

    Returns:
        Compiled LangGraph StateGraph with a single 'plan' node.
    """
    graph = StateGraph(PlannerState)
    graph.add_node("plan", planner_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", END)
    return graph.compile()


planner_subgraph = create_planner_graph()
