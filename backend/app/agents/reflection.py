"""Reflection module: lightweight quality gate for the executor's ReAct loop.

Provides two complementary layers:
1. Heuristic gate (should_reflect) — decides whether to invoke the LLM at all
2. LLM-based reflection (reflect) — LITE-tier call that evaluates answer quality
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, ToolMessage

from app.agents.react_helpers import DEDICATED_SKILL_MODE_NAMES, extract_last_ai_response
from app.ai.llm import extract_text_from_content, llm_service
from app.ai.model_tiers import ModelTier
from app.config import settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.agents.state import ExecutorState

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

REFLECTION_PROMPT = """\
You are a quality reviewer. Evaluate whether the assistant's \
response adequately addresses the step goal.

<step_goal>
{step_goal}
</step_goal>

<assistant_response>
{response}
</assistant_response>

<tool_results_summary>
{tool_summary}
</tool_results_summary>

Evaluate:
1. Does the response address the step goal?
2. Is the response complete and accurate based on the tool results?
3. Are there obvious gaps or missing information?

Respond with EXACTLY one of these verdicts on the first line, \
followed by a brief explanation:
- PASS — the response is adequate
- RETRY — the response is incomplete or off-topic, try again
- COMPLETE_WITH_NOTE — acceptable but has minor gaps

After the verdict line, provide a confidence score (0.0-1.0) \
on the second line.
Then provide a brief quality note on the third line."""

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReflectionResult:
    """Result of the reflection evaluation."""

    verdict: str  # "pass" | "retry" | "complete_with_note"
    confidence: float
    note: str


# ---------------------------------------------------------------------------
# Heuristic gate
# ---------------------------------------------------------------------------


def should_reflect(state: ExecutorState) -> bool:
    """Decide whether to invoke the LLM reflection gate.

    Returns True only when all conditions are met:
    - reflection is enabled in config
    - tools were actually used (min iterations threshold)
    - we haven't already reflected the max number of times
    - not a dedicated mode (app/image/slide bypass)
    """
    if not settings.reflection_enabled:
        return False

    tool_iterations = state.get("tool_iterations", 0)
    if tool_iterations < settings.reflection_min_tool_iterations:
        return False

    reflection_count = state.get("reflection_count", 0)
    if reflection_count >= settings.reflection_max_count:
        return False

    mode = state.get("mode")
    if mode in DEDICATED_SKILL_MODE_NAMES:
        return False

    return True


# ---------------------------------------------------------------------------
# LLM-based reflection
# ---------------------------------------------------------------------------


def _summarize_tool_results(state: ExecutorState) -> str:
    """Extract a brief summary of tool results from messages."""
    lc_messages = state.get("lc_messages", [])
    summaries: list[str] = []
    for msg in lc_messages:
        if isinstance(msg, ToolMessage):
            content = str(msg.content)[:300]
            summaries.append(f"- {msg.name}: {content}")
    if not summaries:
        return "No tool results."
    return "\n".join(summaries[-5:])


def _extract_last_response(state: ExecutorState) -> str:
    """Extract the last assistant response text."""
    return extract_last_ai_response(state.get("lc_messages", []))


def _parse_reflection_output(raw: str) -> ReflectionResult:
    """Parse the LLM reflection output into a ReflectionResult.

    Expected format:
        PASS|RETRY|COMPLETE_WITH_NOTE
        0.85
        Brief quality note here
    """
    lines = raw.strip().splitlines()
    verdict_line = lines[0].strip().upper() if lines else "PASS"

    if "RETRY" in verdict_line:
        verdict = "retry"
    elif "COMPLETE_WITH_NOTE" in verdict_line:
        verdict = "complete_with_note"
    else:
        verdict = "pass"

    confidence = 0.8
    if len(lines) > 1:
        try:
            confidence = float(lines[1].strip())
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            pass

    note = (
        lines[2].strip() if len(lines) > 2 else "No additional notes."
    )

    return ReflectionResult(
        verdict=verdict, confidence=confidence, note=note,
    )


async def reflect(state: ExecutorState) -> ReflectionResult:
    """Run LLM-based reflection on the executor's output.

    Uses LITE tier for cost efficiency. Falls back to "pass" on
    any error so reflection never blocks execution.
    """
    step_goal = state.get("step_goal") or "Complete the user's request"
    response_text = _extract_last_response(state)
    tool_summary = _summarize_tool_results(state)

    prompt = REFLECTION_PROMPT.format(
        step_goal=step_goal,
        response=response_text[:2000],
        tool_summary=tool_summary[:1000],
    )

    try:
        tier = ModelTier(settings.reflection_model_tier)
        provider = state.get("provider")
        llm = llm_service.get_llm_for_tier(
            tier=tier, provider=provider,
        )

        result = await llm.ainvoke(prompt)
        raw_text = extract_text_from_content(result.content)
        return _parse_reflection_output(raw_text)

    except Exception as e:
        logger.warning(
            "reflection_failed_defaulting_to_pass", error=str(e),
        )
        return ReflectionResult(
            verdict="pass",
            confidence=0.5,
            note=f"Reflection error: {e}",
        )
