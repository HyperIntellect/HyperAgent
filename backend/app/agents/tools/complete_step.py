"""Tool for explicitly marking plan steps as completed.

The LLM calls this tool after finishing each step in a planned execution,
decoupling step tracking from tool-call iterations. A single plan step
may require multiple tool calls — the LLM signals completion when ready.
"""

from langchain_core.tools import tool

from app.core.logging import get_logger

logger = get_logger(__name__)


@tool
def complete_step(
    step_number: int,
    result_summary: str,
) -> str:
    """Mark a plan step as completed.

    Call this tool AFTER you have fully finished a step in the execution plan.
    Do NOT call it prematurely — only when all work for the step is done.

    Args:
        step_number: The 1-based step number that was completed
            (must match the current step in the plan).
        result_summary: A brief summary of what was accomplished in this step.

    Returns:
        Confirmation message. The system will automatically advance
        the plan to the next step and update progress tracking.
    """
    logger.info(
        "complete_step_tool_called",
        step_number=step_number,
        summary_length=len(result_summary),
    )
    return (
        f"Step {step_number} marked as completed. "
        f"Summary: {result_summary[:200]}. "
        f"The system will advance to the next step."
    )


complete_step_tool = complete_step
