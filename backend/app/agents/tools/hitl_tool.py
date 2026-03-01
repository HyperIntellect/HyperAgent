"""Human-in-the-Loop tool for agent-initiated user interactions.

This tool allows agents to pause and ask users for:
- Decisions (multiple choice)
- Input (free-form text)
- Confirmation (yes/no)

The agent calls this tool when it needs user guidance to proceed.
"""

from typing import Literal

from langchain_core.tools import tool

from app.agents.hitl.interrupt_manager import (
    create_decision_interrupt,
    create_input_interrupt,
    get_interrupt_manager,
)
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@tool
async def ask_user(
    question: str,
    question_type: Literal["decision", "input", "confirmation"] = "input",
    options: list[dict[str, str]] | None = None,
    context: str | None = None,
    _thread_id: str | None = None,
    _timeout: int | None = None,
) -> str:
    """Ask the user a question and wait for their response.

    Use this tool when you need user input to proceed with a task.
    Common scenarios:
    - Clarifying ambiguous requests
    - Choosing between multiple valid approaches
    - Confirming before making significant changes
    - Gathering additional information

    Args:
        question: The question to ask the user. Be clear and specific.
        question_type: Type of question:
            - "decision": Present multiple choice options (requires options param)
            - "input": Request free-form text input
            - "confirmation": Ask yes/no question (auto-generates Yes/No options)
        options: For "decision" type, list of options like:
            [{"label": "Option A", "value": "a", "description": "Details about A"}]
            Each option needs "label" and "value", "description" is optional.
        context: Additional context to help user understand the question.
        _thread_id: Internal - thread ID for interrupt management.
        _timeout: Internal - timeout in seconds.

    Returns:
        The user's response:
        - For "decision": The selected option's value
        - For "input": The text entered by user
        - For "confirmation": "yes" or "no"
        - If user skips: "skipped"
        - If timeout: "timeout"

    Example usage:
        # Ask for clarification
        response = await ask_user(
            question="Which programming language should I use for this script?",
            question_type="decision",
            options=[
                {"label": "Python", "value": "python", "description": "Good for data processing"},
                {"label": "JavaScript", "value": "javascript", "description": "Good for web tasks"},
                {"label": "Bash", "value": "bash", "description": "Good for system tasks"}
            ]
        )

        # Get user input
        response = await ask_user(
            question="What file name would you like for the output?",
            question_type="input"
        )

        # Confirm action
        response = await ask_user(
            question="This will delete all temporary files. Continue?",
            question_type="confirmation"
        )
    """
    interrupt_manager = get_interrupt_manager()
    thread_id = _thread_id or "default"
    timeout = _timeout or settings.hitl_decision_timeout

    # Build the full message with context
    message = question
    if context:
        message = f"{context}\n\n{question}"

    # Create the appropriate interrupt
    if question_type == "confirmation":
        from app.agents.hitl.interrupt_manager import create_confirm_interrupt

        interrupt_event = create_confirm_interrupt(
            title="Confirmation",
            message=message,
            timeout_seconds=timeout,
        )
    elif question_type == "decision":
        if not options:
            return "Error: decision type requires options"
        interrupt_event = create_decision_interrupt(
            title="Agent Question",
            message=message,
            options=options,
            timeout_seconds=timeout,
        )
    else:  # input
        interrupt_event = create_input_interrupt(
            title="Agent Question",
            message=message,
            timeout_seconds=timeout,
        )

    interrupt_id = interrupt_event["interrupt_id"]

    logger.info(
        "hitl_ask_user",
        question_type=question_type,
        interrupt_id=interrupt_id,
        thread_id=thread_id,
    )

    try:
        # Store interrupt
        await interrupt_manager.create_interrupt(
            thread_id=thread_id,
            interrupt_id=interrupt_id,
            interrupt_data=interrupt_event,
        )

        # Wait for user response
        response = await interrupt_manager.wait_for_response(
            thread_id=thread_id,
            interrupt_id=interrupt_id,
            timeout_seconds=timeout,
        )

        action = response.get("action", "skip")
        value = response.get("value")

        if action == "skip":
            logger.info("hitl_user_skipped", interrupt_id=interrupt_id)
            return "skipped"
        elif action in ("select", "input"):
            logger.info(
                "hitl_user_responded",
                interrupt_id=interrupt_id,
                action=action,
                value=value[:50] if value else None,
            )
            return value or "skipped"
        elif action in ("approve", "deny"):
            logger.info(
                "hitl_user_confirmed",
                interrupt_id=interrupt_id,
                action=action,
                value=value,
            )
            return value or ("yes" if action == "approve" else "no")
        else:
            return f"unknown_action:{action}"

    except TimeoutError:
        logger.warning("hitl_ask_user_timeout", interrupt_id=interrupt_id)
        return "timeout"
    except Exception as e:
        logger.error("hitl_ask_user_error", error=str(e), interrupt_id=interrupt_id)
        return f"error: {str(e)}"


# Export the tool
ask_user_tool = ask_user
