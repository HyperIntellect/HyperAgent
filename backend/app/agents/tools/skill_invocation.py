"""Skill invocation tools for LangGraph agents."""

import json
from typing import Any

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from app.core.logging import get_logger
from app.services.skill_executor import skill_executor
from app.services.skill_registry import skill_registry

logger = get_logger(__name__)


def _remove_internal_fields(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove user_id and task_id from JSON schema so the LLM cannot see or set them."""
    props = schema.get("properties", {})
    props.pop("user_id", None)
    props.pop("task_id", None)
    props.pop("user_intent_source", None)
    required = schema.get("required", [])
    schema["required"] = [
        r for r in required if r not in ("user_id", "task_id", "user_intent_source")
    ]
    return schema


class InvokeSkillInput(BaseModel):
    """Input schema for skill invocation tool."""

    model_config = ConfigDict(json_schema_extra=_remove_internal_fields)

    skill_id: str = Field(description="The ID of the skill to invoke (e.g., 'web_research')")
    params: dict[str, Any] = Field(description="Input parameters required by the skill")
    # Context fields (injected by agent, not provided by LLM)
    user_id: str | None = Field(
        default=None,
        description="User ID for session management (internal use only)",
        json_schema_extra={"exclude": True},
    )
    task_id: str | None = Field(
        default=None,
        description="Task ID for session management (internal use only)",
        json_schema_extra={"exclude": True},
    )
    user_intent_source: str | None = Field(
        default=None,
        description="How this skill invocation was selected (internal use only)",
        json_schema_extra={"exclude": True},
    )


@tool(args_schema=InvokeSkillInput)
async def invoke_skill(
    skill_id: str,
    params: dict[str, Any],
    user_id: str | None = None,
    task_id: str | None = None,
    user_intent_source: str | None = None,
) -> str:
    """Invoke a registered skill to perform a specialized task.

    Skills are composable subgraphs that provide focused capabilities like
    web research, code review, data visualization, and more. Each skill has
    its own input requirements and output format.

    To see available skills and their parameters, list them first or check
    the skill registry.

    Args:
        skill_id: The ID of the skill (e.g., 'web_research', 'code_generation')
        params: Input parameters required by the skill (varies per skill)

    Returns:
        JSON string with skill execution results
    """
    logger.info(
        "invoke_skill_tool_called",
        skill_id=skill_id,
        params_keys=list(params.keys()) if params else [],
        user_id=user_id,
        task_id=task_id,
        user_intent_source=user_intent_source,
        user_id_is_none=user_id is None,
    )

    # Get skill from registry (auto-promotes to Level 2 if needed)
    skill = skill_registry.get_skill(skill_id)
    if not skill:
        available = [s.id for s in skill_registry.list_skills()]
        error_msg = f"Skill '{skill_id}' not found. Available skills: {available}"
        logger.warning("skill_not_found", skill_id=skill_id)
        return json.dumps(
            {
                "error": error_msg,
                "skill_id": skill_id,
            }
        )

    # Validate input
    is_valid, error_msg = skill.validate_input(params)
    if not is_valid:
        logger.warning("skill_input_invalid", skill_id=skill_id, error=error_msg)
        return json.dumps(
            {
                "error": f"Invalid parameters for skill '{skill_id}': {error_msg}",
                "skill_id": skill_id,
                "expected_parameters": [p.model_dump() for p in skill.metadata.parameters],
            }
        )

    # Validate required tools are available
    if skill.metadata.required_tools:
        from app.agents.tools.registry import get_all_tools

        known = {t.name for t in get_all_tools()}
        missing = [t for t in skill.metadata.required_tools if t not in known]
        if missing:
            logger.warning(
                "skill_required_tools_missing",
                skill_id=skill_id,
                missing_tools=missing,
            )
            return json.dumps(
                {
                    "error": f"Skill '{skill_id}' requires unavailable tools: {missing}",
                    "skill_id": skill_id,
                }
            )

    try:
        output = None
        error = None
        effective_user_id = user_id or "anonymous"

        async for event in skill_executor.execute_skill(
            skill_id=skill_id,
            params=params,
            user_id=effective_user_id,
            agent_type="tool",
            task_id=task_id,
        ):
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "skill_output":
                output = event.get("output", {})
            elif event_type == "error":
                error = event.get("error")
            elif event_type in (
                "stage",
                "terminal_command",
                "terminal_output",
                "terminal_error",
                "terminal_complete",
                "browser_stream",
                "workspace_update",
                "image",
            ):
                # Dispatch in real-time via LangChain callback system
                # so the stream processor can yield them immediately
                try:
                    await adispatch_custom_event("skill_event", data=event)
                except Exception as e:
                    logger.warning(
                        "skill_event_dispatch_failed",
                        skill_id=skill_id,
                        event_type=event_type,
                        error=str(e),
                    )

        if error:
            logger.error("skill_execution_error", skill_id=skill_id, error=error)
            return json.dumps(
                {
                    "error": error,
                    "skill_id": skill_id,
                }
            )

        if output is None:
            output = {}

        logger.debug(
            "skill_executed_successfully",
            skill_id=skill_id,
            output_keys=list(output.keys()) if isinstance(output, dict) else None,
        )

        return json.dumps(
            {
                "skill_id": skill_id,
                "output": output,
                "success": True,
            },
            indent=2,
        )

    except Exception as e:
        error_msg = str(e)
        logger.error("skill_execution_failed", skill_id=skill_id, error=error_msg)
        return json.dumps(
            {
                "error": f"Skill execution failed: {error_msg}",
                "skill_id": skill_id,
            }
        )


class ListSkillsInput(BaseModel):
    """Input schema for listing skills."""

    category: str | None = Field(
        default=None,
        description="Optional category filter (research, code, data, creative, automation)",
    )
    # Note: list_skills does not need user_id/task_id since it just lists
    # available skills without any session-specific context


@tool(args_schema=ListSkillsInput)
async def list_skills(category: str | None = None) -> str:
    """List all available skills that can be invoked.

    Use this tool to discover what skills are available and what they do.
    Each skill has a specific purpose and required parameters.

    Args:
        category: Optional category filter (e.g., 'research', 'code', 'data')

    Returns:
        JSON string with available skills and their metadata
    """
    logger.debug("list_skills_tool_called", category=category)

    skills = skill_registry.list_skills(category=category)

    if not skills:
        return json.dumps(
            {
                "message": "No skills found" + (f" in category '{category}'" if category else ""),
                "skills": [],
            }
        )

    skills_data = []
    for skill_metadata in skills:
        skills_data.append(
            {
                "id": skill_metadata.id,
                "name": skill_metadata.name,
                "description": skill_metadata.description,
                "category": skill_metadata.category,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "description": p.description,
                        "required": p.required,
                        "default": p.default,
                    }
                    for p in skill_metadata.parameters
                ],
                "tags": skill_metadata.tags,
            }
        )

    return json.dumps(
        {
            "skills": skills_data,
            "count": len(skills_data),
            "category": category,
        },
        indent=2,
    )


def get_skill_tools() -> list:
    """Get all skill-related tools for the registry.

    Returns:
        List of skill tools to register
    """
    return [invoke_skill, list_skills]
