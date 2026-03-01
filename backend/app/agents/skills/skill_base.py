"""Base classes and types for the skills system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


class SkillParameter(BaseModel):
    """Parameter definition for skill input."""

    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Any = None


class SkillMetadata(BaseModel):
    """Metadata for a skill."""

    id: str  # e.g., "web_research"
    name: str
    version: str = "1.0.0"
    description: str
    category: str  # "research", "data", "creative", "automation", "code"
    parameters: list[SkillParameter]
    output_schema: dict[str, Any]  # JSON schema
    required_tools: list[str] = Field(default_factory=list)
    # Explicit risk metadata for governance ("low"|"medium"|"high")
    risk_level: Literal["low", "medium", "high"]
    side_effect_level: Literal["none", "low", "medium", "high"] = "low"
    data_sensitivity: Literal["public", "internal", "sensitive"] = "internal"
    network_scope: Literal["none", "sandbox_only", "external"] = "none"
    idempotency_hint: bool = True
    requires_confirmation_patterns: list[str] = Field(default_factory=list)
    max_execution_time_seconds: int = 300
    max_iterations: int = 10
    author: str = "hyperagent"
    tags: list[str] = Field(default_factory=list)
    enabled: bool = True


class SkillState(TypedDict, total=False):
    """Base state for skill execution."""

    skill_id: str
    input_params: dict[str, Any]
    output: dict[str, Any]
    error: str | None
    events: list[dict[str, Any]]
    iterations: int
    # Execution context - allows skills to share sessions with agents
    user_id: str | None
    task_id: str | None
    invocation_depth: int


@dataclass
class SkillContext:
    """Execution context passed to ToolSkill.execute().

    Provides access to skill metadata (user/task IDs) and enables
    skill composition by allowing one skill to invoke another.
    """

    skill_id: str
    user_id: str | None = None
    task_id: str | None = None
    invocation_depth: int = 0
    max_depth: int = 3

    async def invoke_skill(self, skill_id: str, params: dict) -> dict:
        """Invoke another skill from within a skill (composition).

        Args:
            skill_id: ID of the skill to invoke
            params: Input parameters for the skill

        Returns:
            Skill output dictionary

        Raises:
            RecursionError: If max invocation depth is exceeded
        """
        if self.invocation_depth >= self.max_depth:
            raise RecursionError(
                f"Skill invocation depth limit ({self.max_depth}) exceeded. "
                f"Current depth: {self.invocation_depth}, attempting to invoke: {skill_id}"
            )

        # Import here to avoid circular imports
        from app.services.skill_executor import skill_executor
        from app.services.skill_registry import skill_registry

        skill = skill_registry.get_skill(skill_id)
        if not skill:
            raise ValueError(f"Skill not found: {skill_id}")

        is_valid, error_msg = skill.validate_input(params)
        if not is_valid:
            raise ValueError(f"Invalid input for skill {skill_id}: {error_msg}")

        # Collect output from the nested execution
        output: dict = {}
        async for event in skill_executor.execute_skill(
            skill_id=skill_id,
            params=params,
            user_id=self.user_id or "",
            agent_type="skill",
            task_id=self.task_id,
            invocation_depth=self.invocation_depth + 1,
        ):
            if event.get("type") == "skill_output":
                output = event.get("output", {})

        return output


class Skill:
    """Base class for all skills."""

    metadata: SkillMetadata

    def create_graph(self) -> StateGraph:
        """Create the LangGraph subgraph for this skill.

        Returns:
            Compiled StateGraph ready for execution
        """
        raise NotImplementedError("Subclasses must implement create_graph()")

    def validate_input(self, params: dict[str, Any]) -> tuple[bool, str]:
        """Validate input parameters against schema.

        Args:
            params: Input parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check all required parameters are present
        for param in self.metadata.parameters:
            if param.required and param.name not in params:
                return False, f"Missing required parameter: {param.name}"

        # Check parameter types
        for param in self.metadata.parameters:
            if param.name not in params:
                continue

            value = params[param.name]
            expected_type = param.type

            # Basic type validation
            type_checks = {
                "string": lambda v: isinstance(v, str),
                "number": lambda v: isinstance(v, (int, float)),
                "boolean": lambda v: isinstance(v, bool),
                "object": lambda v: isinstance(v, dict),
                "array": lambda v: isinstance(v, list),
            }

            if expected_type in type_checks and not type_checks[expected_type](value):
                return False, f"Parameter '{param.name}' must be of type {expected_type}"

        return True, ""


class ToolSkill(Skill):
    """Base for single-step skills that don't need full graph boilerplate.

    Override ``execute()`` instead of ``create_graph()``.  The parent
    ``create_graph()`` auto-generates a one-node StateGraph that delegates
    to ``execute()``, so ToolSkill instances remain fully compatible with
    the executor (which always calls ``create_graph()``).
    """

    async def execute(self, params: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        """Execute the skill logic.

        Args:
            params: Validated input parameters (same as ``state["input_params"]``)
            context: Execution context with user/task IDs and composition helpers

        Returns:
            Output dictionary (becomes ``state["output"]``)
        """
        raise NotImplementedError("ToolSkill subclasses must implement execute()")

    def create_graph(self) -> StateGraph:
        """Auto-generate a single-node graph wrapping ``execute()``."""
        skill_ref = self  # capture for closure

        async def _execute_node(state: SkillState) -> dict:
            context = SkillContext(
                skill_id=state.get("skill_id", skill_ref.metadata.id),
                user_id=state.get("user_id"),
                task_id=state.get("task_id"),
                invocation_depth=state.get("invocation_depth", 0),
            )
            try:
                output = await skill_ref.execute(state["input_params"], context)
                return {
                    "output": output,
                    "iterations": state.get("iterations", 0) + 1,
                }
            except Exception as e:
                return {
                    "error": f"{skill_ref.metadata.name} failed: {e}",
                    "iterations": state.get("iterations", 0) + 1,
                }

        graph = StateGraph(SkillState)
        graph.add_node("execute", _execute_node)
        graph.set_entry_point("execute")
        graph.add_edge("execute", END)
        return graph.compile()
