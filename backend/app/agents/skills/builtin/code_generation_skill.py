"""Code Generation Skill for creating code snippets and functions."""

from typing import Any

from pydantic import BaseModel, Field

from app.agents.skills.skill_base import SkillContext, SkillMetadata, SkillParameter, ToolSkill
from app.ai.llm import llm_service
from app.ai.model_tiers import ModelTier
from app.core.logging import get_logger

logger = get_logger(__name__)


class CodeGenerationResponse(BaseModel):
    """Structured response for code generation."""

    code: str = Field(description="The generated code snippet or program")
    explanation: str = Field(
        description="Brief explanation of implementation details and design choices"
    )
    tests: str | None = Field(default=None, description="Unit tests for the code, if requested")
    language: str = Field(description="The programming language used (e.g., python, typescript)")


class CodeGenerationSkill(ToolSkill):
    """Generates code snippets, functions, and small programs."""

    metadata = SkillMetadata(
        id="code_generation",
        name="Code Generation",
        version="2.0.0",
        description=(
            "Generates well-structured code snippets, functions, classes, "
            "or small programs in any programming language"
        ),
        category="code",
        parameters=[
            SkillParameter(
                name="task",
                type="string",
                description=(
                    "Description of what code to generate "
                    "(e.g., 'function to validate email', "
                    "'API endpoint for user login')"
                ),
                required=True,
            ),
            SkillParameter(
                name="language",
                type="string",
                description=(
                    "Programming language (python, javascript, typescript, java, go, rust, etc.)"
                ),
                required=False,
                default="python",
            ),
            SkillParameter(
                name="style",
                type="string",
                description="Code style: clean, documented, minimal, production-ready",
                required=False,
                default="clean",
            ),
            SkillParameter(
                name="include_tests",
                type="boolean",
                description="Whether to include unit tests",
                required=False,
                default=False,
            ),
            SkillParameter(
                name="context",
                type="string",
                description="Additional context or requirements",
                required=False,
                default="",
            ),
        ],
        output_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The generated code",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language used",
                },
                "explanation": {
                    "type": "string",
                    "description": "Explanation of the code",
                },
                "tests": {
                    "type": "string",
                    "description": "Unit tests if requested",
                },
            },
        },
        required_tools=[],
        risk_level="low",
        side_effect_level="none",
        data_sensitivity="internal",
        network_scope="none",
        idempotency_hint=True,
        max_iterations=2,
        tags=["code", "programming", "generation", "development"],
    )

    async def execute(self, params: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        """Generate code based on requirements."""
        task = params["task"]
        language = params.get("language", "python")
        style = params.get("style", "clean")
        include_tests = params.get("include_tests", False)
        extra_context = params.get("context", "")

        logger.info(
            "code_generation_skill_generating",
            language=language,
            style=style,
            include_tests=include_tests,
        )

        # Build style guidance
        style_guidance = {
            "clean": "Write clean, readable code with meaningful variable names",
            "documented": "Include comprehensive docstrings and inline comments",
            "minimal": "Write concise, minimalist code without unnecessary complexity",
            "production-ready": (
                "Include error handling, type hints, and production-quality patterns"
            ),
        }
        style_desc = style_guidance.get(style, style_guidance["clean"])

        context_section = f"\n\nAdditional Context:\n{extra_context}" if extra_context else ""
        tests_section = (
            "\n\nAlso provide unit tests for the code in the 'tests' field."
            if include_tests
            else ""
        )

        prompt = f"""Generate {language} code for the following task:

Task: {task}

Requirements:
- Language: {language}
- Style: {style_desc}
- Follow best practices for {language}{tests_section}{context_section}"""

        # Get LLM for code generation
        llm = llm_service.get_llm_for_tier(ModelTier.MAX)

        # Use structured output
        structured_llm = llm.with_structured_output(CodeGenerationResponse)

        # Generate code
        result: CodeGenerationResponse = await structured_llm.ainvoke(prompt)

        # Validate syntax for generated code (warn but don't block)
        if result.language == "python" and result.code:
            try:
                compile(result.code, "<generated>", "exec")
            except SyntaxError as syn_err:
                logger.warning(
                    "code_generation_syntax_error",
                    language=result.language,
                    line=syn_err.lineno,
                    error=str(syn_err),
                )

        logger.info(
            "code_generation_skill_completed",
            language=result.language,
            code_length=len(result.code),
            has_tests=bool(result.tests),
        )

        return {
            "code": result.code,
            "language": result.language,
            "explanation": result.explanation,
            "tests": result.tests if include_tests else None,
        }
