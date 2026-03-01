"""Tool Result Validators.

This module provides validation for tool outputs to ensure they match
expected schemas before processing. This helps catch malformed tool
responses early and provides better error messages.
"""

import json
from typing import Any, TypedDict

from pydantic import BaseModel, Field, ValidationError

from app.core.logging import get_logger

logger = get_logger(__name__)


class ValidationResult(TypedDict):
    """Result of tool output validation."""

    valid: bool
    errors: list[str]
    data: Any | None


# Pydantic models for expected tool outputs


class SearchResultItem(BaseModel):
    """Schema for individual search result."""

    title: str = Field(default="")
    url: str = Field(default="")
    snippet: str = Field(default="")
    content: str | None = Field(default=None)
    relevance_score: float | None = Field(default=None)


class WebSearchOutput(BaseModel):
    """Expected output schema for web_search tool."""

    formatted: str = Field(default="")
    results: list[SearchResultItem] = Field(default_factory=list)
    query: str = Field(default="")
    error: str | None = Field(default=None)


class WebExtractStructuredOutput(BaseModel):
    """Expected output schema for web_extract_structured tool."""

    success: bool = Field(default=True)
    query: str = Field(default="")
    url: str | None = Field(default=None)
    fields: list[str] = Field(default_factory=list)
    count: int = Field(default=0)
    items: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = Field(default=None)


class ImageGenerationOutput(BaseModel):
    """Expected output schema for generate_image tool."""

    success: bool = Field(default=False)
    images: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = Field(default=None)
    prompt: str | None = Field(default=None)


class ImageAnalysisOutput(BaseModel):
    """Expected output schema for analyze_image tool."""

    description: str = Field(default="")
    objects: list[str] = Field(default_factory=list)
    text: str | None = Field(default=None)
    error: str | None = Field(default=None)


class CodeExecutionOutput(BaseModel):
    """Expected output schema for code execution tools."""

    success: bool = Field(default=False)
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    exit_code: int | None = Field(default=None)
    images: list[dict[str, str]] = Field(default_factory=list)
    error: str | None = Field(default=None)


class HandoffOutput(BaseModel):
    """Expected output schema for handoff tools."""

    handoff: bool = Field(default=True)
    target_agent: str = Field(...)
    task_description: str = Field(...)
    context: str = Field(default="")
    source_agent: str = Field(...)


class FileOperationOutput(BaseModel):
    """Expected output schema for sandbox file operations."""

    success: bool = Field(default=False)
    operation: str = Field(default="")
    path: str = Field(default="")
    content: str | None = Field(default=None)
    is_binary: bool = Field(default=False)
    entries: list[dict] | None = Field(default=None)
    exists: bool | None = Field(default=None)
    bytes_written: int | None = Field(default=None)
    error: str | None = Field(default=None)


# Tool name to validator mapping
TOOL_VALIDATORS: dict[str, type[BaseModel]] = {
    "web_search": WebSearchOutput,
    "web_extract_structured": WebExtractStructuredOutput,
    "generate_image": ImageGenerationOutput,
    "analyze_image": ImageAnalysisOutput,
    "execute_code": CodeExecutionOutput,
    "execute_python": CodeExecutionOutput,
    "sandbox_file": FileOperationOutput,
    # Handoff tools are dynamically named (handoff_to_*)
}


def validate_tool_output(
    tool_name: str,
    output: str | dict | Any,
    strict: bool = False,
) -> ValidationResult:
    """Validate tool output against expected schema.

    Attempts to parse and validate tool output. In non-strict mode,
    validation errors are logged but the original data is returned.
    In strict mode, validation errors result in None data.

    Args:
        tool_name: Name of the tool that produced the output
        output: Tool output (JSON string or dict)
        strict: If True, return None data on validation failure

    Returns:
        ValidationResult with valid flag, errors list, and parsed data
    """
    errors: list[str] = []
    data: Any = None

    # Handle handoff tools (dynamically named)
    if tool_name.startswith("handoff_to_"):
        validator = HandoffOutput
    else:
        validator = TOOL_VALIDATORS.get(tool_name)

    # No validator registered for this tool
    if validator is None:
        logger.debug("no_validator_for_tool", tool=tool_name)
        # Return the raw output as data
        if isinstance(output, str):
            try:
                data = json.loads(output)
            except json.JSONDecodeError:
                data = output
        else:
            data = output
        return ValidationResult(valid=True, errors=[], data=data)

    # Parse JSON string if needed
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON output: {e}"
            errors.append(error_msg)
            logger.warning(
                "tool_output_json_error",
                tool=tool_name,
                error=str(e),
                output_preview=output[:200] if output else "empty",
            )
            if strict:
                return ValidationResult(valid=False, errors=errors, data=None)
            return ValidationResult(valid=False, errors=errors, data=output)
    else:
        parsed = output

    # Validate against schema
    try:
        validated = validator.model_validate(parsed)
        data = validated.model_dump()
        return ValidationResult(valid=True, errors=[], data=data)

    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"{loc}: {err['msg']}")

        logger.warning(
            "tool_output_validation_failed",
            tool=tool_name,
            errors=errors,
        )

        if strict:
            return ValidationResult(valid=False, errors=errors, data=None)

        # In non-strict mode, return parsed data even if validation failed
        return ValidationResult(valid=False, errors=errors, data=parsed)


def validate_search_results(output: str | dict) -> ValidationResult:
    """Convenience function to validate web search results.

    Args:
        output: Web search tool output

    Returns:
        ValidationResult
    """
    return validate_tool_output("web_search", output)


def validate_image_generation(output: str | dict) -> ValidationResult:
    """Convenience function to validate image generation results.

    Args:
        output: Image generation tool output

    Returns:
        ValidationResult
    """
    return validate_tool_output("generate_image", output)


def validate_code_execution(output: str | dict) -> ValidationResult:
    """Convenience function to validate code execution results.

    Args:
        output: Code execution tool output

    Returns:
        ValidationResult
    """
    return validate_tool_output("execute_code", output)


def validate_file_operation(output: str | dict) -> ValidationResult:
    """Convenience function to validate sandbox file operation results.

    Args:
        output: Sandbox file operation output

    Returns:
        ValidationResult
    """
    return validate_tool_output("sandbox_file", output)


def extract_search_sources(validated_data: dict) -> list[dict[str, str]]:
    """Extract source information from validated search results.

    Converts validated search output into a list of source dicts
    suitable for event emission or state storage.

    Args:
        validated_data: Validated WebSearchOutput data

    Returns:
        List of source dicts with title, url, snippet keys
    """
    sources = []
    for result in validated_data.get("results", []):
        sources.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "snippet": result.get("snippet", ""),
        })
    return sources


def extract_images(validated_data: dict) -> list[dict[str, str]]:
    """Extract image data from validated code execution results.

    Args:
        validated_data: Validated CodeExecutionOutput data

    Returns:
        List of image dicts with data, type keys
    """
    images = []
    for img in validated_data.get("images", []):
        if img.get("data"):
            images.append({
                "data": img.get("data", ""),
                "type": img.get("type", "image/png"),
            })
    return images


def is_tool_error_response(output: str | dict) -> bool:
    """Check if tool output indicates an error.

    Args:
        output: Tool output

    Returns:
        True if output indicates an error
    """
    # Parse string output to dict if needed
    data: Any = output
    if isinstance(output, str):
        # Check for common error prefixes
        lower_output = output.lower()
        error_prefixes = ("error:", "error invoking", "failed to", "exception:")
        if any(prefix in lower_output for prefix in error_prefixes):
            return True

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return False

    # Check for error indicators in dict
    if isinstance(data, dict):
        return bool(data.get("error")) or data.get("success") is False

    return False


def get_error_message(output: str | dict) -> str | None:
    """Extract error message from tool output.

    Args:
        output: Tool output

    Returns:
        Error message string or None
    """
    if isinstance(output, str):
        if output.lower().startswith("error"):
            return output

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return None
    else:
        data = output

    if isinstance(data, dict):
        return data.get("error")

    return None
