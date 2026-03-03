"""Code Execution Tool.

Provides a LangChain tool for executing code in E2B sandboxes with
session-based sandbox management for reuse across tool calls.
"""

import json
import time
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.sandbox import get_execution_sandbox_manager

logger = get_logger(__name__)


class ExecuteCodeInput(BaseModel):
    """Input schema for code execution tool."""

    code: str = Field(
        ...,
        description="The code to execute in the sandbox",
    )
    language: Literal["python", "javascript", "typescript", "bash"] = Field(
        default="python",
        description="Programming language of the code",
    )
    packages: list[str] | None = Field(
        default=None,
        description="Optional list of packages to install before execution (pip for Python, npm for JS/TS)",
    )
    capture_images: bool = Field(
        default=True,
        description="Whether to capture image outputs (PNG/HTML from /tmp/output*)",
    )
    timeout: int = Field(
        default=180,
        ge=1,
        le=600,
        description="Execution timeout in seconds (1-600)",
    )
    # Context fields (injected by agent, not provided by LLM)
    # These are excluded from the JSON schema so the LLM doesn't see them
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


@tool(args_schema=ExecuteCodeInput)
async def execute_code(
    code: str,
    language: Literal["python", "javascript", "typescript", "bash"] = "python",
    packages: list[str] | None = None,
    capture_images: bool = True,
    timeout: int = 180,
    # Session context (injected by agent, not provided by LLM)
    user_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Execute code in an E2B sandbox.

    Runs code in an isolated sandbox environment with support for multiple
    languages, package installation, and image capture. The sandbox
    is reused within the same user/task session for efficiency.

    Args:
        code: The code to execute
        language: Programming language (python, javascript, typescript, bash)
        packages: Optional packages to install before execution
        capture_images: Whether to capture output files from /tmp/output*
        timeout: Execution timeout in seconds
        user_id: User ID for session management (injected)
        task_id: Task ID for session management (injected)

    Returns:
        JSON string with execution results including:
        - success: Whether execution completed without errors
        - stdout: Standard output from execution
        - stderr: Standard error output
        - exit_code: Process exit code
        - images: List of captured images (if enabled)
        - error: Error message if execution failed
        - sandbox_id: ID of the sandbox used
    """
    # Check if execution sandbox is available (provider-agnostic)
    from app.sandbox import is_execution_sandbox_available

    if not is_execution_sandbox_available():
        logger.warning("execution_sandbox_not_available")
        return json.dumps({
            "success": False,
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "images": [],
            "error": "Code execution sandbox not available. Check SANDBOX_PROVIDER configuration.",
            "sandbox_id": None,
        })

    # Display command for terminal events
    lang_cmd_map = {
        "python": "python3 script.py",
        "javascript": "node script.js",
        "typescript": "npx ts-node script.ts",
        "bash": "bash script.sh",
    }
    display_cmd = lang_cmd_map.get(language, f"{language} script")

    # Import shared helpers for terminal/workspace events
    from app.agents.tools.shell_tools import (
        _build_terminal_events,
        _detect_workspace_changes,
    )

    try:
        # Get or create sandbox session
        sandbox_manager = get_execution_sandbox_manager()
        session = await sandbox_manager.get_or_create_sandbox(
            user_id=user_id,
            task_id=task_id,
        )
        executor = session.executor
        sandbox_id = (
            session.sandbox_id or task_id or user_id or "exec-sandbox"
        )

        # Install packages if requested
        if packages:
            # Determine package manager based on language
            pkg_manager_map = {
                "python": "pip",
                "py": "pip",
                "javascript": "npm",
                "js": "npm",
                "typescript": "npm",
                "ts": "npm",
            }
            pkg_manager = pkg_manager_map.get(language)

            if pkg_manager:
                success, stdout, stderr = await executor.install_packages(
                    packages,
                    package_manager=pkg_manager,
                )
                if not success:
                    logger.warning(
                        "package_installation_warning",
                        packages=packages,
                        stderr=stderr[:500] if stderr else None,
                    )

        # Execute the code
        pre_exec_time = time.time()
        exec_result = await executor.execute_code(
            code=code,
            language=language,
            timeout=timeout,
        )

        # Capture images if requested
        images = []
        if capture_images:
            images = await executor.capture_images()

        exec_stdout = exec_result.get("stdout") or ""
        exec_stderr = exec_result.get("stderr") or ""
        exit_code = exec_result.get("exit_code", 0)

        # Build terminal events using shared helper
        terminal_events = _build_terminal_events(
            command=display_cmd,
            stdout=exec_stdout,
            stderr=exec_stderr if not exec_result["success"] else None,
            exit_code=exit_code,
        )

        # Detect workspace changes
        runtime = executor.get_runtime()
        workspace_events = await _detect_workspace_changes(
            runtime, pre_exec_time, sandbox_id,
        )

        result = {
            "success": exec_result["success"],
            "stdout": exec_result["stdout"],
            "stderr": exec_result["stderr"],
            "exit_code": exec_result["exit_code"],
            "images": images,
            "error": (
                None
                if exec_result["success"]
                else exec_result.get("stderr", "")
            ),
            "sandbox_id": session.sandbox_id,
            "terminal_events": terminal_events,
            "workspace_events": workspace_events,
        }

        logger.info(
            "code_execution_completed",
            language=language,
            success=result["success"],
            exit_code=result["exit_code"],
            sandbox_id=session.sandbox_id,
            image_count=len(images),
        )

        return json.dumps(result)

    except Exception as e:
        import traceback
        logger.error(
            "code_execution_error",
            error=str(e),
            traceback=traceback.format_exc(),
        )
        terminal_events = _build_terminal_events(
            command=display_cmd, stderr=str(e), exit_code=1,
        )
        return json.dumps({
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "exit_code": None,
            "images": [],
            "error": str(e),
            "sandbox_id": None,
            "terminal_events": terminal_events,
        })


async def execute_code_with_context(
    code: str,
    language: Literal["python", "javascript", "typescript", "bash"] = "python",
    packages: list[str] | None = None,
    capture_images: bool = True,
    timeout: int = 180,
    user_id: str | None = None,
    task_id: str | None = None,
) -> dict:
    """Execute code and return parsed result dict.

    Convenience wrapper that returns a dict instead of JSON string.

    Args:
        code: The code to execute
        language: Programming language
        packages: Optional packages to install
        capture_images: Whether to capture output files
        timeout: Execution timeout in seconds
        user_id: User ID for session management
        task_id: Task ID for session management

    Returns:
        Dict with execution results
    """
    result_json = await execute_code.ainvoke({
        "code": code,
        "language": language,
        "packages": packages,
        "capture_images": capture_images,
        "timeout": timeout,
        "user_id": user_id,
        "task_id": task_id,
    })
    return json.loads(result_json)
