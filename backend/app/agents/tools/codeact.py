"""CodeAct tool for hybrid code-action execution.

Provides an `execute_script` tool that accepts multi-line Python code with
access to a pre-installed `hyperagent` helper library in the sandbox.
Gated behind `execution_mode: "codeact"` configuration.
"""

from __future__ import annotations

import json
import textwrap

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.sandbox import get_execution_sandbox_manager

logger = get_logger(__name__)

# Timeout for script execution (seconds)
DEFAULT_SCRIPT_TIMEOUT = 300

# Helper library source (installed in sandbox on first use)
_HYPERAGENT_LIB_INSTALLED: set[str] = set()


class ExecuteScriptInput(BaseModel):
    """Input schema for CodeAct execute_script tool."""

    code: str = Field(
        ...,
        description=(
            "Multi-line Python script to execute. The `hyperagent` helper library "
            "is pre-installed and provides: hyperagent.web_search(query), "
            "hyperagent.read_file(path), hyperagent.write_file(path, content), "
            "hyperagent.run_command(cmd), hyperagent.browse(url). "
            "Print results to stdout for the agent to see."
        ),
    )
    timeout: int = Field(
        default=DEFAULT_SCRIPT_TIMEOUT,
        ge=10,
        le=600,
        description="Execution timeout in seconds (10-600, default 300)",
    )
    # Context fields (injected by agent, not provided by LLM)
    user_id: str | None = Field(
        default=None,
        json_schema_extra={"exclude": True},
    )
    task_id: str | None = Field(
        default=None,
        json_schema_extra={"exclude": True},
    )


async def _ensure_hyperagent_lib(sandbox_session) -> None:
    """Install the hyperagent helper library in the sandbox if not already done.

    The library is written as a single-file module to /tmp/hyperagent_lib/
    and installed via `pip install -e`. Subsequent calls skip installation
    if the sandbox ID is already in the installed set.

    Args:
        sandbox_session: ExecutionSandboxSession with active executor
    """
    sandbox_id = sandbox_session.sandbox_id
    if sandbox_id in _HYPERAGENT_LIB_INSTALLED:
        return

    runtime = sandbox_session.executor.get_runtime()

    # Read the helper library source via importlib (robust regardless of file location)
    from pathlib import Path

    import app.sandbox.hyperagent_lib as _lib_mod

    lib_source = Path(_lib_mod.__file__).read_text()

    # Write the library module to sandbox
    await runtime.run_command("mkdir -p /tmp/hyperagent_lib/hyperagent", timeout=10)
    await runtime.write_file("/tmp/hyperagent_lib/hyperagent/__init__.py", lib_source)

    # Write a minimal setup.py for pip install -e
    setup_py = textwrap.dedent("""\
        from setuptools import setup, find_packages
        setup(
            name="hyperagent",
            version="0.1.0",
            packages=find_packages(),
        )
    """)
    await runtime.write_file("/tmp/hyperagent_lib/setup.py", setup_py)

    # Install in development mode
    result = await runtime.run_command(
        "cd /tmp/hyperagent_lib && pip install -e . -q 2>&1 | tail -3",
        timeout=60,
    )

    if result.exit_code == 0:
        _HYPERAGENT_LIB_INSTALLED.add(sandbox_id)
        logger.info(
            "hyperagent_lib_installed",
            sandbox_id=sandbox_id,
        )
    else:
        logger.warning(
            "hyperagent_lib_install_failed",
            sandbox_id=sandbox_id,
            stderr=result.stderr[:500] if result.stderr else "",
            stdout=result.stdout[:500] if result.stdout else "",
        )

    # Create helper directories
    await runtime.run_command(
        "mkdir -p /tmp/hyperagent/search_cache /tmp/hyperagent/requests",
        timeout=10,
    )


@tool(args_schema=ExecuteScriptInput)
async def execute_script(
    code: str,
    timeout: int = DEFAULT_SCRIPT_TIMEOUT,
    user_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Execute a multi-line Python script with the hyperagent helper library.

    Runs a Python script in an isolated sandbox with access to the
    `hyperagent` helper library for file I/O, web search, shell commands,
    and HTTP fetching. The sandbox is reused within the same session.

    The hyperagent library provides:
    - hyperagent.web_search(query) - search the web
    - hyperagent.read_file(path) - read a file
    - hyperagent.write_file(path, content) - write a file
    - hyperagent.run_command(cmd) - run a shell command
    - hyperagent.browse(url) - fetch a URL
    - hyperagent.list_files(dir) - list directory contents

    Print results to stdout. The script output will be returned to you.

    Args:
        code: Python script to execute
        timeout: Execution timeout in seconds
        user_id: User ID for session management (injected)
        task_id: Task ID for session management (injected)

    Returns:
        JSON string with execution results
    """
    from app.sandbox import is_execution_sandbox_available

    if not is_execution_sandbox_available():
        return json.dumps({
            "success": False,
            "stdout": "",
            "stderr": "Code execution sandbox not available.",
            "exit_code": None,
            "created_files": [],
            "error": "Sandbox not available. Check SANDBOX_PROVIDER configuration.",
        })

    try:
        # Get or create sandbox session
        sandbox_manager = get_execution_sandbox_manager()
        session = await sandbox_manager.get_or_create_sandbox(
            user_id=user_id,
            task_id=task_id,
        )

        # Ensure hyperagent helper library is installed
        await _ensure_hyperagent_lib(session)

        runtime = session.executor.get_runtime()

        # Snapshot files before execution to detect created files
        pre_files_result = await runtime.run_command(
            "find /home/user -maxdepth 3 -type f -newer /tmp/hyperagent_lib/setup.py "
            "2>/dev/null | sort",
            timeout=10,
        )
        pre_files = set(
            pre_files_result.stdout.strip().split("\n")
        ) if pre_files_result.exit_code == 0 and pre_files_result.stdout.strip() else set()

        # Write the script to a temp file and execute
        script_path = "/tmp/hyperagent/current_script.py"
        await runtime.write_file(script_path, code)

        exec_result = await runtime.run_command(
            f"cd /home/user && python {script_path}",
            timeout=timeout,
        )

        # Detect newly created files
        post_files_result = await runtime.run_command(
            "find /home/user -maxdepth 3 -type f -newer /tmp/hyperagent_lib/setup.py "
            "2>/dev/null | sort",
            timeout=10,
        )
        post_files = set(
            post_files_result.stdout.strip().split("\n")
        ) if post_files_result.exit_code == 0 and post_files_result.stdout.strip() else set()

        created_files = sorted(post_files - pre_files)

        result = {
            "success": exec_result.exit_code == 0,
            "stdout": exec_result.stdout or "",
            "stderr": exec_result.stderr or "",
            "exit_code": exec_result.exit_code,
            "created_files": created_files,
            "sandbox_id": session.sandbox_id,
        }

        logger.info(
            "codeact_script_executed",
            success=result["success"],
            exit_code=result["exit_code"],
            sandbox_id=session.sandbox_id,
            stdout_len=len(result["stdout"]),
            created_files_count=len(created_files),
        )

        return json.dumps(result)

    except Exception as e:
        import traceback

        logger.error(
            "codeact_script_error",
            error=str(e),
            traceback=traceback.format_exc(),
        )
        return json.dumps({
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "exit_code": None,
            "created_files": [],
            "error": str(e),
        })
