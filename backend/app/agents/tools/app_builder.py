"""App Builder Tools for creating and running web applications.

These tools enable the agent to scaffold, build, and run web applications
in an E2B sandbox with port forwarding for live previews.
"""

import os
import time
from typing import Any, Literal

from langchain_core.tools import tool

from app.agents import events
from app.core.logging import get_logger
from app.sandbox.app_sandbox_manager import (
    APP_TEMPLATES,
    get_app_sandbox_manager,
)

logger = get_logger(__name__)

# Blocklist of dangerous commands/patterns that should never run in the sandbox.
DANGEROUS_COMMANDS = [
    "rm -rf",
    "curl",
    "wget",
    "nc",
    "ncat",
    "dd",
    "mkfs",
    "shutdown",
    "reboot",
    "python -c",
    "python3 -c",
    "bash -c",
    "sh -c",
    "chmod",
    "chown",
]

# Maximum allowed timeout for app_run_command (seconds)
MAX_COMMAND_TIMEOUT = 300


def _is_command_blocked(command: str) -> str | None:
    """Check if a command matches the dangerous commands blocklist.

    Returns the matched pattern if blocked, None if allowed.
    """
    cmd_lower = command.strip().lower()

    # Block command substitution patterns that could bypass the blocklist
    for injection_pattern in ["$(", "`", "eval ", "exec "]:
        if injection_pattern in cmd_lower:
            return injection_pattern.strip()

    for pattern in DANGEROUS_COMMANDS:
        # Check if the dangerous command appears as a standalone token
        # (at start, after pipe, after semicolon, after &&, after ||)
        segments = cmd_lower.replace("&&", ";").replace("||", ";").replace("|", ";").split(";")
        for segment in segments:
            segment = segment.strip()
            if segment.startswith(pattern) or segment.startswith(f"sudo {pattern}"):
                return pattern
    return None


def _create_terminal_events(
    command: str,
    output: str | None = None,
    error: str | None = None,
    exit_code: int = 0,
    cwd: str = "/home/user/app",
) -> list[dict[str, Any]]:
    """Create terminal event dicts to embed in tool results.

    These events will be extracted by chat.py and forwarded to the frontend.
    """
    events = []
    timestamp = int(time.time() * 1000)

    # Command event
    events.append({
        "type": "terminal_command",
        "command": command,
        "cwd": cwd,
        "timestamp": timestamp,
    })

    # Output event
    if output:
        events.append({
            "type": "terminal_output",
            "content": output,
            "stream": "stdout",
            "timestamp": timestamp,
        })

    # Error event
    if error:
        events.append({
            "type": "terminal_error",
            "content": error,
            "exit_code": exit_code,
            "timestamp": timestamp,
        })

    # Complete event
    events.append({
        "type": "terminal_complete",
        "exit_code": exit_code,
        "timestamp": timestamp,
    })

    return events


@tool
async def create_app_project(
    template: Literal[
        "react", "react-ts", "nextjs", "vue", "express", "fastapi", "flask", "static"
    ] = "react",
    user_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Create a new web application project using a template.

    This tool scaffolds a new project in an isolated sandbox environment.
    The sandbox persists across tool calls so you can iteratively develop the app.

    Available templates:
    - react: React with Vite (port 5173)
    - react-ts: React with TypeScript and Vite (port 5173)
    - nextjs: Next.js with TypeScript and Tailwind (port 3000)
    - vue: Vue 3 with Vite (port 5173)
    - express: Express.js backend (port 3000)
    - fastapi: FastAPI Python backend (port 8000)
    - flask: Flask Python backend (port 5000)
    - static: Static HTML site (port 3000)

    Args:
        template: The template to use for scaffolding
        user_id: User ID for session management (optional)
        task_id: Task ID for session management (optional)

    Returns:
        Dict with project creation result including project directory

    .. note::
        TODO: user_id and task_id should be injected from the authenticated
        request context rather than passed as tool arguments. This requires
        framework-level support for binding request-scoped values to tool
        invocations.
    """
    logger.info(
        "create_app_project_called",
        template=template,
        user_id=user_id,
        task_id=task_id,
    )

    try:
        manager = get_app_sandbox_manager()

        # Get or create sandbox session
        session = await manager.get_or_create_sandbox(
            user_id=user_id,
            task_id=task_id,
            template=template,
        )

        # Scaffold the project
        result = await manager.scaffold_project(session, template)

        # Create terminal events for the scaffold command
        scaffold_cmd = APP_TEMPLATES[template]["scaffold_cmd"]
        terminal_events = _create_terminal_events(
            command=scaffold_cmd,
            output=f"Created {APP_TEMPLATES[template]['name']} project successfully",
            exit_code=0 if result["success"] else 1,
            cwd="/home/user",
        )

        if result["success"]:
            return {
                "success": True,
                "template": template,
                "template_name": APP_TEMPLATES[template]["name"],
                "project_dir": session.project_dir,
                "sandbox_id": session.sandbox_id,
                "message": f"Created {APP_TEMPLATES[template]['name']} project. Use app_write_file to add code, then app_start_server to run it.",
                "terminal_events": terminal_events,
            }
        else:
            result["terminal_events"] = terminal_events
            return result

    except Exception as e:
        logger.error("create_app_project_failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
        }


@tool
async def app_write_file(
    path: str,
    content: str,
    user_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Write or update a file in the app project.

    Use this to create or modify source files in your application.
    Paths are relative to the project directory (e.g., "src/App.jsx").

    Args:
        path: File path relative to project directory (e.g., "src/App.jsx", "index.html")
        content: The content to write to the file
        user_id: User ID for session management (optional)
        task_id: Task ID for session management (optional)

    Returns:
        Dict with write result

    .. note::
        TODO: user_id and task_id should be injected from the authenticated
        request context rather than passed as tool arguments.
    """
    logger.info(
        "app_write_file_called",
        path=path,
    )

    # Reject path traversal attempts
    if ".." in path:
        logger.warning("app_write_file_path_traversal", path=path)
        return {
            "success": False,
            "error": "Path traversal ('..') is not allowed in file paths.",
        }

    try:
        manager = get_app_sandbox_manager()

        # Get existing session; require create_app_project to be called first
        session = await manager.get_session(user_id=user_id, task_id=task_id)
        if not session:
            return {
                "success": False,
                "error": (
                    "No active app session. Use create_app_project"
                    " first to scaffold a project before writing files."
                ),
            }

        # Write the file
        result = await manager.write_file(session, path, content)

        # Emit workspace_update event so the frontend file browser updates
        if result.get("success"):
            sandbox_id = getattr(session, "session_key", "") or task_id or user_id or "app-sandbox"
            file_path = result.get("path", path)
            workspace_event = events.workspace_update(
                operation="create",
                path=file_path,
                name=os.path.basename(file_path),
                sandbox_type="app",
                sandbox_id=sandbox_id,
                size=result.get("size") or result.get("bytes_written"),
            )
            result["workspace_events"] = [workspace_event]

        return result

    except Exception as e:
        logger.error("app_write_file_failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
        }


@tool
async def app_read_file(
    path: str,
    user_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Read a file from the app project.

    Use this to examine existing source files in the project.
    Paths are relative to the project directory.

    Args:
        path: File path relative to project directory
        user_id: User ID for session management (optional)
        task_id: Task ID for session management (optional)

    Returns:
        Dict with file content

    .. note::
        TODO: user_id and task_id should be injected from the authenticated
        request context rather than passed as tool arguments.
    """
    logger.info(
        "app_read_file_called",
        path=path,
    )

    try:
        manager = get_app_sandbox_manager()

        # Get existing session
        session = await manager.get_session(user_id=user_id, task_id=task_id)
        if not session:
            return {
                "success": False,
                "error": "No active app session. Use create_app_project first.",
            }

        # Read the file
        result = await manager.read_file(session, path)

        return result

    except Exception as e:
        logger.error("app_read_file_failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
        }


@tool
async def app_list_files(
    path: str = "",
    user_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """List files in a directory of the app project.

    Use this to explore the project structure.
    Leave path empty to list the project root.

    Args:
        path: Directory path relative to project directory (empty for root)
        user_id: User ID for session management (optional)
        task_id: Task ID for session management (optional)

    Returns:
        Dict with file listing

    .. note::
        TODO: user_id and task_id should be injected from the authenticated
        request context rather than passed as tool arguments.
    """
    logger.info(
        "app_list_files_called",
        path=path,
    )

    try:
        manager = get_app_sandbox_manager()

        # Get existing session
        session = await manager.get_session(user_id=user_id, task_id=task_id)
        if not session:
            return {
                "success": False,
                "error": "No active app session. Use create_app_project first.",
            }

        # List files
        result = await manager.list_files(session, path)

        return result

    except Exception as e:
        logger.error("app_list_files_failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
        }


@tool
async def app_install_packages(
    packages: list[str],
    package_manager: Literal["npm", "pip"] = "npm",
    user_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Install npm or pip packages in the app project.

    Use this to add dependencies to your project.

    Args:
        packages: List of package names to install (e.g., ["axios", "lodash"])
        package_manager: Package manager to use ("npm" or "pip")
        user_id: User ID for session management (optional)
        task_id: Task ID for session management (optional)

    Returns:
        Dict with installation result

    .. note::
        TODO: user_id and task_id should be injected from the authenticated
        request context rather than passed as tool arguments.
    """
    logger.info(
        "app_install_packages_called",
        packages=packages,
        manager=package_manager,
    )

    try:
        manager = get_app_sandbox_manager()

        # Get existing session
        session = await manager.get_session(user_id=user_id, task_id=task_id)
        if not session:
            return {
                "success": False,
                "error": "No active app session. Use create_app_project first.",
            }

        # Install packages
        result = await manager.install_dependencies(session, packages, package_manager)

        # Create terminal events
        install_cmd = f"{package_manager} install {' '.join(packages)}"
        terminal_events = _create_terminal_events(
            command=install_cmd,
            output=f"Installed {len(packages)} package(s)" if result.get("success") else None,
            error=result.get("error") if not result.get("success") else None,
            exit_code=0 if result.get("success") else 1,
            cwd="/home/user/app",
        )
        result["terminal_events"] = terminal_events
        return result

    except Exception as e:
        logger.error("app_install_packages_failed", error=str(e))
        packages_str = " ".join(packages) if packages else ""
        return {
            "success": False,
            "error": str(e),
            "terminal_events": _create_terminal_events(
                command=f"{package_manager} install {packages_str}",
                error=str(e),
                exit_code=1,
                cwd="/home/user/app",
            ),
        }


@tool
async def app_start_server(
    custom_command: str | None = None,
    port: int | None = None,
    user_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Start the development server and get a live preview URL.

    This starts the dev server for the current project and returns
    a public URL where the app can be viewed in a browser.

    The server runs in the background. Use app_stop_server to stop it.

    Args:
        custom_command: Custom start command (optional, uses template default)
        port: Custom port (optional, uses template default)
        user_id: User ID for session management (optional)
        task_id: Task ID for session management (optional)

    Returns:
        Dict with server info including preview_url

    .. note::
        TODO: user_id and task_id should be injected from the authenticated
        request context rather than passed as tool arguments.
    """
    logger.info(
        "app_start_server_called",
        custom_command=custom_command,
        port=port,
    )

    try:
        manager = get_app_sandbox_manager()

        # Get existing session
        session = await manager.get_session(user_id=user_id, task_id=task_id)
        if not session:
            return {
                "success": False,
                "error": "No active app session. Use create_app_project first.",
            }

        # Start the dev server
        result = await manager.start_dev_server(session, custom_command, port)

        # Create terminal events for the start command
        template = session.template or "react"
        start_cmd = custom_command or APP_TEMPLATES.get(template, APP_TEMPLATES["react"])["start_cmd"]

        if result["success"]:
            display_url = result.get("display_url", f"http://localhost:{result['port']}")
            terminal_events = _create_terminal_events(
                command=start_cmd,
                output=f"Server started at {display_url}",
                exit_code=0,
                cwd="/home/user/app",
            )
            return {
                "success": True,
                "preview_url": result["preview_url"],
                "display_url": display_url,
                "port": result["port"],
                "message": f"App is running! View it at: {display_url}",
                "terminal_events": terminal_events,
            }

        terminal_events = _create_terminal_events(
            command=start_cmd,
            error=result.get("error", "Failed to start server"),
            exit_code=1,
            cwd="/home/user/app",
        )
        result["terminal_events"] = terminal_events
        return result

    except Exception as e:
        logger.error("app_start_server_failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
        }


@tool
async def app_stop_server(
    user_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Stop the running development server.

    Use this when you need to stop the current server,
    for example before making major changes or restarting.

    Args:
        user_id: User ID for session management (optional)
        task_id: Task ID for session management (optional)

    Returns:
        Dict with stop result

    .. note::
        TODO: user_id and task_id should be injected from the authenticated
        request context rather than passed as tool arguments.
    """
    logger.info("app_stop_server_called")

    try:
        manager = get_app_sandbox_manager()

        # Get existing session
        session = await manager.get_session(user_id=user_id, task_id=task_id)
        if not session:
            return {
                "success": True,
                "message": "No active app session.",
            }

        # Stop the server
        result = await manager.stop_dev_server(session)

        return result

    except Exception as e:
        logger.error("app_stop_server_failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
        }


@tool
async def app_run_command(
    command: str,
    timeout: int = 60,
    user_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Run a shell command in the app sandbox.

    Use this for build commands, linting, testing, or other CLI operations.

    Args:
        command: The command to run (e.g., "npm run build", "npm test")
        timeout: Command timeout in seconds (default 60)
        user_id: User ID for session management (optional)
        task_id: Task ID for session management (optional)

    Returns:
        Dict with command output

    .. note::
        TODO: user_id and task_id should be injected from the authenticated
        request context rather than passed as tool arguments. This requires
        framework-level support for binding request-scoped values to tool
        invocations. See also: create_app_project, app_write_file, and other
        tool functions with the same pattern.
    """
    # Cap timeout to prevent abuse
    if timeout > MAX_COMMAND_TIMEOUT:
        timeout = MAX_COMMAND_TIMEOUT

    logger.info(
        "app_run_command_called",
        command=command,
    )

    # Check command against the dangerous commands blocklist
    blocked = _is_command_blocked(command)
    if blocked:
        logger.warning(
            "app_run_command_blocked",
            command=command,
            matched_pattern=blocked,
        )
        return {
            "success": False,
            "error": f"Command blocked for safety: contains '{blocked}'",
            "terminal_events": _create_terminal_events(
                command=command,
                error=f"Command blocked: '{blocked}' is not allowed in the sandbox",
                exit_code=1,
                cwd="/home/user/app",
            ),
        }

    try:
        manager = get_app_sandbox_manager()

        # Get existing session
        session = await manager.get_session(user_id=user_id, task_id=task_id)
        if not session:
            return {
                "success": False,
                "error": "No active app session. Use create_app_project first.",
            }

        # Run the command
        result = await manager.run_command(session, command, timeout)

        # Create terminal events
        terminal_events = _create_terminal_events(
            command=command,
            output=result.get("stdout", "") if result.get("success") else None,
            error=result.get("stderr") or result.get("error") if not result.get("success") else None,
            exit_code=result.get("exit_code", 0 if result.get("success") else 1),
            cwd="/home/user/app",
        )
        result["terminal_events"] = terminal_events
        return result

    except Exception as e:
        logger.error("app_run_command_failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "terminal_events": _create_terminal_events(
                command=command,
                error=str(e),
                exit_code=1,
                cwd="/home/user/app",
            ),
        }


@tool
async def app_get_preview_url(
    user_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Get the current preview URL for the running app.

    Use this to retrieve the live preview URL for the user.

    Args:
        user_id: User ID for session management (optional)
        task_id: Task ID for session management (optional)

    Returns:
        Dict with preview URL if server is running

    .. note::
        TODO: user_id and task_id should be injected from the authenticated
        request context rather than passed as tool arguments.
    """
    logger.info("app_get_preview_url_called")

    try:
        manager = get_app_sandbox_manager()

        # Get existing session
        session = await manager.get_session(user_id=user_id, task_id=task_id)
        if not session:
            return {
                "success": False,
                "error": "No active app session.",
            }

        if session.preview_url:
            return {
                "success": True,
                "preview_url": session.preview_url,
                "is_running": session.app_process.is_running if session.app_process else False,
            }
        else:
            return {
                "success": False,
                "error": "Server not running. Use app_start_server first.",
            }

    except Exception as e:
        logger.error("app_get_preview_url_failed", error=str(e))
        return {
            "success": False,
            "error": str(e),
        }


# List all app builder tools
APP_BUILDER_TOOLS = [
    create_app_project,
    app_write_file,
    app_read_file,
    app_list_files,
    app_install_packages,
    app_start_server,
    app_stop_server,
    app_run_command,
    app_get_preview_url,
]


def get_app_builder_tools():
    """Get all app builder tools.

    Returns:
        List of app builder tool instances
    """
    return APP_BUILDER_TOOLS
