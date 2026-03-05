"""App Sandbox Manager for Session Lifecycle.

Provides session-based app development sandbox lifecycle management, enabling sandbox
sharing across multiple tool calls within the same user/task context.

Key features:
- Port forwarding for web applications
- Project scaffolding with templates
- Dev server management
- Long-running sessions for iterative development
"""

import asyncio
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from app.core.logging import get_logger
from app.sandbox import file_operations
from app.sandbox.runtime import SandboxRuntime

logger = get_logger(__name__)


def get_default_app_session_timeout() -> timedelta:
    """Get default app session timeout from settings (longer than code execution)."""
    # Use a longer timeout for app development (30 minutes default)
    return timedelta(minutes=30)


# Cleanup interval for expired sessions (60 seconds)
CLEANUP_INTERVAL_SECONDS = 60

# Skip health checks if last successful check was less than this many seconds ago
HEALTH_CHECK_SKIP_SECONDS = 30

# Regex for validating package names (prevents shell injection)
_PACKAGE_NAME_RE = re.compile(r'^(@[a-zA-Z0-9._-]+/)?[a-zA-Z0-9._-]+([<>=!~@]+[a-zA-Z0-9.*]+)?$')

# Shell metacharacters that are not allowed in custom commands
_SHELL_METACHARACTERS = set(';|&$`><(){}\n')

# Pre-cached template directory inside the custom app sandbox image
TEMPLATE_CACHE_DIR = "/opt/templates"
TEMPLATE_CACHE_MARKER = "/opt/templates/.cached"

# Supported app templates
# Note: Using vite@5 instead of vite@latest because E2B sandbox has Node.js 20.9.0
# and Vite 6+ requires Node.js 20.19+ or 22.12+
APP_TEMPLATES = {
    "react": {
        "name": "React (Vite)",
        "scaffold_cmd": "npm create vite@5 app -- --template react && cd app && npm install",
        "start_cmd": "cd app && npm run dev -- --host 0.0.0.0",
        "port": 5173,
    },
    "react-ts": {
        "name": "React TypeScript (Vite)",
        "scaffold_cmd": "npm create vite@5 app -- --template react-ts && cd app && npm install",
        "start_cmd": "cd app && npm run dev -- --host 0.0.0.0",
        "port": 5173,
    },
    "nextjs": {
        "name": "Next.js",
        "scaffold_cmd": "npx create-next-app@14 app --typescript --tailwind --eslint --app --src-dir --use-npm --no-git",
        "start_cmd": "cd app && npm run dev -- -H 0.0.0.0",
        "port": 3000,
    },
    "vue": {
        "name": "Vue 3 (Vite)",
        "scaffold_cmd": "npm create vite@5 app -- --template vue && cd app && npm install",
        "start_cmd": "cd app && npm run dev -- --host 0.0.0.0",
        "port": 5173,
    },
    "express": {
        "name": "Express.js",
        "scaffold_cmd": "mkdir -p app && cd app && npm init -y && npm install express",
        "start_cmd": "cd app && node index.js",
        "port": 3000,
    },
    "fastapi": {
        "name": "FastAPI",
        "scaffold_cmd": "mkdir -p app && cd app && pip install fastapi uvicorn",
        "start_cmd": "cd app && uvicorn main:app --host 0.0.0.0 --port 8000 --reload",
        "port": 8000,
    },
    "flask": {
        "name": "Flask",
        "scaffold_cmd": "mkdir -p app && cd app && pip install flask",
        "start_cmd": "cd app && flask run --host 0.0.0.0 --port 5000",
        "port": 5000,
    },
    "static": {
        "name": "Static HTML",
        "scaffold_cmd": "mkdir -p app",
        "start_cmd": "npx -y serve app -l 3000 --listen 0.0.0.0",
        "port": 3000,
    },
}


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


@dataclass
class AppProcess:
    """Tracks a running app process."""

    pid: int | None = None
    command: str = ""
    started_at: datetime = field(default_factory=_utcnow)
    is_running: bool = False


@dataclass
class AppSandboxSession:
    """Tracks an active app development sandbox session."""

    sandbox: SandboxRuntime
    session_key: str
    template: str = "react"
    project_dir: str = "/home/user/app"
    created_at: datetime = field(default_factory=_utcnow)
    last_accessed: datetime = field(default_factory=_utcnow)
    timeout: timedelta = field(default_factory=get_default_app_session_timeout)

    # Running process info
    app_process: AppProcess | None = None
    preview_url: str | None = None
    internal_url: str | None = None  # Raw localhost URL for proxy use

    # Timestamp of last successful health check (monotonic clock)
    last_health_check: float = 0.0

    @property
    def sandbox_id(self) -> str | None:
        """Get the underlying sandbox ID."""
        if self.sandbox:
            return self.sandbox.sandbox_id
        return None

    @property
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return datetime.now(timezone.utc) > (self.last_accessed + self.timeout)

    def touch(self) -> None:
        """Update last accessed time to prevent expiry."""
        self.last_accessed = datetime.now(timezone.utc)


class AppSandboxManager:
    """Manages app development sandbox sessions across multiple tool invocations.

    Uses a session key (user_id:task_id) to enable sandbox reuse
    within the same context. Provides automatic cleanup of expired
    sessions via background task.
    """

    _instance: "AppSandboxManager | None" = None
    _lock: "asyncio.Lock | None" = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        """Lazily create the class-level lock to avoid binding to wrong event loop."""
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    def __init__(self) -> None:
        """Initialize the app sandbox manager."""
        self._sessions: dict[str, AppSandboxSession] = {}
        self._cleanup_task: asyncio.Task[None] | None = None
        self._session_lock = asyncio.Lock()
        # Port pool for BoxLite host port mapping (reclaimed on cleanup)
        self._used_ports: set[int] = set()
        # Metrics counters
        self._total_created: int = 0
        self._total_cleaned: int = 0
        self._total_reused: int = 0
        self._health_check_failures: int = 0

    def _allocate_host_port(self) -> int:
        """Allocate the lowest available host port for BoxLite.

        Returns:
            An available host port number.

        Raises:
            RuntimeError: If no ports are available in the allowed range.
        """
        from app.config import settings

        start = settings.boxlite_app_host_port_start
        max_port = min(start + 1000, 65535)
        pool_size = max_port - start + 1
        for port in range(start, max_port + 1):
            if port not in self._used_ports:
                self._used_ports.add(port)
                utilization = len(self._used_ports) / pool_size
                if utilization > 0.8:
                    logger.warning(
                        "app_port_pool_high_utilization",
                        used=len(self._used_ports),
                        total=pool_size,
                        utilization_pct=round(utilization * 100),
                    )
                return port
        raise RuntimeError(
            f"No available ports in range {start}-{max_port}. "
            f"{len(self._used_ports)} ports in use."
        )

    def _release_host_port(self, port: int) -> None:
        """Release a previously allocated host port back to the pool."""
        self._used_ports.discard(port)

    @classmethod
    def get_instance(cls) -> "AppSandboxManager":
        """Get the singleton instance of AppSandboxManager."""
        if cls._instance is None:
            cls._instance = AppSandboxManager()
        return cls._instance

    @staticmethod
    def make_session_key(user_id: str | None, task_id: str | None) -> str:
        """Create a session key from user and task IDs.

        Args:
            user_id: User identifier
            task_id: Task identifier

        Returns:
            Session key string
        """
        user = user_id or "anonymous"
        task = task_id or "default"
        return f"app:{user}:{task}"

    async def get_or_create_sandbox(
        self,
        user_id: str | None = None,
        task_id: str | None = None,
        template: str = "react",
        timeout: timedelta | None = None,
    ) -> AppSandboxSession:
        """Get an existing sandbox session or create a new one.

        Args:
            user_id: User identifier
            task_id: Task identifier
            template: App template to use (react, nextjs, vue, express, fastapi, etc.)
            timeout: Session timeout (defaults to 30 minutes)

        Returns:
            AppSandboxSession with active sandbox

        Raises:
            ValueError: If E2B API key not configured or invalid template
        """
        # Validate prerequisites at manager level (fail fast)
        from app.sandbox.provider import is_provider_available

        available, issue = is_provider_available("app")
        if not available:
            raise ValueError(issue)

        if template not in APP_TEMPLATES:
            raise ValueError(
                f"Invalid template: {template}. Available: {list(APP_TEMPLATES.keys())}"
            )

        session_key = self.make_session_key(user_id, task_id)
        session_timeout = timeout or get_default_app_session_timeout()

        async with self._session_lock:
            # Check for existing valid session
            if session_key in self._sessions:
                session = self._sessions[session_key]
                # Perform health check: not expired and sandbox still alive
                if not session.is_expired and await self._is_sandbox_healthy(session):
                    session.touch()
                    self._total_reused += 1
                    logger.info(
                        "app_sandbox_session_reused",
                        session_key=session_key,
                        sandbox_id=session.sandbox_id,
                    )
                    return session
                else:
                    # Clean up unhealthy or expired session
                    await self._cleanup_session_internal(session_key)

            # Create new sandbox via provider factory
            from app.config import settings
            from app.sandbox.provider import create_app_runtime

            # Determine the container port for this template and allocate
            # a host port so the dev server is reachable from the host.
            template_config = APP_TEMPLATES[template]
            container_port = int(template_config["port"])

            runtime_kwargs: dict[str, Any] = {}
            allocated_host_port: int | None = None
            if settings.sandbox_provider == "boxlite":
                host_port = self._allocate_host_port()
                allocated_host_port = host_port
                runtime_kwargs["ports"] = {container_port: host_port}

            try:
                sandbox = await create_app_runtime(**runtime_kwargs)
                logger.info("app_sandbox_created", sandbox_id=sandbox.sandbox_id)
            except Exception as e:
                if allocated_host_port is not None:
                    self._release_host_port(allocated_host_port)
                logger.error("app_sandbox_creation_failed", error=str(e))
                raise

            session = AppSandboxSession(
                sandbox=sandbox,
                session_key=session_key,
                template=template,
                timeout=session_timeout,
            )
            # Mark as healthy since we just created it — avoids a redundant
            # health-check in the very next get_session() call which can
            # fail if the container is still settling after scaffolding.
            session.last_health_check = time.monotonic()
            self._sessions[session_key] = session
            self._total_created += 1

            logger.info(
                "app_sandbox_session_created",
                session_key=session_key,
                sandbox_id=session.sandbox_id,
                template=template,
            )

            # Start cleanup task if not running
            self._ensure_cleanup_task()

            return session

    async def _try_cached_scaffold(
        self,
        session: AppSandboxSession,
        template: str,
    ) -> dict[str, Any] | None:
        """Try to scaffold from a pre-cached template in the Docker image.

        Returns a result dict on cache hit, or None if the cache is
        unavailable (missing marker, missing template dir, or copy failure).
        Callers should fall through to the live scaffold on None.
        """
        try:
            # Check marker file and template directory exist
            check = await session.sandbox.run_command(
                f"test -f {TEMPLATE_CACHE_MARKER} && test -d {TEMPLATE_CACHE_DIR}/{template}",
                timeout=5,
            )
            if check.exit_code != 0:
                return None

            # Copy cached template to project directory
            result = await session.sandbox.run_command(
                f"cp -a {TEMPLATE_CACHE_DIR}/{template} /home/user/app",
                timeout=30,
            )
            if result.exit_code != 0:
                logger.warning(
                    "app_cached_scaffold_copy_failed",
                    template=template,
                    exit_code=result.exit_code,
                    stderr=result.stderr,
                )
                return None

            logger.info(
                "app_cached_scaffold_hit",
                template=template,
                sandbox_id=session.sandbox_id,
            )
            return {
                "success": True,
                "template": template,
                "project_dir": "/home/user/app",
                "message": (
                    f"Project scaffolded from cache with"
                    f" {APP_TEMPLATES[template]['name']} template"
                ),
                "cached": True,
            }
        except Exception as e:
            logger.warning("app_cached_scaffold_error", template=template, error=str(e))
            return None

    async def scaffold_project(
        self,
        session: AppSandboxSession,
        template: str | None = None,
    ) -> dict[str, Any]:
        """Scaffold a new project in the sandbox.

        Args:
            session: Active sandbox session
            template: Template to use (defaults to session template)

        Returns:
            Dict with scaffold result info
        """
        template = template or session.template
        template_config = APP_TEMPLATES.get(template)

        if not template_config:
            return {
                "success": False,
                "error": f"Unknown template: {template}",
            }

        logger.info(
            "app_scaffold_starting",
            template=template,
            sandbox_id=session.sandbox_id,
        )

        try:
            # Ensure project parent directory exists (may not exist in all images)
            await session.sandbox.run_command("mkdir -p /home/user", timeout=10)

            # Try fast path: copy from pre-cached template
            cached_result = await self._try_cached_scaffold(session, template)
            if cached_result is not None:
                session.template = template
                session.project_dir = "/home/user/app"
                session.last_health_check = time.monotonic()
                logger.info(
                    "app_scaffold_completed",
                    template=template,
                    sandbox_id=session.sandbox_id,
                    cached=True,
                )
                return cached_result

            # Slow path: live scaffold from npm
            result = await session.sandbox.run_command(
                template_config["scaffold_cmd"],
                timeout=180,  # 3 minutes for scaffolding (budget: 180+120+60 < 600s skill limit)
                cwd="/home/user",
            )

            if result.exit_code != 0:
                logger.error(
                    "app_scaffold_failed",
                    template=template,
                    exit_code=result.exit_code,
                    stderr=result.stderr,
                )
                return {
                    "success": False,
                    "error": result.stderr or "Scaffold command failed",
                    "exit_code": result.exit_code,
                }

            session.template = template
            session.project_dir = "/home/user/app"

            logger.info(
                "app_scaffold_completed",
                template=template,
                sandbox_id=session.sandbox_id,
            )

            # Refresh health-check timestamp — scaffold ran commands
            # successfully, so the sandbox is known to be alive.
            session.last_health_check = time.monotonic()

            return {
                "success": True,
                "template": template,
                "project_dir": session.project_dir,
                "message": f"Project scaffolded with {template_config['name']} template",
            }

        except Exception as e:
            logger.error("app_scaffold_error", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    async def start_dev_server(
        self,
        session: AppSandboxSession,
        custom_command: str | None = None,
        port: int | None = None,
        max_wait_seconds: int = 60,
    ) -> dict[str, Any]:
        """Start the development server and return the preview URL.

        Args:
            session: Active sandbox session
            custom_command: Custom start command (uses template default if not provided)
            port: Custom port (uses template default if not provided)
            max_wait_seconds: Maximum time to wait for server to be ready (default 60s)

        Returns:
            Dict with server info including preview URL
        """
        template_config = APP_TEMPLATES.get(session.template, APP_TEMPLATES["react"])
        start_cmd = custom_command or template_config["start_cmd"]

        # Reject custom commands containing shell metacharacters to prevent injection
        if custom_command:
            if any(ch in custom_command for ch in _SHELL_METACHARACTERS):
                raise ValueError(
                    "Custom command contains disallowed shell"
                    " metacharacters: avoid using ; | & $ or backticks"
                )

        raw_port = port or template_config["port"]
        # Validate port is an integer to prevent command injection
        server_port = int(raw_port)
        if not (1 <= server_port <= 65535):
            raise ValueError(f"Invalid port number: {server_port}")

        logger.info(
            "app_dev_server_starting",
            command=start_cmd,
            port=server_port,
            sandbox_id=session.sandbox_id,
        )

        try:
            # Stop any existing process
            if session.app_process and session.app_process.is_running:
                await self.stop_dev_server(session)

            # Get the raw URL for the port (before starting server)
            from app.config import settings

            raw_url = await session.sandbox.get_host_url(server_port)
            session.internal_url = raw_url  # Store for proxy use

            # For BoxLite, route through backend proxy so iframe doesn't need
            # direct Docker port access. E2B URLs are publicly accessible.
            if settings.sandbox_provider == "boxlite":
                preview_url = f"/api/v1/sandbox/app/{session.sandbox_id}/"
            else:
                preview_url = raw_url

            # User-friendly display URL (what the dev server looks like inside the sandbox)
            display_url = f"http://localhost:{server_port}"

            # Start the dev server using E2B's background mode
            # The background=True flag runs the command asynchronously
            logger.info(
                "app_dev_server_launching",
                command=start_cmd,
                sandbox_id=session.sandbox_id,
            )

            # Write the start command to a temp script to avoid shell injection
            # from unescaped user-provided custom_command values.
            # The script records its PID to /tmp/dev_server.pid for reliable cleanup.
            script_path = "/tmp/_dev_server_start.sh"
            await session.sandbox.write_file(
                script_path,
                "#!/bin/bash\n"
                "echo $$ > /tmp/dev_server.pid\n"
                f"{start_cmd} > /tmp/dev_server.log 2>&1\n",
            )
            await session.sandbox.run_command(f"chmod +x {script_path}", timeout=5)

            # Use setsid to create a new session so the process survives after return
            background_cmd = f"setsid {script_path} &"
            result = await session.sandbox.run_command(
                background_cmd,
                timeout=30,
                cwd="/home/user",
            )

            logger.info(
                "app_dev_server_background_started",
                exit_code=result.exit_code,
                stdout=result.stdout[:200] if result.stdout else "",
                stderr=result.stderr[:200] if result.stderr else "",
                sandbox_id=session.sandbox_id,
            )

            # Read the PID written by the start script
            dev_server_pid: int | None = None
            try:
                pid_result = await session.sandbox.run_command(
                    "cat /tmp/dev_server.pid 2>/dev/null",
                    timeout=5,
                )
                if pid_result.exit_code == 0 and pid_result.stdout.strip().isdigit():
                    dev_server_pid = int(pid_result.stdout.strip())
                    logger.info(
                        "app_dev_server_pid_captured",
                        pid=dev_server_pid,
                        sandbox_id=session.sandbox_id,
                    )
            except Exception as e:
                logger.debug("app_dev_server_pid_read_failed", error=str(e))

            # Store process info
            session.app_process = AppProcess(
                pid=dev_server_pid,
                command=start_cmd,
                is_running=True,
            )
            session.preview_url = preview_url

            # Give the server a moment to initialize before polling
            await asyncio.sleep(2)

            # Poll to check if the server is actually listening on the port
            server_ready = False
            poll_interval = 2  # seconds between checks
            max_attempts = max_wait_seconds // poll_interval

            logger.info(
                "app_dev_server_waiting_for_port",
                port=server_port,
                max_wait_seconds=max_wait_seconds,
                sandbox_id=session.sandbox_id,
            )

            for attempt in range(max_attempts):
                # Check if something is listening on the port using multiple methods
                # Try: 1) curl, 2) /dev/tcp (bash built-in), 3) lsof, 4) ss
                port_check_script = f"""
if curl -s --connect-timeout 2 http://localhost:{server_port} >/dev/null 2>&1; then
    echo 'PORT_OPEN'
elif (echo >/dev/tcp/localhost/{server_port}) 2>/dev/null; then
    echo 'PORT_OPEN'
elif lsof -i:{server_port} -sTCP:LISTEN >/dev/null 2>&1; then
    echo 'PORT_OPEN'
elif ss -tlnp | grep -q ':{server_port} ' 2>/dev/null; then
    echo 'PORT_OPEN'
else
    echo 'PORT_CLOSED'
fi
"""
                check_result = await session.sandbox.run_command(
                    port_check_script,
                    timeout=10,
                    cwd="/home/user",
                )

                if check_result.stdout and "PORT_OPEN" in check_result.stdout:
                    server_ready = True
                    logger.info(
                        "app_dev_server_port_ready",
                        port=server_port,
                        attempt=attempt + 1,
                        sandbox_id=session.sandbox_id,
                    )
                    break

                # Also check if the process crashed by looking at the log
                if attempt > 0 and attempt % 5 == 0:
                    log_result = await session.sandbox.run_command(
                        "tail -20 /tmp/dev_server.log 2>/dev/null || echo 'No log yet'",
                        timeout=5,
                        cwd="/home/user",
                    )
                    logger.debug(
                        "app_dev_server_log_check",
                        attempt=attempt + 1,
                        log_tail=log_result.stdout[:500] if log_result.stdout else "empty",
                    )

                await asyncio.sleep(poll_interval)

            if not server_ready:
                # Get the server log to help debug
                log_result = await session.sandbox.run_command(
                    "cat /tmp/dev_server.log 2>/dev/null || echo 'No log available'",
                    timeout=5,
                    cwd="/home/user",
                )
                error_log = log_result.stdout[:2000] if log_result.stdout else "No log"

                logger.warning(
                    "app_dev_server_not_ready",
                    port=server_port,
                    max_wait_seconds=max_wait_seconds,
                    server_log=error_log,
                    sandbox_id=session.sandbox_id,
                )

                # Still return the URL, but indicate the server might not be ready
                return {
                    "success": True,  # Sandbox is running, just server might be slow
                    "preview_url": session.preview_url,
                    "display_url": display_url,
                    "port": server_port,
                    "command": start_cmd,
                    "message": (
                        f"Dev server starting at {display_url} "
                        "(may take a moment to be ready)"
                    ),
                    "warning": "Server port not yet responding. It may still be starting up.",
                    "server_log": error_log,
                }

            logger.info(
                "app_dev_server_started",
                preview_url=session.preview_url,
                sandbox_id=session.sandbox_id,
            )

            return {
                "success": True,
                "preview_url": session.preview_url,
                "display_url": display_url,
                "port": server_port,
                "command": start_cmd,
                "message": f"Dev server running at {display_url}",
            }

        except Exception as e:
            logger.error("app_dev_server_start_failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    async def stop_dev_server(
        self,
        session: AppSandboxSession,
    ) -> dict[str, Any]:
        """Stop the running development server.

        Args:
            session: Active sandbox session

        Returns:
            Dict with stop result
        """
        if not session.app_process or not session.app_process.is_running:
            return {
                "success": True,
                "message": "No server running",
            }

        try:
            pid = session.app_process.pid
            pid_killed = False

            # Primary method: kill by PID (and its process group) if available
            if pid is not None:
                # Kill the process group rooted at the PID (negative PID targets the group)
                kill_result = await session.sandbox.run_command(
                    f"kill -- -{pid} 2>/dev/null; kill {pid} 2>/dev/null; echo $?",
                    timeout=5,
                )
                pid_killed = True
                logger.info(
                    "app_dev_server_pid_kill_attempted",
                    pid=pid,
                    result=kill_result.stdout.strip() if kill_result.stdout else "",
                    sandbox_id=session.sandbox_id,
                )

            # Fallback: kill by port and process name patterns
            if not pid_killed:
                template_config = APP_TEMPLATES.get(session.template, {})
                port = template_config.get("port", 3000)

                kill_commands = [
                    f"fuser -k {port}/tcp 2>/dev/null || true",
                    "pkill -f 'vite' || true",
                    "pkill -f 'npm run dev' || true",
                    f"pkill -f 'node.*{port}' || true",
                    f"pkill -f 'python.*{port}' || true",
                ]

                for cmd in kill_commands:
                    await session.sandbox.run_command(cmd, timeout=5)

            # Clean up the PID file
            await session.sandbox.run_command(
                "rm -f /tmp/dev_server.pid 2>/dev/null || true",
                timeout=5,
            )

            session.app_process.is_running = False
            session.app_process.pid = None
            session.preview_url = None

            logger.info(
                "app_dev_server_stopped",
                sandbox_id=session.sandbox_id,
            )

            return {
                "success": True,
                "message": "Dev server stopped",
            }

        except Exception as e:
            logger.error("app_dev_server_stop_failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    @staticmethod
    def _resolve_safe_path(path: str, base_dir: str) -> str:
        """Resolve a path and verify it stays within the base directory.

        Args:
            path: File path (relative or absolute)
            base_dir: Base directory that the resolved path must stay within

        Returns:
            Resolved absolute path

        Raises:
            ValueError: If the resolved path escapes the base directory
        """
        if not path.startswith("/"):
            full_path = os.path.join(base_dir, path)
        else:
            full_path = path
        resolved = os.path.normpath(full_path)
        # Ensure the resolved path is within the base directory
        norm_base = os.path.normpath(base_dir)
        if not resolved.startswith(norm_base + os.sep) and resolved != norm_base:
            raise ValueError(f"Path traversal denied: {path!r} resolves outside {base_dir}")
        return resolved

    async def write_file(
        self,
        session: AppSandboxSession,
        path: str,
        content: str,
    ) -> dict[str, Any]:
        """Write a file to the sandbox.

        Args:
            session: Active sandbox session
            path: File path (relative to project dir or absolute)
            content: File content

        Returns:
            Dict with write result
        """
        path = self._resolve_safe_path(path, session.project_dir)

        # Use shared file operations
        return await file_operations.write_file(session.sandbox, path, content, is_binary=False)

    async def read_file(
        self,
        session: AppSandboxSession,
        path: str,
    ) -> dict[str, Any]:
        """Read a file from the sandbox.

        Args:
            session: Active sandbox session
            path: File path (relative to project dir or absolute)

        Returns:
            Dict with file content
        """
        path = self._resolve_safe_path(path, session.project_dir)

        # Use shared file operations
        return await file_operations.read_file(session.sandbox, path)

    async def list_files(
        self,
        session: AppSandboxSession,
        path: str = "",
    ) -> dict[str, Any]:
        """List files in a directory.

        Args:
            session: Active sandbox session
            path: Directory path (relative to project dir or absolute)

        Returns:
            Dict with file listing
        """
        if not path:
            path = session.project_dir
        else:
            path = self._resolve_safe_path(path, session.project_dir)

        # Use shared file operations
        return await file_operations.list_directory(session.sandbox, path)

    async def install_dependencies(
        self,
        session: AppSandboxSession,
        packages: list[str],
        package_manager: Literal["npm", "pip"] = "npm",
    ) -> dict[str, Any]:
        """Install dependencies in the sandbox.

        Args:
            session: Active sandbox session
            packages: List of packages to install
            package_manager: Package manager to use (npm or pip)

        Returns:
            Dict with installation result
        """
        if not packages:
            return {
                "success": True,
                "message": "No packages to install",
            }

        # Validate package names to prevent shell injection
        for pkg in packages:
            if not _PACKAGE_NAME_RE.match(pkg):
                return {
                    "success": False,
                    "error": f"Invalid package name: {pkg!r}",
                }

        packages_str = " ".join(shlex.quote(pkg) for pkg in packages)

        if package_manager == "npm":
            cmd = f"cd {session.project_dir} && npm install {packages_str}"
        elif package_manager == "pip":
            cmd = f"pip install {packages_str}"
        else:
            return {
                "success": False,
                "error": f"Unknown package manager: {package_manager}",
            }

        logger.info(
            "app_installing_deps",
            packages=packages,
            manager=package_manager,
            sandbox_id=session.sandbox_id,
        )

        try:
            result = await session.sandbox.run_command(cmd, timeout=120)  # 2 minutes (budget: 180+120+60 < 600s skill limit)

            if result.exit_code != 0:
                return {
                    "success": False,
                    "error": result.stderr or "Installation failed",
                    "stdout": result.stdout,
                }

            logger.info(
                "app_deps_installed",
                packages=packages,
                sandbox_id=session.sandbox_id,
            )

            return {
                "success": True,
                "packages": packages,
                "message": f"Installed {len(packages)} package(s)",
            }

        except Exception as e:
            logger.error("app_install_deps_failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    async def run_command(
        self,
        session: AppSandboxSession,
        command: str,
        timeout: int = 60,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Run an arbitrary command in the sandbox.

        Args:
            session: Active sandbox session
            command: Command to run
            timeout: Command timeout in seconds
            cwd: Working directory (defaults to project dir)

        Returns:
            Dict with command result
        """
        working_dir = cwd or session.project_dir

        try:
            result = await session.sandbox.run_command(
                command,
                timeout=timeout,
                cwd=working_dir,
            )

            return {
                "success": result.exit_code == 0,
                "exit_code": result.exit_code,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
            }

        except Exception as e:
            logger.error("app_run_command_failed", command=command, error=str(e))
            return {
                "success": False,
                "error": str(e),
            }

    async def get_session_by_sandbox_id(self, sandbox_id: str) -> AppSandboxSession | None:
        """Look up a session by its sandbox_id (used by proxy endpoint).

        Args:
            sandbox_id: The sandbox identifier

        Returns:
            AppSandboxSession if found, None otherwise
        """
        async with self._session_lock:
            for session in self._sessions.values():
                if session.sandbox_id == sandbox_id:
                    session.touch()
                    return session
        return None

    async def get_session(
        self,
        user_id: str | None = None,
        task_id: str | None = None,
    ) -> AppSandboxSession | None:
        """Get an existing sandbox session without creating one.

        Args:
            user_id: User identifier
            task_id: Task identifier

        Returns:
            AppSandboxSession if exists and valid, None otherwise
        """
        session_key = self.make_session_key(user_id, task_id)

        async with self._session_lock:
            session = self._sessions.get(session_key)
            if session and not session.is_expired and await self._is_sandbox_healthy(session):
                session.touch()
                return session
            return None

    async def _is_sandbox_healthy(self, session: AppSandboxSession) -> bool:
        """Check if a sandbox session is still healthy and responsive.

        Skips the actual health check command if the last successful check
        was less than HEALTH_CHECK_SKIP_SECONDS ago.

        Args:
            session: The session to check

        Returns:
            True if sandbox is healthy, False otherwise
        """
        if not session.sandbox:
            return False

        # Skip health check if last successful check was recent
        now = time.monotonic()
        recently_checked = (
            session.last_health_check > 0
            and (now - session.last_health_check) < HEALTH_CHECK_SKIP_SECONDS
        )
        if recently_checked:
            return True

        try:
            # Perform a lightweight health check
            result = await session.sandbox.run_command(
                "echo 'health_check'",
                timeout=5,
            )
            healthy = result.exit_code == 0 and "health_check" in result.stdout
            if healthy:
                session.last_health_check = now
            return healthy
        except Exception as e:
            self._health_check_failures += 1
            logger.warning(
                "app_sandbox_health_check_failed",
                session_key=session.session_key,
                sandbox_id=session.sandbox_id,
                error=str(e),
            )
            return False

    async def save_snapshot(
        self,
        user_id: str | None = None,
        task_id: str | None = None,
        paths: list[str] | None = None,
    ) -> dict | None:
        """Save a snapshot of the app sandbox workspace.

        Args:
            user_id: User identifier
            task_id: Task identifier
            paths: Paths to include in snapshot (uses defaults if None)

        Returns:
            Snapshot metadata dict, or None if failed
        """
        session = await self.get_session(user_id=user_id, task_id=task_id)
        if not session:
            return None

        from app.services.snapshot_service import save_snapshot

        return await save_snapshot(
            runtime=session.sandbox,
            user_id=user_id or "anonymous",
            task_id=task_id or "default",
            sandbox_type="app",
            paths=paths,
        )

    async def restore_snapshot(
        self,
        user_id: str | None = None,
        task_id: str | None = None,
        snapshot_id: str | None = None,
    ) -> bool:
        """Restore a snapshot into the app sandbox.

        Args:
            user_id: User identifier
            task_id: Task identifier
            snapshot_id: Specific snapshot ID (latest if None)

        Returns:
            True if restore succeeded
        """
        session = await self.get_session(user_id=user_id, task_id=task_id)
        if not session:
            return False

        from app.services.snapshot_service import restore_snapshot

        return await restore_snapshot(
            runtime=session.sandbox,
            user_id=user_id or "anonymous",
            task_id=task_id or "default",
            sandbox_type="app",
            snapshot_id=snapshot_id,
        )

    async def cleanup_session(
        self,
        user_id: str | None = None,
        task_id: str | None = None,
    ) -> bool:
        """Explicitly clean up a sandbox session.

        Args:
            user_id: User identifier
            task_id: Task identifier

        Returns:
            True if session was cleaned up, False if not found
        """
        session_key = self.make_session_key(user_id, task_id)

        async with self._session_lock:
            return await self._cleanup_session_internal(session_key)

    async def _cleanup_session_internal(self, session_key: str) -> bool:
        """Internal cleanup method (assumes lock is held).

        Auto-saves a snapshot before killing the sandbox. On kill failure
        the session is still removed to avoid infinite retry, but an ERROR
        is logged indicating the sandbox may be leaked.

        Args:
            session_key: Session key to clean up

        Returns:
            True if session was cleaned up, False if not found
        """
        session = self._sessions.get(session_key)
        if session is None:
            return False

        # Auto-snapshot before cleanup
        try:
            # Parse user_id and task_id from session key (format: "app:user_id:task_id")
            parts = session_key.split(":", 2)
            user_id = parts[1] if len(parts) > 1 else "anonymous"
            task_id = parts[2] if len(parts) > 2 else "default"

            from app.services.snapshot_service import save_snapshot

            await save_snapshot(
                runtime=session.sandbox,
                user_id=user_id,
                task_id=task_id,
                sandbox_type="app",
            )
        except Exception as e:
            logger.debug(
                "app_auto_snapshot_failed",
                session_key=session_key,
                error=str(e),
            )

        # Reclaim the host port allocated for this session (if any)
        if session.sandbox and hasattr(session.sandbox, '_port_map'):
            for host_port in getattr(session.sandbox, '_port_map', {}).values():
                self._release_host_port(host_port)

        try:
            await session.sandbox.kill()
            self._total_cleaned += 1
            logger.info(
                "app_sandbox_session_cleaned",
                session_key=session_key,
                sandbox_id=session.sandbox_id,
            )
        except Exception as e:
            # Still remove from dict to avoid infinite retry, but log at ERROR
            # level so operators know the sandbox may be leaked.
            self._total_cleaned += 1
            logger.error(
                "app_sandbox_session_cleanup_failed_sandbox_may_be_leaked",
                session_key=session_key,
                sandbox_id=session.sandbox_id,
                error=str(e),
            )
        finally:
            self._sessions.pop(session_key, None)

        return True

    async def cleanup_expired(self) -> int:
        """Clean up all expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        cleaned = 0

        async with self._session_lock:
            expired_keys = [key for key, session in self._sessions.items() if session.is_expired]

            for key in expired_keys:
                if await self._cleanup_session_internal(key):
                    cleaned += 1

        if cleaned > 0:
            logger.info("app_expired_sessions_cleaned", count=cleaned)

        return cleaned

    async def cleanup_all(self) -> int:
        """Clean up all sessions.

        Returns:
            Number of sessions cleaned up
        """
        cleaned = 0

        async with self._session_lock:
            keys = list(self._sessions.keys())
            for key in keys:
                if await self._cleanup_session_internal(key):
                    cleaned += 1

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        logger.info("app_all_sessions_cleaned", count=cleaned)
        return cleaned

    def _ensure_cleanup_task(self) -> None:
        """Ensure the background cleanup task is running."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """Background loop that periodically cleans up expired sessions."""
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("app_cleanup_loop_error", error=str(e))

    @property
    def active_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self._sessions)

    def get_metrics(self) -> dict[str, Any]:
        """Get sandbox manager metrics.

        Returns:
            Dict with metrics
        """
        return {
            "active_sessions": len(self._sessions),
            "total_created": self._total_created,
            "total_cleaned": self._total_cleaned,
            "total_reused": self._total_reused,
            "health_check_failures": self._health_check_failures,
        }

    def get_available_templates(self) -> dict[str, dict[str, Any]]:
        """Get available app templates.

        Returns:
            Dict of template name to template info
        """
        return {
            name: {"name": config["name"], "port": config["port"]}
            for name, config in APP_TEMPLATES.items()
        }


# Singleton accessor
def get_app_sandbox_manager() -> AppSandboxManager:
    """Get the global AppSandboxManager instance.

    Returns:
        AppSandboxManager singleton
    """
    return AppSandboxManager.get_instance()
