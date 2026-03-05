"""BoxLite Runtime adapter.

Wraps boxlite.SimpleBox to satisfy the SandboxRuntime protocol for local
Docker-based sandbox execution.
"""

import asyncio
import base64
import shlex
import uuid
from pathlib import PurePosixPath as Path

import boxlite

from app.config import settings
from app.core.logging import get_logger
from app.sandbox.runtime import CommandResult, OutputCallback

logger = get_logger(__name__)


class BoxLiteRuntime:
    """Adapter wrapping boxlite.SimpleBox to satisfy SandboxRuntime protocol.

    Maps protocol methods to boxlite SDK calls:
    - run_command() -> box.exec("bash", "-c", cmd)
    - read_file()   -> exec cat (text) or base64 (binary)
    - write_file()  -> exec with heredoc (text) or base64 pipe (binary)
    - get_host_url() -> lookup pre-configured host:guest port map
    - kill()        -> box.__aexit__() (async context manager cleanup)
    """

    def __init__(self, box: "boxlite.SimpleBox", port_map: dict[int, int] | None = None) -> None:
        self._box = box
        self._id = f"boxlite-{uuid.uuid4().hex[:12]}"
        # Maps guest port -> host port
        self._port_map: dict[int, int] = port_map or {}
        self._killed: bool = False

    @property
    def sandbox_id(self) -> str:
        """Get a unique identifier for this BoxLite sandbox."""
        return self._id

    @classmethod
    async def create(
        cls,
        image: str = "python:3.12-slim",
        cpus: int | None = None,
        memory_mib: int | None = None,
        disk_size_gb: int | None = None,
        ports: dict[int, int] | None = None,
        working_dir: str | None = None,
    ) -> "BoxLiteRuntime":
        """Create a new BoxLite sandbox.

        Args:
            image: Docker image to use
            cpus: Number of CPUs
            memory_mib: Memory limit in MiB
            disk_size_gb: Disk size limit in GB
            ports: Guest port -> host port mapping
            working_dir: Working directory inside the container

        Returns:
            BoxLiteRuntime instance
        """
        _cpus = cpus or settings.boxlite_cpus
        _memory = memory_mib or settings.boxlite_memory_mib
        _disk = disk_size_gb or settings.boxlite_disk_size_gb
        _ports = ports or {}

        # Build SimpleBox kwargs
        kwargs: dict = {}
        # Only set working_dir when explicitly passed by the caller.
        # Defaulting to /home/user breaks images where that directory
        # doesn't exist (e.g. node:20-slim runs as root with /root).
        if working_dir:
            kwargs["working_dir"] = working_dir
        if _ports:
            # Convert {guest: host} dict to [(host, guest)] tuples
            kwargs["ports"] = [(host, guest) for guest, host in _ports.items()]

        box = boxlite.SimpleBox(
            image=image,
            cpus=_cpus,
            memory_mib=_memory,
            disk_size_gb=_disk,
            auto_remove=settings.boxlite_auto_remove,
            **kwargs,
        )
        await box.start()

        runtime = cls(box, port_map=_ports)
        logger.info(
            "boxlite_sandbox_created",
            sandbox_id=runtime.sandbox_id,
            image=image,
            cpus=_cpus,
            memory_mib=_memory,
            disk_size_gb=_disk,
        )
        return runtime

    async def run_command(
        self,
        command: str,
        timeout: int = 60,
        cwd: str | None = None,
        on_stdout: OutputCallback | None = None,
        on_stderr: OutputCallback | None = None,
    ) -> CommandResult:
        """Run a command via BoxLite exec."""
        full_cmd = command
        if cwd:
            full_cmd = f"cd {shlex.quote(cwd)} && {command}"

        try:
            # SimpleBox.exec is async and returns ExecResult directly
            result = await asyncio.wait_for(
                self._box.exec("bash", "-c", full_cmd),
                timeout=timeout,
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            # Simulate line-by-line streaming after completion
            if on_stdout and stdout:
                for line in stdout.splitlines(keepends=True):
                    on_stdout(line)
            if on_stderr and stderr:
                for line in stderr.splitlines(keepends=True):
                    on_stderr(line)

            return CommandResult(
                exit_code=result.exit_code,
                stdout=stdout,
                stderr=stderr,
            )
        except asyncio.TimeoutError:
            return CommandResult(
                exit_code=124,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
            )

    async def read_file(
        self,
        path: str,
        format: str = "text",
    ) -> bytes | str:
        """Read a file via exec cat/base64."""
        if format == "bytes":
            result = await self.run_command(
                f"base64 {shlex.quote(path)}",
                timeout=30,
            )
            if result.exit_code != 0:
                # Fallback: use python3 if the base64 CLI is not available
                logger.debug(
                    "boxlite_base64_cmd_failed_trying_python_fallback",
                    path=path,
                    stderr=result.stderr,
                )
                py_cmd = (
                    "import base64, sys; sys.stdout.write("
                    "base64.b64encode(open(sys.argv[1],'rb')"
                    ".read()).decode())"
                )
                result = await self.run_command(
                    f'python3 -c "{py_cmd}" {shlex.quote(path)}',
                    timeout=30,
                )
                if result.exit_code != 0:
                    raise FileNotFoundError(f"Failed to read {path}: {result.stderr}")
            return base64.b64decode(result.stdout.strip())
        else:
            result = await self.run_command(
                f"cat {shlex.quote(path)}",
                timeout=30,
            )
            if result.exit_code != 0:
                raise FileNotFoundError(f"Failed to read {path}: {result.stderr}")
            return result.stdout

    async def write_file(
        self,
        path: str,
        content: bytes | str,
    ) -> None:
        """Write content to a file via exec.

        Note: parent directory creation is NOT handled here because the
        caller (file_operations.write_file) already runs ``mkdir -p``
        before invoking this method.  Keeping it only in the caller
        avoids a redundant round-trip and keeps the logic in one place
        (the caller must do it anyway for E2B which does not auto-create
        parent directories).
        """
        # Ensure parent directory exists (some images lack /home/user)
        parent_dir = str(Path(path).parent)
        if parent_dir != "/":
            await self.run_command(f"mkdir -p {shlex.quote(parent_dir)}", timeout=10)

        if isinstance(content, bytes):
            # Binary write via base64 pipe using printf to avoid echo length limits
            encoded = base64.b64encode(content).decode("ascii")
            result = await self.run_command(
                f"printf '%s' {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}",
                timeout=30,
            )
        else:
            # Text write via heredoc with random delimiter to prevent injection
            delimiter = f"BOXLITE_EOF_{uuid.uuid4().hex}"
            result = await self.run_command(
                f"cat > {shlex.quote(path)} << '{delimiter}'\n{content}\n{delimiter}",
                timeout=30,
            )

        if result.exit_code != 0:
            raise IOError(f"Failed to write {path}: {result.stderr}")

    async def get_host_url(self, port: int) -> str:
        """Get the localhost URL for a forwarded port.

        Returns a full URL with scheme, e.g. "http://localhost:10000".
        """
        if port in self._port_map:
            host_port = self._port_map[port]
        else:
            # If no explicit mapping, assume identity mapping.
            # This likely means the port was not published when the container
            # was created, so the returned URL will probably not be reachable.
            logger.warning(
                "boxlite_port_not_mapped_using_identity_fallback",
                sandbox_id=self._id,
                guest_port=port,
                available_mappings=self._port_map,
            )
            host_port = port

        return f"http://localhost:{host_port}"

    async def save_snapshot(
        self,
        paths: list[str],
        snapshot_id: str,
    ) -> bytes:
        """Tar specified paths and return the archive bytes."""
        paths_str = " ".join(shlex.quote(p) for p in paths)
        archive_path = f"/tmp/snapshot_{snapshot_id}.tar.gz"

        # Create tar archive of the specified paths (ignore missing paths)
        await self.run_command(
            f"tar czf {shlex.quote(archive_path)} {paths_str} 2>/dev/null || true",
            timeout=120,
        )

        # Read the archive as bytes
        data = await self.read_file(archive_path, format="bytes")
        if not isinstance(data, bytes):
            data = data.encode("utf-8")

        # Clean up the temp archive
        await self.run_command(f"rm -f {shlex.quote(archive_path)}", timeout=10)

        return data

    async def restore_snapshot(
        self,
        snapshot_data: bytes,
        target_path: str,
    ) -> bool:
        """Restore a tar archive to the specified target path."""
        archive_path = "/tmp/_restore_snapshot.tar.gz"

        # Write archive to sandbox
        await self.write_file(archive_path, snapshot_data)

        # Extract to target path
        await self.run_command(
            f"mkdir -p {shlex.quote(target_path)}",
            timeout=10,
        )
        result = await self.run_command(
            f"tar xzf {shlex.quote(archive_path)} -C {shlex.quote(target_path)}",
            timeout=120,
        )

        # Clean up
        await self.run_command(f"rm -f {shlex.quote(archive_path)}", timeout=10)

        return result.exit_code == 0

    async def kill(self) -> None:
        """Stop and remove the BoxLite container.

        Uses shutdown() first for a clean stop, falling back to __aexit__
        if shutdown raises. A _killed flag prevents double-cleanup.
        """
        if self._killed:
            return
        self._killed = True

        try:
            await self._box.shutdown()
            logger.info("boxlite_sandbox_stopped", sandbox_id=self._id)
        except Exception as shutdown_err:
            logger.debug(
                "boxlite_shutdown_failed_trying_aexit",
                sandbox_id=self._id,
                error=str(shutdown_err),
            )
            try:
                await self._box.__aexit__(None, None, None)
                logger.info("boxlite_sandbox_stopped_via_aexit", sandbox_id=self._id)
            except Exception as e:
                logger.warning("boxlite_sandbox_stop_failed", sandbox_id=self._id, error=str(e))

    @property
    def raw_box(self) -> "boxlite.SimpleBox":
        """Access the underlying boxlite.SimpleBox for provider-specific operations."""
        return self._box
