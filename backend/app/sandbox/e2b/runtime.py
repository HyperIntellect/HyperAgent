"""E2B Runtime adapter.

Wraps E2B's AsyncSandbox to satisfy the SandboxRuntime protocol.
"""

from e2b import AsyncSandbox

from app.sandbox.runtime import CommandResult, OutputCallback, SandboxRuntime


class E2BRuntime:
    """Adapter wrapping AsyncSandbox to satisfy SandboxRuntime protocol.

    Maps protocol methods to E2B SDK calls:
    - run_command() -> sandbox.commands.run()
    - read_file()   -> sandbox.files.read()
    - write_file()  -> sandbox.files.write()
    - get_host_url() -> sandbox.get_host()
    - kill()        -> sandbox.kill()
    """

    def __init__(self, sandbox: AsyncSandbox) -> None:
        self._sandbox = sandbox

    @property
    def sandbox_id(self) -> str:
        """Get the E2B sandbox ID."""
        return self._sandbox.sandbox_id

    async def run_command(
        self,
        command: str,
        timeout: int = 60,
        cwd: str | None = None,
        on_stdout: OutputCallback | None = None,
        on_stderr: OutputCallback | None = None,
    ) -> CommandResult:
        """Run a command via E2B sandbox."""
        kwargs: dict = {"timeout": timeout}
        if cwd is not None:
            kwargs["cwd"] = cwd
        if on_stdout is not None:
            kwargs["on_stdout"] = on_stdout
        if on_stderr is not None:
            kwargs["on_stderr"] = on_stderr
        result = await self._sandbox.commands.run(command, **kwargs)
        return CommandResult(
            exit_code=result.exit_code,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    async def read_file(
        self,
        path: str,
        format: str = "text",
    ) -> bytes | str:
        """Read a file via E2B sandbox."""
        return await self._sandbox.files.read(path, format=format)

    async def write_file(
        self,
        path: str,
        content: bytes | str,
    ) -> None:
        """Write a file via E2B sandbox."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        await self._sandbox.files.write(path, content)

    async def get_host_url(self, port: int) -> str:
        """Get the public host URL for a port via E2B.

        Returns a full URL with scheme, e.g. "https://sandbox-id-5173.e2b.dev".
        """
        return f"https://{self._sandbox.get_host(port)}"

    async def save_snapshot(
        self,
        paths: list[str],
        snapshot_id: str,
    ) -> bytes:
        """Tar specified paths and return the archive bytes."""
        import shlex

        paths_str = " ".join(shlex.quote(p) for p in paths)
        archive_path = f"/tmp/snapshot_{snapshot_id}.tar.gz"

        # Create tar archive of the specified paths (ignore missing paths)
        result = await self.run_command(
            f"tar czf {shlex.quote(archive_path)} {paths_str} 2>/dev/null || true",
            timeout=120,
        )

        # Read the archive as bytes
        data = await self.read_file(archive_path, format="bytes")
        if isinstance(data, (bytes, bytearray)):
            data = bytes(data)
        else:
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
        import shlex

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
        """Kill the E2B sandbox."""
        await self._sandbox.kill()

    @property
    def raw_sandbox(self) -> AsyncSandbox:
        """Access the underlying AsyncSandbox for E2B-specific operations.

        Use sparingly — prefer protocol methods for portability.
        """
        return self._sandbox


# Ensure E2BRuntime satisfies the protocol at import time
assert isinstance(
    E2BRuntime.__new__(E2BRuntime), SandboxRuntime
), "E2BRuntime must satisfy the SandboxRuntime protocol"
