"""Sandbox Runtime Protocol.

Defines the low-level interface for sandbox command execution, file I/O,
port forwarding, and lifecycle management. All sandbox providers (E2B, BoxLite)
must implement this protocol.
"""

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

OutputCallback = Callable[[str], None]


@dataclass
class CommandResult:
    """Result of a command execution in a sandbox."""

    exit_code: int
    stdout: str
    stderr: str


@runtime_checkable
class SandboxRuntime(Protocol):
    """Protocol for sandbox runtime operations.

    This protocol abstracts the low-level sandbox operations that differ between
    providers (E2B, BoxLite, etc.). Higher-level code should depend on this
    protocol rather than on provider-specific classes.
    """

    @property
    def sandbox_id(self) -> str:
        """Get the unique identifier for this sandbox."""
        ...

    async def run_command(
        self,
        command: str,
        timeout: int = 60,
        cwd: str | None = None,
        on_stdout: OutputCallback | None = None,
        on_stderr: OutputCallback | None = None,
    ) -> CommandResult:
        """Run a shell command in the sandbox.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds
            cwd: Working directory for the command
            on_stdout: Optional callback invoked per stdout line/chunk
            on_stderr: Optional callback invoked per stderr line/chunk

        Returns:
            CommandResult with exit_code, stdout, stderr
        """
        ...

    async def read_file(
        self,
        path: str,
        format: str = "text",
    ) -> bytes | str:
        """Read a file from the sandbox.

        Args:
            path: File path in the sandbox
            format: "text" for string, "bytes" for binary

        Returns:
            File content as str (text) or bytes (binary)
        """
        ...

    async def write_file(
        self,
        path: str,
        content: bytes | str,
    ) -> None:
        """Write content to a file in the sandbox.

        Args:
            path: File path in the sandbox
            content: Content to write (str or bytes)
        """
        ...

    async def get_host_url(self, port: int) -> str:
        """Get the public URL for a forwarded port.

        Args:
            port: Port number in the sandbox

        Returns:
            Full URL with scheme (e.g., "https://sandbox-id-port.e2b.dev"
            or "http://localhost:10000")
        """
        ...

    async def save_snapshot(
        self,
        paths: list[str],
        snapshot_id: str,
    ) -> bytes:
        """Tar the specified paths and return the archive bytes.

        Args:
            paths: List of absolute paths in the sandbox to include
            snapshot_id: Unique identifier for this snapshot (used in tar name)

        Returns:
            Raw tar.gz bytes of the snapshot archive
        """
        ...

    async def restore_snapshot(
        self,
        snapshot_data: bytes,
        target_path: str,
    ) -> bool:
        """Restore a tar archive to the specified target path.

        Args:
            snapshot_data: Raw tar.gz bytes to restore
            target_path: Directory to extract the archive into

        Returns:
            True if restore succeeded, False otherwise
        """
        ...

    async def kill(self) -> None:
        """Terminate and clean up the sandbox."""
        ...
