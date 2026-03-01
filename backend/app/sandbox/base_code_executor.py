"""Base Code Executor ABC.

Defines the abstract interface for code execution sandbox executors.
All provider implementations (E2B, BoxLite) must extend this class.
"""

from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Literal

from app.sandbox.runtime import SandboxRuntime


class BaseCodeExecutor(ABC):
    """Abstract base class for code execution sandbox executors.

    Provides the interface that managers and tools depend on, abstracting
    the underlying sandbox provider.
    """

    @abstractmethod
    async def create_sandbox(self) -> str:
        """Create a new sandbox instance.

        Returns:
            Sandbox ID string

        Raises:
            ValueError: If required configuration is missing
        """
        ...

    @abstractmethod
    async def upload_file(
        self,
        file_data: BytesIO | bytes,
        filename: str,
    ) -> None:
        """Upload a file to the sandbox.

        Args:
            file_data: File content as BytesIO or bytes
            filename: Destination filename in sandbox
        """
        ...

    @abstractmethod
    async def install_packages(
        self,
        packages: list[str],
        package_manager: Literal["pip", "npm"] = "pip",
        timeout: int = 120,
    ) -> tuple[bool, str, str]:
        """Install packages in the sandbox.

        Args:
            packages: List of package names to install
            package_manager: Package manager to use ("pip" or "npm")
            timeout: Installation timeout in seconds

        Returns:
            Tuple of (success, stdout, stderr)
        """
        ...

    @abstractmethod
    async def execute_code(
        self,
        code: str,
        language: Literal["python", "javascript", "typescript", "bash"] = "python",
        timeout: int = 180,
        auto_install_packages: bool = True,
    ) -> dict[str, Any]:
        """Execute code in the sandbox.

        Args:
            code: Code to execute
            language: Programming language
            timeout: Execution timeout in seconds
            auto_install_packages: Whether to auto-detect and install required packages

        Returns:
            Dict with keys: success, stdout, stderr, exit_code
        """
        ...

    @abstractmethod
    async def capture_images(
        self,
        max_files: int = 10,
    ) -> list[dict[str, str]]:
        """Capture image/output files from sandbox.

        Args:
            max_files: Maximum number of numbered files to check

        Returns:
            List of dicts with keys: data, type, path
        """
        ...

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        ...

    @abstractmethod
    def get_runtime(self) -> SandboxRuntime:
        """Get the underlying SandboxRuntime for low-level operations.

        Returns:
            SandboxRuntime instance for this executor's sandbox
        """
        ...

    def set_runtime(self, runtime: SandboxRuntime) -> None:
        """Set an external SandboxRuntime for this executor.

        Used by the unified sandbox manager to share a single runtime
        across code execution and app development.

        Args:
            runtime: Pre-existing SandboxRuntime to use
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support set_runtime()"
        )

    @property
    @abstractmethod
    def sandbox_id(self) -> str | None:
        """Get the sandbox ID, or None if not yet created."""
        ...

    async def __aenter__(self):
        """Context manager entry - creates sandbox."""
        await self.create_sandbox()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleans up sandbox."""
        await self.cleanup()
