"""E2B Code Executor.

E2B-specific implementation of the code execution sandbox executor.
Wraps AsyncSandbox for code execution, file uploads, package installation,
and image capture.
"""

import base64
import re
import shlex
from io import BytesIO
from typing import Any, Literal

from e2b import AsyncSandbox

from app.config import settings
from app.core.logging import get_logger
from app.middleware.circuit_breaker import CircuitBreakerOpen, get_e2b_breaker
from app.sandbox.base_code_executor import BaseCodeExecutor
from app.sandbox.code_utils import detect_required_packages, inject_python_imports
from app.sandbox.e2b.runtime import E2BRuntime
from app.sandbox.runtime import SandboxRuntime

logger = get_logger(__name__)


class E2BSandboxExecutor(BaseCodeExecutor):
    """Manages E2B sandbox creation, file uploads, and code execution."""

    def __init__(
        self,
        api_key: str | None = None,
        template_id: str | None = None,
        timeout: int | None = None,
    ):
        """Initialize E2B executor.

        Args:
            api_key: E2B API key (defaults to settings.e2b_api_key)
            template_id: Custom E2B template ID (defaults to settings.e2b_template_id)
            timeout: Sandbox timeout in seconds (defaults to settings.e2b_code_timeout)
        """
        self.api_key = api_key or settings.e2b_api_key
        self.template_id = template_id or (
            settings.e2b_template_id if hasattr(settings, "e2b_template_id") else None
        )
        self.timeout = timeout or settings.e2b_code_timeout
        self.sandbox: AsyncSandbox | None = None
        self._runtime: E2BRuntime | None = None

    @property
    def sandbox_id(self) -> str | None:
        """Get the sandbox ID."""
        if self.sandbox:
            return self.sandbox.sandbox_id
        if self._runtime:
            return self._runtime.sandbox_id
        return None

    def get_runtime(self) -> SandboxRuntime:
        """Get the E2BRuntime wrapping the underlying sandbox."""
        if self._runtime is not None:
            return self._runtime
        if not self.sandbox:
            raise RuntimeError("Sandbox not created. Call create_sandbox() first.")
        self._runtime = E2BRuntime(self.sandbox)
        return self._runtime

    def set_runtime(self, runtime: SandboxRuntime) -> None:
        """Set an external SandboxRuntime for this executor.

        Args:
            runtime: Pre-existing SandboxRuntime to use
        """
        self._runtime = runtime

    async def create_sandbox(self) -> str:
        """Create E2B sandbox.

        Returns:
            Sandbox ID

        Raises:
            ValueError: If API key not configured
            CircuitBreakerOpen: If E2B service is unavailable
        """
        if not self.api_key:
            raise ValueError("E2B API key not configured. Set E2B_API_KEY environment variable.")

        sandbox_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout,
        }

        # Use custom template if configured (significantly faster startup)
        if self.template_id:
            sandbox_kwargs["template"] = self.template_id
            logger.info("using_e2b_template", template_id=self.template_id)

        breaker = get_e2b_breaker()

        try:
            async with breaker.call():
                self.sandbox = await AsyncSandbox.create(**sandbox_kwargs)
            self._runtime = E2BRuntime(self.sandbox)
            logger.info("e2b_sandbox_created", sandbox_id=self.sandbox.sandbox_id)
            return self.sandbox.sandbox_id

        except CircuitBreakerOpen as e:
            logger.warning(
                "e2b_sandbox_circuit_open",
                service="e2b",
                retry_after=e.retry_after,
            )
            raise
        except Exception as e:
            logger.error("e2b_sandbox_creation_failed", error=str(e))
            raise

    async def upload_file(
        self,
        file_data: BytesIO | bytes,
        filename: str,
    ) -> None:
        """Upload a file to the sandbox."""
        if not self.sandbox:
            raise RuntimeError("Sandbox not created. Call create_sandbox() first.")

        if isinstance(file_data, BytesIO):
            content = file_data.getvalue()
        else:
            content = file_data

        await self.sandbox.files.write(filename, content)
        logger.info("file_uploaded_to_sandbox", filename=filename, size=len(content))

    async def install_packages(
        self,
        packages: list[str],
        package_manager: Literal["pip", "npm"] = "pip",
        timeout: int = 120,
    ) -> tuple[bool, str, str]:
        """Install packages in the sandbox."""
        if not self.sandbox:
            raise RuntimeError("Sandbox not created. Call create_sandbox() first.")

        # Validate package names to prevent command injection
        pkg_pattern = re.compile(r'^[a-zA-Z0-9._-]+([<>=!~]+[a-zA-Z0-9.*]+)?$')
        for pkg in packages:
            if not pkg_pattern.match(pkg):
                raise ValueError(f"Invalid package name: {pkg}")

        if package_manager == "pip":
            packages_str = " ".join(shlex.quote(p) for p in packages)
            cmd = f"pip install -q {packages_str}"
        elif package_manager == "npm":
            packages_str = " ".join(shlex.quote(p) for p in packages)
            cmd = f"npm install -g {packages_str}"
        else:
            raise ValueError(f"Unsupported package manager: {package_manager}")

        logger.info("installing_packages", manager=package_manager, packages=packages)

        result = await self.sandbox.commands.run(cmd, timeout=timeout)

        success = result.exit_code == 0
        if success:
            logger.info("packages_installed", manager=package_manager, count=len(packages))
        else:
            logger.warning(
                "package_installation_failed",
                manager=package_manager,
                exit_code=result.exit_code,
                stderr=result.stderr,
            )

        return success, result.stdout or "", result.stderr or ""

    async def execute_code(
        self,
        code: str,
        language: Literal["python", "javascript", "typescript", "bash"] = "python",
        timeout: int = 180,
        auto_install_packages: bool = True,
    ) -> dict[str, Any]:
        """Execute code in the sandbox."""
        if not self.sandbox:
            raise RuntimeError("Sandbox not created. Call create_sandbox() first.")

        # Write code to file and determine execution command
        if language in ("python", "py"):
            # Auto-detect and install required packages (if not using custom template)
            if auto_install_packages and not self.template_id:
                required_packages = detect_required_packages(code)
                if required_packages:
                    logger.info("auto_installing_packages", packages=required_packages)
                    success, stdout, stderr = await self.install_packages(
                        required_packages, package_manager="pip"
                    )
                    if not success:
                        logger.warning(
                            "auto_package_installation_warning",
                            packages=required_packages,
                            stderr=stderr[:500] if stderr else None,
                        )

            script_path = "/home/user/script.py"
            # Inject common imports if not present for Python data analysis
            code = inject_python_imports(code)
            await self.sandbox.files.write(script_path, code.encode("utf-8"))
            cmd = f"python3 {script_path}"
        elif language in ("javascript", "js"):
            script_path = "/home/user/script.js"
            await self.sandbox.files.write(script_path, code.encode("utf-8"))
            cmd = f"node {script_path}"
        elif language in ("typescript", "ts"):
            script_path = "/home/user/script.ts"
            await self.sandbox.files.write(script_path, code.encode("utf-8"))
            # Install ts-node if needed
            await self.sandbox.commands.run(
                "npm install -g ts-node typescript 2>/dev/null || true",
                timeout=60,
            )
            cmd = f"ts-node {script_path}"
        elif language in ("bash", "sh", "shell"):
            script_path = "/home/user/script.sh"
            await self.sandbox.files.write(script_path, code.encode("utf-8"))
            await self.sandbox.commands.run("chmod +x /home/user/script.sh")
            cmd = script_path
        else:
            raise ValueError(f"Unsupported language: {language}")

        logger.info("executing_code", language=language, script_path=script_path)

        execution = await self.sandbox.commands.run(cmd, timeout=timeout)

        success = execution.exit_code == 0
        stdout = execution.stdout or ""
        stderr = execution.stderr or ""

        logger.info(
            "code_execution_completed",
            language=language,
            exit_code=execution.exit_code,
            success=success,
        )

        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": execution.exit_code,
        }

    async def capture_images(
        self,
        max_files: int = 10,
    ) -> list[dict[str, str]]:
        """Capture image/output files from sandbox."""
        if not self.sandbox:
            raise RuntimeError("Sandbox not created. Call create_sandbox() first.")

        images: list[dict[str, str]] = []

        # List files in /tmp for debugging
        try:
            list_result = await self.sandbox.commands.run(
                "ls -la /tmp/output* 2>/dev/null || echo 'No output files found'", timeout=10
            )
            logger.info("tmp_output_files", stdout=list_result.stdout, stderr=list_result.stderr)
        except Exception as e:
            logger.debug("failed_to_list_tmp_files", error=str(e))

        # Helper function to try reading a file as bytes
        async def try_read_file(path: str, as_bytes: bool = True) -> bytes | str | None:
            try:
                if as_bytes:
                    content = await self.sandbox.files.read(path, format="bytes")
                else:
                    content = await self.sandbox.files.read(path, format="text")
                return content if content else None
            except Exception as e:
                logger.debug("image_file_not_found", path=path, error=str(e))
                return None

        # Try to capture PNG outputs
        primary_png = await try_read_file("/tmp/output.png")
        if primary_png:
            image_data = base64.b64encode(primary_png).decode("utf-8")
            images.append(
                {
                    "data": image_data,
                    "type": "image/png",
                    "path": "/tmp/output.png",
                }
            )
            logger.info("image_captured", type="png", path="/tmp/output.png", size=len(primary_png))

        for i in range(max_files):
            png_path = f"/tmp/output_{i}.png"
            png_content = await try_read_file(png_path)
            if png_content:
                image_data = base64.b64encode(png_content).decode("utf-8")
                images.append(
                    {
                        "data": image_data,
                        "type": "image/png",
                        "path": png_path,
                    }
                )
                logger.info("image_captured", type="png", path=png_path, size=len(png_content))
            else:
                break

        # Try to capture HTML outputs
        primary_html = await try_read_file("/tmp/output.html", as_bytes=False)
        if primary_html:
            images.append(
                {
                    "data": primary_html,
                    "type": "text/html",
                    "path": "/tmp/output.html",
                }
            )
            logger.info(
                "image_captured",
                type="html",
                path="/tmp/output.html",
                size=len(primary_html),
            )

        for i in range(max_files):
            html_path = f"/tmp/output_{i}.html"
            html_content = await try_read_file(html_path, as_bytes=False)
            if html_content:
                images.append(
                    {
                        "data": html_content,
                        "type": "text/html",
                        "path": html_path,
                    }
                )
                logger.info("image_captured", type="html", path=html_path, size=len(html_content))
            else:
                break

        if images:
            logger.info("total_images_captured", count=len(images))
        else:
            logger.info("no_images_found_in_sandbox")

        return images

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        if self.sandbox:
            try:
                await self.sandbox.kill()
                logger.info("sandbox_cleaned_up", sandbox_id=self.sandbox.sandbox_id)
            except Exception as e:
                logger.warning("sandbox_cleanup_failed", error=str(e))
            finally:
                self.sandbox = None
                self._runtime = None


# Convenience functions for common use cases


async def execute_python_with_data(
    code: str,
    files: dict[str, BytesIO | bytes] | None = None,
    packages: list[str] | None = None,
    capture_images: bool = True,
) -> dict[str, Any]:
    """Execute Python code with data files and optional package installation."""
    async with E2BSandboxExecutor() as executor:
        # Upload files if provided
        if files:
            for filename, file_data in files.items():
                await executor.upload_file(file_data, filename)

        # Install packages if needed and not using template
        if packages and not executor.template_id:
            await executor.install_packages(packages, package_manager="pip")

        # Execute code
        result = await executor.execute_code(code, language="python")

        # Capture images if requested
        if capture_images:
            images = await executor.capture_images()
            result["images"] = images

        result["sandbox_id"] = executor.sandbox.sandbox_id if executor.sandbox else None

        return result


