"""BoxLite Code Executor.

Local Docker-based implementation of the code execution sandbox executor.
Uses boxlite.Box for running code in isolated containers.
"""

import base64
import re
import shlex
from io import BytesIO
from typing import Any, Literal

from app.config import settings
from app.core.logging import get_logger
from app.sandbox.base_code_executor import BaseCodeExecutor
from app.sandbox.boxlite.runtime import BoxLiteRuntime
from app.sandbox.code_utils import detect_required_packages, inject_python_imports
from app.sandbox.runtime import SandboxRuntime

logger = get_logger(__name__)


class BoxLiteCodeExecutor(BaseCodeExecutor):
    """Manages BoxLite sandbox creation, file uploads, and code execution."""

    def __init__(
        self,
        image: str | None = None,
        timeout: int | None = None,
    ):
        self._image = image or settings.boxlite_code_image
        self._timeout = timeout or settings.boxlite_code_timeout
        self._runtime: BoxLiteRuntime | None = None

    @property
    def sandbox_id(self) -> str | None:
        if self._runtime:
            return self._runtime.sandbox_id
        return None

    def get_runtime(self) -> SandboxRuntime:
        if not self._runtime:
            raise RuntimeError("Sandbox not created. Call create_sandbox() first.")
        return self._runtime

    def set_runtime(self, runtime: SandboxRuntime) -> None:
        """Set an external SandboxRuntime for this executor.

        Args:
            runtime: Pre-existing SandboxRuntime to use
        """
        self._runtime = runtime

    async def create_sandbox(self) -> str:
        self._runtime = await BoxLiteRuntime.create(
            image=self._image,
            cpus=settings.boxlite_cpus,
            memory_mib=settings.boxlite_memory_mib,
        )
        logger.info(
            "boxlite_code_sandbox_created",
            sandbox_id=self._runtime.sandbox_id,
            image=self._image,
        )
        return self._runtime.sandbox_id

    async def upload_file(
        self,
        file_data: BytesIO | bytes,
        filename: str,
    ) -> None:
        if not self._runtime:
            raise RuntimeError("Sandbox not created. Call create_sandbox() first.")

        if isinstance(file_data, BytesIO):
            content = file_data.getvalue()
        else:
            content = file_data

        await self._runtime.write_file(filename, content)
        logger.info("file_uploaded_to_boxlite", filename=filename, size=len(content))

    async def install_packages(
        self,
        packages: list[str],
        package_manager: Literal["pip", "npm"] = "pip",
        timeout: int = 120,
    ) -> tuple[bool, str, str]:
        if not self._runtime:
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

        logger.info("boxlite_installing_packages", manager=package_manager, packages=packages)

        result = await self._runtime.run_command(cmd, timeout=timeout)

        success = result.exit_code == 0
        if success:
            logger.info("boxlite_packages_installed", manager=package_manager, count=len(packages))
        else:
            logger.warning(
                "boxlite_package_installation_failed",
                manager=package_manager,
                exit_code=result.exit_code,
                stderr=result.stderr,
            )

        return success, result.stdout, result.stderr

    async def execute_code(
        self,
        code: str,
        language: Literal["python", "javascript", "typescript", "bash"] = "python",
        timeout: int = 180,
        auto_install_packages: bool = True,
    ) -> dict[str, Any]:
        if not self._runtime:
            raise RuntimeError("Sandbox not created. Call create_sandbox() first.")

        if language in ("python", "py"):
            if auto_install_packages:
                required_packages = detect_required_packages(code)
                if required_packages:
                    logger.info("boxlite_auto_installing_packages", packages=required_packages)
                    await self.install_packages(required_packages, package_manager="pip")

            script_path = "/home/user/script.py"
            code = inject_python_imports(code)
            await self._runtime.write_file(script_path, code)
            cmd = f"python3 {script_path}"
        elif language in ("javascript", "js"):
            script_path = "/home/user/script.js"
            await self._runtime.write_file(script_path, code)
            cmd = f"node {script_path}"
        elif language in ("typescript", "ts"):
            script_path = "/home/user/script.ts"
            await self._runtime.write_file(script_path, code)
            await self._runtime.run_command(
                "npm install -g ts-node typescript 2>/dev/null || true",
                timeout=60,
            )
            cmd = f"ts-node {script_path}"
        elif language in ("bash", "sh", "shell"):
            script_path = "/home/user/script.sh"
            await self._runtime.write_file(script_path, code)
            await self._runtime.run_command(f"chmod +x {script_path}")
            cmd = script_path
        else:
            raise ValueError(f"Unsupported language: {language}")

        logger.info("boxlite_executing_code", language=language, script_path=script_path)

        result = await self._runtime.run_command(cmd, timeout=timeout)

        success = result.exit_code == 0
        logger.info(
            "boxlite_code_execution_completed",
            language=language,
            exit_code=result.exit_code,
            success=success,
        )

        return {
            "success": success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
        }

    async def capture_images(
        self,
        max_files: int = 10,
    ) -> list[dict[str, str]]:
        if not self._runtime:
            raise RuntimeError("Sandbox not created. Call create_sandbox() first.")

        images: list[dict[str, str]] = []

        async def try_read_file(path: str, as_bytes: bool = True) -> bytes | str | None:
            try:
                fmt = "bytes" if as_bytes else "text"
                content = await self._runtime.read_file(path, format=fmt)
                return content if content else None
            except Exception:
                return None

        # Capture PNG outputs
        primary_png = await try_read_file("/tmp/output.png")
        if primary_png:
            images.append(
                {
                    "data": base64.b64encode(primary_png).decode("utf-8"),
                    "type": "image/png",
                    "path": "/tmp/output.png",
                }
            )

        for i in range(max_files):
            png_path = f"/tmp/output_{i}.png"
            png_content = await try_read_file(png_path)
            if png_content:
                images.append(
                    {
                        "data": base64.b64encode(png_content).decode("utf-8"),
                        "type": "image/png",
                        "path": png_path,
                    }
                )
            else:
                break

        # Capture HTML outputs
        primary_html = await try_read_file("/tmp/output.html", as_bytes=False)
        if primary_html:
            images.append(
                {
                    "data": primary_html,
                    "type": "text/html",
                    "path": "/tmp/output.html",
                }
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
            else:
                break

        if images:
            logger.info("boxlite_images_captured", count=len(images))

        return images

    async def cleanup(self) -> None:
        if self._runtime:
            try:
                await self._runtime.kill()
                logger.info("boxlite_code_sandbox_cleaned_up", sandbox_id=self._runtime.sandbox_id)
            except Exception as e:
                logger.warning("boxlite_cleanup_failed", error=str(e))
            finally:
                self._runtime = None
