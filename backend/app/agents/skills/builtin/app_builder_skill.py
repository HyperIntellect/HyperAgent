"""App Builder Skill for creating and running web applications.

This skill orchestrates the full workflow of building and deploying
web applications in an isolated sandbox environment.
"""

import json
import re
from operator import add
from typing import Annotated, Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.agents import events as agent_events
from app.agents.skills.skill_base import Skill, SkillMetadata, SkillParameter, SkillState
from app.ai.llm import llm_service
from app.ai.model_tiers import ModelTier
from app.core.logging import get_logger
from app.sandbox.app_sandbox_manager import (
    APP_TEMPLATES,
    get_app_sandbox_manager,
)

logger = get_logger(__name__)


class FileSpec(BaseModel):
    """File specification for app planning."""

    path: str = Field(description="File path relative to project root (e.g., 'src/App.tsx')")
    description: str = Field(description="What this file should contain/do")


class AppPlan(BaseModel):
    """Plan for building an application."""

    template: str = Field(
        description="Template to use (react, react-ts, nextjs, vue, express, fastapi, flask, static)"
    )
    features: list[str] = Field(description="List of features to implement")
    files: list[FileSpec] = Field(
        description="List of files to create with path and description"
    )
    packages: list[str] = Field(default_factory=list, description="Additional packages to install")
    explanation: str = Field(description="Brief explanation of the app architecture")


class FileContent(BaseModel):
    """Generated file content."""

    path: str = Field(description="File path relative to project root")
    content: str = Field(description="Complete file content")
    description: str = Field(description="What this file does")


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from text if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove opening fence (```lang)
        lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return text


def _parse_json_from_text(text: str) -> dict | None:
    """Try to extract a JSON object from text."""
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try after stripping code fences
    stripped = _strip_code_fences(text)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to find JSON object in text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


class AppBuilderState(SkillState):
    """State for app builder skill execution."""

    plan: dict[str, Any] | None
    generated_files: list[dict[str, Any]]
    preview_url: str | None
    build_errors: list[str]
    current_step: str
    retry_count: int
    # Events to emit during streaming — accumulated across nodes via `add` reducer
    pending_events: Annotated[list[dict[str, Any]], add]


class AppBuilderSkill(Skill):
    """Builds and runs web applications from natural language descriptions."""

    metadata = SkillMetadata(
        id="app_builder",
        name="App Builder",
        version="1.0.0",
        description="Builds and runs web applications from descriptions. Supports React, Next.js, Vue, Express, FastAPI, Flask, and static sites. Creates a live preview URL.",
        category="automation",
        parameters=[
            SkillParameter(
                name="description",
                type="string",
                description="Description of the app to build (e.g., 'A todo list app with dark mode')",
                required=True,
            ),
            SkillParameter(
                name="template",
                type="string",
                description="Template to use: react, react-ts, nextjs, vue, express, fastapi, flask, static. If not specified, the skill will choose based on the description.",
                required=False,
                default=None,
            ),
            SkillParameter(
                name="features",
                type="array",
                description="List of specific features to include",
                required=False,
                default=[],
            ),
        ],
        output_schema={
            "type": "object",
            "properties": {
                "preview_url": {
                    "type": "string",
                    "description": "Live preview URL where the app can be viewed",
                },
                "files_created": {
                    "type": "array",
                    "description": "List of files created",
                },
                "template": {
                    "type": "string",
                    "description": "Template used",
                },
                "message": {
                    "type": "string",
                    "description": "Summary of what was built",
                },
            },
        },
        required_tools=[],
        risk_level="high",
        side_effect_level="high",
        data_sensitivity="sensitive",
        network_scope="sandbox_only",
        idempotency_hint=False,
        max_execution_time_seconds=600,  # 10 minutes for full build
        max_iterations=20,
        tags=["app", "web", "development", "react", "nextjs", "vue", "express", "fastapi"],
    )

    def create_graph(self) -> StateGraph:
        """Create the LangGraph subgraph for app building."""
        graph = StateGraph(AppBuilderState)

        async def plan_app(state: AppBuilderState) -> dict:
            """Plan the application structure and files."""
            description = state["input_params"]["description"]
            requested_template = state["input_params"].get("template")
            requested_features = state["input_params"].get("features", [])

            # Log skill entry for debugging
            logger.info(
                "app_builder_skill_started",
                description=description[:100] if description else "no description",
                user_id=state.get("user_id"),
                task_id=state.get("task_id"),
            )

            # Fresh event list per node — Annotated[..., add] accumulates across nodes
            pending_events = []
            pending_events.append(
                agent_events.stage(
                    name="plan",
                    description="Planning app structure and architecture",
                    status="running",
                )
            )

            logger.info(
                "app_builder_planning",
                description=description[:100],
                template=requested_template,
            )

            try:
                # Build prompt for planning
                templates_info = "\n".join(
                    [
                        f"- {name}: {config['name']} (port {config['port']})"
                        for name, config in APP_TEMPLATES.items()
                    ]
                )

                features_hint = ""
                if requested_features:
                    features_hint = f"\n\nRequested features: {', '.join(requested_features)}"

                template_hint = ""
                if requested_template:
                    template_hint = f"\n\nUser requested template: {requested_template}"

                prompt = f"""Plan a web application based on this description:

{description}{features_hint}{template_hint}

Available templates:
{templates_info}

Create a plan that includes:
1. The best template to use (considering the app type and requirements)
2. List of key features to implement
3. List of files to create (path and brief description)
4. Any additional packages needed

Be practical and create a working app, not just boilerplate. Focus on the core functionality described."""

                # Get LLM for planning (use MAX tier for better architecture decisions)
                llm = llm_service.get_llm_for_tier(ModelTier.MAX)
                plan: AppPlan | None = None

                # Try structured output first
                try:
                    structured_llm = llm.with_structured_output(AppPlan)
                    plan = await structured_llm.ainvoke(prompt)
                except Exception as so_err:
                    logger.warning(
                        "plan_structured_output_fallback",
                        error=str(so_err)[:200],
                    )

                # Fallback: plain text JSON generation
                if plan is None:
                    json_prompt = (
                        f"{prompt}\n\n"
                        "Respond with a JSON object:\n"
                        '{"template": "...", "features": [...], '
                        '"files": [{"path": "...", '
                        '"description": "..."}], '
                        '"packages": [...], '
                        '"explanation": "..."}'
                    )
                    resp = await llm.ainvoke(
                        [HumanMessage(content=json_prompt)]
                    )
                    text = resp.content or ""
                    if not text:
                        text = resp.additional_kwargs.get(
                            "reasoning_content", ""
                        )
                    parsed = _parse_json_from_text(text)
                    if not parsed:
                        raise ValueError(
                            "Failed to parse plan from LLM"
                        )
                    plan = AppPlan(**parsed)

                logger.info(
                    "app_builder_planned",
                    template=plan.template,
                    file_count=len(plan.files),
                    package_count=len(plan.packages),
                )

                # Mark planning as complete
                pending_events.append(
                    agent_events.stage(
                        name="plan",
                        description=f"Planned {plan.template} app with {len(plan.files)} files",
                        status="completed",
                    )
                )

                # Emit plan details as terminal output
                pending_events.append(
                    agent_events.terminal_command(
                        command=f"# App Plan: {plan.template} application",
                        cwd="/home/user",
                    )
                )
                plan_lines = [f"Template: {plan.template}"]
                if plan.features:
                    plan_lines.append(f"Features: {', '.join(plan.features)}")
                if plan.packages:
                    plan_lines.append(f"Packages: {', '.join(plan.packages)}")
                plan_lines.append(f"\nFiles to create ({len(plan.files)}):")
                for f in plan.files:
                    plan_lines.append(f"  - {f.path}: {f.description}")
                pending_events.append(
                    agent_events.terminal_output(
                        content="\n".join(plan_lines),
                        stream="stdout",
                    )
                )
                pending_events.append(
                    agent_events.terminal_complete(exit_code=0)
                )

                return {
                    "plan": plan.model_dump(),
                    "current_step": "plan",  # Return current step, route_step maps to next
                    "iterations": state.get("iterations", 0) + 1,
                    "pending_events": pending_events,
                }

            except Exception as e:
                logger.error("app_builder_plan_failed", error=str(e))
                pending_events.append(
                    agent_events.stage(
                        name="plan",
                        description="Failed to plan app structure",
                        status="failed",
                    )
                )
                return {
                    "error": f"Failed to plan app: {str(e)}",
                    "current_step": "error",
                    "iterations": state.get("iterations", 0) + 1,
                    "pending_events": pending_events,
                }

        async def scaffold_project(state: AppBuilderState) -> dict:
            """Scaffold the project using the selected template."""
            plan = state.get("plan", {})
            template = plan.get("template", "react")
            if template not in APP_TEMPLATES:
                logger.warning(
                    "app_builder_unknown_template_fallback",
                    requested=template,
                    fallback="react",
                )
                template = "react"
            user_id = state.get("user_id")
            task_id = state.get("task_id")

            # Check if sandbox is available (provider-agnostic)
            from app.sandbox.provider import is_provider_available
            available, reason = is_provider_available("app")
            if not available:
                logger.error("app_builder_sandbox_not_available", reason=reason)
                pending_events = []
                pending_events.append(
                    agent_events.stage(
                        name="scaffold",
                        description=f"Sandbox not available - {reason}",
                        status="failed",
                    )
                )
                return {
                    "error": f"Sandbox not available: {reason}",
                    "current_step": "error",
                    "iterations": state.get("iterations", 0) + 1,
                    "pending_events": pending_events,
                }

            # Fresh event list per node — Annotated[..., add] accumulates across nodes
            pending_events = []
            pending_events.append(
                agent_events.stage(
                    name="scaffold",
                    description=f"Creating {template} project structure",
                    status="running",
                )
            )

            logger.info(
                "app_builder_scaffolding",
                template=template,
                user_id=user_id,
                task_id=task_id,
            )

            try:
                manager = get_app_sandbox_manager()

                # Create sandbox and scaffold
                session = await manager.get_or_create_sandbox(
                    user_id=user_id,
                    task_id=task_id,
                    template=template,
                )

                template_config = APP_TEMPLATES.get(template)
                if not template_config:
                    logger.warning(
                        "app_builder_template_config_fallback",
                        requested=template,
                        fallback="react",
                    )
                    template_config = APP_TEMPLATES["react"]

                result = await manager.scaffold_project(session, template)

                # Emit terminal command event — show cached vs live scaffold
                if result.get("cached"):
                    pending_events.append(
                        agent_events.terminal_command(
                            command=f"# Using cached {template} template (fast path)",
                            cwd="/home/user",
                        )
                    )
                else:
                    pending_events.append(
                        agent_events.terminal_command(
                            command=template_config["scaffold_cmd"],
                            cwd="/home/user",
                        )
                    )

                # Emit terminal output or error based on result
                if result.get("success"):
                    pending_events.append(
                        agent_events.terminal_output(
                            content=result.get("message", "Project scaffolded successfully"),
                            stream="stdout",
                        )
                    )
                    pending_events.append(
                        agent_events.terminal_complete(exit_code=0)
                    )
                else:
                    pending_events.append(
                        agent_events.terminal_error(
                            content=result.get("error", "Scaffold failed"),
                            exit_code=result.get("exit_code"),
                        )
                    )

                if not result["success"]:
                    pending_events.append(
                        agent_events.stage(
                            name="scaffold",
                            description="Failed to create project structure",
                            status="failed",
                        )
                    )
                    return {
                        "error": f"Scaffold failed: {result.get('error', 'Unknown error')}",
                        "current_step": "error",
                        "iterations": state.get("iterations", 0) + 1,
                        "pending_events": pending_events,
                    }

                # Install additional packages if needed
                packages = plan.get("packages", [])
                # Sanitize package names to prevent command injection
                _valid_pkg_re = re.compile(r"^[a-zA-Z0-9@._/-]+$")
                sanitized_packages = []
                for pkg in packages:
                    if _valid_pkg_re.match(pkg):
                        sanitized_packages.append(pkg)
                    else:
                        logger.warning("app_builder_invalid_package_name", package=pkg)
                packages = sanitized_packages
                if packages:
                    pkg_manager = "pip" if template in ["fastapi", "flask"] else "npm"
                    packages_str = " ".join(packages)
                    install_cmd = f"cd /home/user/app && {pkg_manager} install {packages_str}"
                    pending_events.append(
                        agent_events.terminal_command(
                            command=install_cmd,
                            cwd="/home/user/app",
                        )
                    )
                    install_result = await manager.install_dependencies(session, packages, pkg_manager)
                    if install_result.get("success"):
                        pending_events.append(
                            agent_events.terminal_output(
                                content=f"Installed {len(packages)} package(s)",
                                stream="stdout",
                            )
                        )
                        pending_events.append(
                            agent_events.terminal_complete(exit_code=0)
                        )
                    else:
                        pending_events.append(
                            agent_events.terminal_error(
                                content=install_result.get("error", "Installation failed"),
                            )
                        )

                pending_events.append(
                    agent_events.stage(
                        name="scaffold",
                        description=f"Created {template} project with dependencies",
                        status="completed",
                    )
                )

                return {
                    "current_step": "scaffold",  # Return current step, route_step maps to next
                    "iterations": state.get("iterations", 0) + 1,
                    "pending_events": pending_events,
                }

            except Exception as e:
                logger.error("app_builder_scaffold_failed", error=str(e))
                pending_events.append(
                    agent_events.stage(
                        name="scaffold",
                        description="Failed to scaffold project",
                        status="failed",
                    )
                )
                return {
                    "error": f"Failed to scaffold: {str(e)}",
                    "current_step": "error",
                    "iterations": state.get("iterations", 0) + 1,
                    "pending_events": pending_events,
                }

        async def generate_files(state: AppBuilderState) -> dict:
            """Generate the application files."""
            plan = state.get("plan", {})
            description = state["input_params"]["description"]
            template = plan.get("template", "react")
            files_to_create = plan.get("files", [])
            user_id = state.get("user_id")
            task_id = state.get("task_id")

            # Fresh event list per node — Annotated[..., add] accumulates across nodes
            pending_events = []
            pending_events.append(
                agent_events.stage(
                    name="generate",
                    description=f"Generating {len(files_to_create)} application files",
                    status="running",
                )
            )

            logger.info(
                "app_builder_generating_files",
                file_count=len(files_to_create),
            )

            try:
                manager = get_app_sandbox_manager()
                session = await manager.get_session(user_id=user_id, task_id=task_id)

                if not session:
                    pending_events.append(
                        agent_events.stage(
                            name="generate",
                            description="No active sandbox session",
                            status="failed",
                        )
                    )
                    return {
                        "error": "No active sandbox session",
                        "current_step": "error",
                        "iterations": state.get("iterations", 0) + 1,
                        "pending_events": pending_events,
                    }

                generated_files = []
                build_errors = []

                # Generate each file using LLM
                llm = llm_service.get_llm_for_tier(ModelTier.PRO)

                total_files = len(files_to_create)
                project_structure = chr(10).join(
                    [f"- {f.get('path')}: {f.get('description')}" for f in files_to_create]
                )

                # Emit "running" stage events for all files upfront
                for file_idx, file_spec in enumerate(files_to_create):
                    fp = file_spec.get("path", "")
                    if not fp:
                        continue
                    pending_events.append(
                        agent_events.stage(
                            name=f"generate:{fp}",
                            description=f"Generating {fp} ({file_idx + 1}/{total_files})",
                            status="running",
                        )
                    )
                    pending_events.append(
                        agent_events.terminal_command(
                            command=f"# Generating: {fp}",
                            cwd="/home/user/app",
                        )
                    )

                # --- Helper: generate one file (LLM call only) ---
                import asyncio as _asyncio

                async def _generate_one(
                    file_path: str, file_desc: str
                ) -> FileContent:
                    # Build context of other files for import/export coordination
                    other_files = "\n".join(
                        f"- {fp}: {fd}"
                        for fp, fd in valid_specs
                        if fp != file_path
                    )

                    prompt = f"""Generate the complete content for \
this file in a {template} project:

App Description: {description}

File: {file_path}
Purpose: {file_desc}

Other files in this project (coordinate imports/exports):
{other_files}

Project Structure:
{project_structure}

Generate production-quality, working code. Include all necessary imports.
For React: use functional components and hooks.
For Next.js: use App Router conventions.
For Vue: use Composition API.
For Express/FastAPI/Flask: include proper error handling.

Output ONLY the raw file code. Do NOT wrap in markdown fences or JSON.

The code should be complete and immediately runnable."""

                    resp = await llm.ainvoke(
                        [HumanMessage(content=prompt)]
                    )
                    text = resp.content or ""
                    if not text:
                        rc = resp.additional_kwargs.get(
                            "reasoning_content", ""
                        )
                        parsed = _parse_json_from_text(rc)
                        if parsed and "content" in parsed:
                            text = parsed["content"]
                        else:
                            text = rc
                    text = _strip_code_fences(text)
                    if not text:
                        raise ValueError(
                            f"LLM returned empty content for {file_path}"
                        )
                    return FileContent(
                        path=file_path,
                        content=text,
                        description=file_desc,
                    )

                # Launch all LLM calls in parallel (capped at 5 concurrent)
                valid_specs = [
                    (fs.get("path", ""), fs.get("description", ""))
                    for fs in files_to_create
                    if fs.get("path")
                ]
                _sem = _asyncio.Semaphore(5)

                async def _generate_with_limit(fp: str, fd: str) -> FileContent:
                    async with _sem:
                        return await _generate_one(fp, fd)

                tasks = [
                    _generate_with_limit(fp, fd) for fp, fd in valid_specs
                ]
                results = await _asyncio.gather(
                    *tasks, return_exceptions=True
                )

                # Write files sequentially and emit events
                sandbox_id = session.sandbox_id
                if not sandbox_id:
                    logger.warning("app_builder_missing_sandbox_id", task_id=task_id)
                    sandbox_id = f"app-{task_id or user_id or 'unknown'}"
                for (file_path, file_desc), gen_result in zip(
                    valid_specs, results
                ):
                    if isinstance(gen_result, Exception):
                        pending_events.append(
                            agent_events.terminal_error(
                                content=f"Failed to generate {file_path}: {gen_result}",
                            )
                        )
                        pending_events.append(
                            agent_events.stage(
                                name=f"generate:{file_path}",
                                description=f"Failed {file_path}",
                                status="failed",
                            )
                        )
                        build_errors.append(
                            f"Failed to generate {file_path}: {gen_result}"
                        )
                        logger.warning(
                            "app_builder_file_generation_failed",
                            path=file_path,
                            error=str(gen_result),
                        )
                        continue

                    file_content = gen_result
                    try:
                        write_result = await manager.write_file(
                            session,
                            file_content.path,
                            file_content.content,
                        )

                        if write_result["success"]:
                            generated_files.append(
                                {
                                    "path": file_content.path,
                                    "description": file_content.description,
                                }
                            )
                            content_len = len(file_content.content)
                            pending_events.append(
                                agent_events.terminal_output(
                                    content=f"Created {file_content.path} ({content_len} bytes)",
                                    stream="stdout",
                                )
                            )
                            parts = file_content.path.rsplit("/", 1)
                            file_name = (
                                parts[-1] if len(parts) > 1 else file_content.path
                            )
                            pending_events.append(
                                agent_events.workspace_update(
                                    operation="create",
                                    path=f"/home/user/app/{file_content.path}",
                                    name=file_name,
                                    sandbox_type="app",
                                    sandbox_id=sandbox_id,
                                    size=content_len,
                                )
                            )
                            pending_events.append(
                                agent_events.stage(
                                    name=f"generate:{file_path}",
                                    description=f"Created {file_path}",
                                    status="completed",
                                )
                            )
                            logger.info(
                                "app_builder_file_written",
                                path=file_content.path,
                            )
                        else:
                            pending_events.append(
                                agent_events.terminal_error(
                                    content=f"Failed to write {file_path}: "
                            f"{write_result.get('error')}",
                                )
                            )
                            pending_events.append(
                                agent_events.stage(
                                    name=f"generate:{file_path}",
                                    description=f"Failed {file_path}",
                                    status="failed",
                                )
                            )
                            build_errors.append(
                                f"Failed to write {file_path}: {write_result.get('error')}"
                            )
                    except Exception as e:
                        pending_events.append(
                            agent_events.terminal_error(
                                content=f"Failed to write {file_path}: {e}",
                            )
                        )
                        pending_events.append(
                            agent_events.stage(
                                name=f"generate:{file_path}",
                                description=f"Failed {file_path}",
                                status="failed",
                            )
                        )
                        build_errors.append(
                            f"Failed to write {file_path}: {e}"
                        )
                        logger.warning(
                            "app_builder_file_write_failed",
                            path=file_path,
                            error=str(e),
                        )

                pending_events.append(
                    agent_events.stage(
                        name="generate",
                        description=f"Generated {len(generated_files)}/{total_files} files"
                        + (f" ({len(build_errors)} failed)" if build_errors else ""),
                        status="completed",
                    )
                )

                return {
                    "generated_files": generated_files,
                    "build_errors": build_errors,
                    "current_step": "generate_files",  # Return current step, route_step maps to next
                    "iterations": state.get("iterations", 0) + 1,
                    "pending_events": pending_events,
                }

            except Exception as e:
                logger.error("app_builder_generate_files_failed", error=str(e))
                pending_events.append(
                    agent_events.stage(
                        name="generate",
                        description="Failed to generate files",
                        status="failed",
                    )
                )
                return {
                    "error": f"Failed to generate files: {str(e)}",
                    "current_step": "error",
                    "iterations": state.get("iterations", 0) + 1,
                    "pending_events": pending_events,
                }

        async def start_server(state: AppBuilderState) -> dict:
            """Start the development server and get preview URL."""
            plan = state.get("plan", {})
            template = plan.get("template", "react")
            user_id = state.get("user_id")
            task_id = state.get("task_id")

            # Fresh event list per node — Annotated[..., add] accumulates across nodes
            pending_events = []
            pending_events.append(
                agent_events.stage(
                    name="server",
                    description="Starting development server",
                    status="running",
                )
            )

            logger.info("app_builder_starting_server")

            try:
                manager = get_app_sandbox_manager()
                session = await manager.get_session(user_id=user_id, task_id=task_id)

                if not session:
                    pending_events.append(
                        agent_events.stage(
                            name="server",
                            description="No active sandbox session",
                            status="failed",
                        )
                    )
                    return {
                        "error": "No active sandbox session",
                        "current_step": "error",
                        "iterations": state.get("iterations", 0) + 1,
                        "pending_events": pending_events,
                    }

                # Emit terminal command for dev server
                template_config = APP_TEMPLATES.get(template, APP_TEMPLATES["react"])
                start_cmd = template_config["start_cmd"]
                pending_events.append(
                    agent_events.terminal_command(
                        command=start_cmd,
                        cwd="/home/user/app",
                    )
                )

                # Start the dev server
                result = await manager.start_dev_server(session)

                if result["success"]:
                    # Emit terminal output with server URL
                    pending_events.append(
                        agent_events.terminal_output(
                            content=f"Server running at {result['preview_url']}",
                            stream="stdout",
                        )
                    )
                    # Emit browser_stream event for virtual computer display
                    sandbox_id = session.sandbox_id
                    if not sandbox_id:
                        logger.warning("app_builder_missing_sandbox_id", task_id=task_id)
                        sandbox_id = f"app-{task_id or user_id or 'unknown'}"
                    pending_events.append(
                        agent_events.browser_stream(
                            stream_url=result["preview_url"],
                            sandbox_id=sandbox_id,
                            auth_key=None,
                        )
                    )
                    pending_events.append(
                        agent_events.stage(
                            name="server",
                            description="Server started successfully",
                            status="completed",
                        )
                    )
                    return {
                        "preview_url": result["preview_url"],
                        "current_step": "start_server",  # Return current step, route_step maps to next
                        "iterations": state.get("iterations", 0) + 1,
                        "pending_events": pending_events,
                    }
                else:
                    pending_events.append(
                        agent_events.stage(
                            name="server",
                            description="Failed to start server",
                            status="failed",
                        )
                    )
                    return {
                        "build_errors": state.get("build_errors", [])
                        + [result.get("error", "Server start failed")],
                        "current_step": "error",
                        "iterations": state.get("iterations", 0) + 1,
                        "pending_events": pending_events,
                    }

            except Exception as e:
                logger.error("app_builder_start_server_failed", error=str(e))
                pending_events.append(
                    agent_events.stage(
                        name="server",
                        description="Failed to start server",
                        status="failed",
                    )
                )
                return {
                    "error": f"Failed to start server: {str(e)}",
                    "current_step": "error",
                    "iterations": state.get("iterations", 0) + 1,
                    "pending_events": pending_events,
                }

        async def fix_build_errors(state: AppBuilderState) -> dict:
            """Attempt to fix build errors by reading server log and regenerating broken files."""
            error = state.get("error")
            build_errors = state.get("build_errors", [])
            if not error and not build_errors:
                # No errors — pass through to finalize
                return {
                    "current_step": "fix_build_errors",
                    "iterations": state.get("iterations", 0) + 1,
                    "pending_events": [],
                }

            user_id = state.get("user_id")
            task_id = state.get("task_id")
            retry_count = state.get("retry_count", 0)

            pending_events = []

            if retry_count >= 2:
                pending_events.append(
                    agent_events.stage(
                        name="fix_errors",
                        description="Max retries exceeded, proceeding with errors",
                        status="failed",
                    )
                )
                return {
                    "current_step": "fix_build_errors_exhausted",
                    "iterations": state.get("iterations", 0) + 1,
                    "pending_events": pending_events,
                }

            pending_events.append(
                agent_events.stage(
                    name="fix_errors",
                    description=f"Attempting to fix build errors (attempt {retry_count + 1}/2)",
                    status="running",
                )
            )

            try:
                manager = get_app_sandbox_manager()
                session = await manager.get_session(user_id=user_id, task_id=task_id)

                if not session:
                    return {
                        "current_step": "fix_build_errors_exhausted",
                        "iterations": state.get("iterations", 0) + 1,
                        "pending_events": pending_events,
                    }

                # Read the server error log
                log_result = await session.sandbox.run_command(
                    "cat /tmp/dev_server.log 2>/dev/null", timeout=5
                )
                error_log = log_result.stdout[:3000] if log_result.stdout else ""

                if not error_log:
                    pending_events.append(
                        agent_events.stage(
                            name="fix_errors",
                            description="No error log available to diagnose",
                            status="failed",
                        )
                    )
                    return {
                        "current_step": "fix_build_errors_exhausted",
                        "iterations": state.get("iterations", 0) + 1,
                        "pending_events": pending_events,
                    }

                generated_files = state.get("generated_files", [])
                file_paths = [f["path"] for f in generated_files]

                # Ask LLM to identify and fix the broken file
                llm = llm_service.get_llm_for_tier(ModelTier.PRO)
                fix_prompt = f"""The app build failed with this error log:
{error_log}

Generated files: {file_paths}

Identify which file has the error and provide the corrected content.
Respond with JSON: {{"file_path": "path/to/file", "content": "corrected file content"}}
Output ONLY the JSON, no explanation."""

                resp = await llm.ainvoke([HumanMessage(content=fix_prompt)])
                text = resp.content or ""
                if not text:
                    text = resp.additional_kwargs.get("reasoning_content", "")

                parsed = _parse_json_from_text(text)
                if not parsed or "file_path" not in parsed or "content" not in parsed:
                    pending_events.append(
                        agent_events.stage(
                            name="fix_errors",
                            description="Could not parse fix from LLM",
                            status="failed",
                        )
                    )
                    return {
                        "current_step": "fix_build_errors_exhausted",
                        "retry_count": retry_count + 1,
                        "iterations": state.get("iterations", 0) + 1,
                        "pending_events": pending_events,
                    }

                fix_path = parsed["file_path"]
                fix_content = _strip_code_fences(parsed["content"])

                # Write the fixed file
                write_result = await manager.write_file(session, fix_path, fix_content)
                if write_result.get("success"):
                    pending_events.append(
                        agent_events.terminal_command(
                            command=f"# Fix applied: {fix_path}",
                            cwd="/home/user/app",
                        )
                    )
                    pending_events.append(
                        agent_events.terminal_output(
                            content=f"Rewrote {fix_path} ({len(fix_content)} bytes)",
                            stream="stdout",
                        )
                    )
                    pending_events.append(
                        agent_events.stage(
                            name="fix_errors",
                            description=f"Fixed {fix_path}, retrying server",
                            status="completed",
                        )
                    )

                    logger.info(
                        "app_builder_fix_applied",
                        path=fix_path,
                        retry_count=retry_count + 1,
                    )

                    # Clear error so start_server can retry
                    return {
                        "current_step": "fix_build_errors",
                        "retry_count": retry_count + 1,
                        "error": None,
                        "iterations": state.get("iterations", 0) + 1,
                        "pending_events": pending_events,
                    }
                else:
                    pending_events.append(
                        agent_events.stage(
                            name="fix_errors",
                            description=f"Failed to write fix for {fix_path}",
                            status="failed",
                        )
                    )
                    return {
                        "current_step": "fix_build_errors_exhausted",
                        "retry_count": retry_count + 1,
                        "iterations": state.get("iterations", 0) + 1,
                        "pending_events": pending_events,
                    }

            except Exception as e:
                logger.error("app_builder_fix_errors_failed", error=str(e))
                pending_events.append(
                    agent_events.stage(
                        name="fix_errors",
                        description="Error recovery failed",
                        status="failed",
                    )
                )
                return {
                    "current_step": "fix_build_errors_exhausted",
                    "retry_count": retry_count + 1,
                    "iterations": state.get("iterations", 0) + 1,
                    "pending_events": pending_events,
                }

        async def finalize(state: AppBuilderState) -> dict:
            """Finalize the build and prepare output."""
            plan = state.get("plan", {})
            generated_files = state.get("generated_files", [])
            preview_url = state.get("preview_url")
            build_errors = state.get("build_errors", [])

            # Fresh event list per node — Annotated[..., add] accumulates across nodes
            pending_events = []
            pending_events.append(
                agent_events.stage(
                    name="finalize",
                    description="Finalizing app build",
                    status="running",
                )
            )

            logger.info(
                "app_builder_finalizing",
                preview_url=preview_url,
                files_count=len(generated_files),
                errors_count=len(build_errors),
            )

            # Prepare output
            if preview_url:
                output = {
                    "success": True,
                    "preview_url": preview_url,
                    "template": plan.get("template", "unknown"),
                    "files_created": [f["path"] for f in generated_files],
                    "message": f"App built successfully! View it at: {preview_url}",
                }

                if build_errors:
                    output["warnings"] = build_errors

                pending_events.append(
                    agent_events.stage(
                        name="finalize",
                        description="App ready for preview",
                        status="completed",
                    )
                )

            else:
                output = {
                    "success": False,
                    "template": plan.get("template", "unknown"),
                    "files_created": [f["path"] for f in generated_files],
                    "errors": build_errors or ["Unknown error occurred"],
                    "message": "App build encountered errors. Check the errors list for details.",
                }

                pending_events.append(
                    agent_events.stage(
                        name="finalize",
                        description="Build completed with errors",
                        status="failed",
                    )
                )

            return {
                "output": output,
                "current_step": "done",
                "iterations": state.get("iterations", 0) + 1,
                "pending_events": pending_events,
            }

        def route_step(state: AppBuilderState) -> str:
            """Route to the next step based on current state."""
            current = state.get("current_step", "plan")
            error = state.get("error")

            if current == "start_server" and not error:
                return "finalize"
            if current == "start_server" and error:
                return "fix_build_errors"
            if current == "fix_build_errors":
                # Retry: error was cleared → go back to start_server
                return "start_server"
            if current == "fix_build_errors_exhausted":
                return "finalize"
            if error:
                return "finalize"

            step_routing = {
                "plan": "scaffold",
                "scaffold": "generate_files",
                "generate_files": "start_server",
                "error": "finalize",
            }

            return step_routing.get(current, "finalize")

        # Build the graph
        graph.add_node("plan", plan_app)
        graph.add_node("scaffold", scaffold_project)
        graph.add_node("generate_files", generate_files)
        graph.add_node("start_server", start_server)
        graph.add_node("fix_build_errors", fix_build_errors)
        graph.add_node("finalize", finalize)

        # Set entry point
        graph.set_entry_point("plan")

        # Add conditional edges based on step
        graph.add_conditional_edges(
            "plan",
            route_step,
            {
                "scaffold": "scaffold",
                "finalize": "finalize",
            },
        )
        graph.add_conditional_edges(
            "scaffold",
            route_step,
            {
                "generate_files": "generate_files",
                "finalize": "finalize",
            },
        )
        graph.add_conditional_edges(
            "generate_files",
            route_step,
            {
                "start_server": "start_server",
                "finalize": "finalize",
            },
        )
        graph.add_conditional_edges(
            "start_server",
            route_step,
            {
                "fix_build_errors": "fix_build_errors",
                "finalize": "finalize",
            },
        )
        graph.add_conditional_edges(
            "fix_build_errors",
            route_step,
            {
                "start_server": "start_server",
                "finalize": "finalize",
            },
        )
        graph.add_edge("finalize", END)

        return graph.compile()
