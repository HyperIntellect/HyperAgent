"""Data Analysis Skill for full data analysis workflows.

This skill orchestrates a 3-node graph (plan -> code_loop -> summarize)
that plans analysis, generates/executes Python code in sandbox, captures
visualizations, and summarizes results. Replaces the standalone data agent.
"""

import base64
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from sqlalchemy import select

from app.agents import events as agent_events
from app.agents.prompts import (
    DATA_ANALYSIS_SYSTEM_MESSAGE,
    PLANNING_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    get_code_generation_prompt,
    get_data_system_prompt,
    get_planning_prompt,
    get_summary_prompt,
)
from app.agents.skills.skill_base import Skill, SkillMetadata, SkillParameter, SkillState
from app.agents.tools import execute_react_loop, get_react_config
from app.agents.utils import (
    append_history,
)
from app.ai.llm import extract_text_from_content, llm_service
from app.ai.model_tiers import ModelTier
from app.core.logging import get_logger
from app.db.base import async_session_maker
from app.db.models import File as FileModel
from app.sandbox import sandbox_file_with_context
from app.services.file_storage import file_storage_service

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal state for the skill graph
# ---------------------------------------------------------------------------


class DataAnalysisSkillState(SkillState, total=False):
    """State for data analysis skill execution."""

    # Data analysis specific
    query: str
    attachment_ids: list[str]
    data_source: str
    analysis_type: str
    analysis_plan: str

    # Code generation / execution
    code: str
    execution_result: str
    images: list[dict[str, str]]

    # Conversation context
    messages: list[dict[str, Any]]
    locale: str

    # LLM selection overrides
    provider: str | None
    model: str | None
    tier: Any | None

    # Streaming events
    pending_events: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Helper functions (ported from data.py)
# ---------------------------------------------------------------------------


def _safe_filename(file_id: str, filename: str) -> str:
    base_name = filename.rsplit("/", 1)[-1] or "file"
    return f"{file_id}_{base_name}"


def _extract_code(response: str) -> str:
    """Extract Python code from markdown code blocks."""
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        if len(matches) > 1:
            return max(matches, key=len).strip()
        return matches[0].strip()

    pattern = r"```\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        if len(matches) > 1:
            return max(matches, key=len).strip()
        return matches[0].strip()

    return ""


async def _upload_attachments_to_sandbox(
    attachment_ids: list[str],
    user_id: str,
    task_id: str | None,
) -> tuple[list[dict], bool, list[str]]:
    """Upload user's file attachments to the sandbox.

    Returns:
        Tuple of (events, success, uploaded_ids) where uploaded_ids contains
        the IDs of files that were actually written to the sandbox.
    """
    event_list: list[dict] = []
    uploaded_ids: list[str] = []
    try:
        async with async_session_maker() as session:
            result = await session.execute(
                select(FileModel).where(
                    FileModel.id.in_(attachment_ids),
                    FileModel.user_id == user_id,
                )
            )
            files = result.scalars().all()

            # Fallback: if user_id filter excluded files (e.g. deduplication),
            # retry without the user_id constraint.
            if not files:
                result = await session.execute(
                    select(FileModel).where(
                        FileModel.id.in_(attachment_ids),
                    )
                )
                files = result.scalars().all()
                if files:
                    logger.warning(
                        "files_found_without_user_id_filter",
                        attachment_ids=attachment_ids,
                        user_id=user_id,
                        file_user_ids=[f.user_id for f in files],
                    )

            if not files:
                err = "No files found for the provided attachment IDs"
                logger.error(
                    "no_files_found_for_attachments",
                    attachment_ids=attachment_ids,
                    user_id=user_id,
                )
                event_list.append(
                    agent_events.error(
                        error_msg=err, name="execute",
                        description=f"Error: {err}",
                    )
                )
                return event_list, False, []

            for f in files:
                event_list.append(
                    agent_events.stage(
                        "execute",
                        f"Uploading {f.original_filename} to sandbox...",
                        "running",
                    )
                )
                safe_name = _safe_filename(f.id, f.original_filename)
                file_data = await file_storage_service.download_file(
                    f.storage_key,
                )
                file_bytes = file_data.getvalue() if hasattr(file_data, "getvalue") else file_data

                write_result = await sandbox_file_with_context(
                    operation="write",
                    path=f"/home/user/{safe_name}",
                    content=base64.b64encode(file_bytes).decode("utf-8"),
                    is_binary=True,
                    user_id=user_id,
                    task_id=task_id,
                )

                if write_result.get("success"):
                    uploaded_ids.append(f.id)
                else:
                    err = f"Failed to upload {f.original_filename}: {write_result.get('error')}"
                    logger.error(
                        "file_upload_failed",
                        filename=f.original_filename,
                        error=write_result.get("error"),
                    )
                    event_list.append(
                        agent_events.error(
                            error_msg=err, name="execute",
                            description=f"Error: {err}",
                        ),
                    )

    except Exception as e:
        err = f"Failed to upload files to sandbox: {e}"
        logger.error(
            "failed_to_upload_attachments_to_sandbox",
            error=str(e),
        )
        event_list.append(agent_events.error(error_msg=str(e), name="execute", description=err))
        return event_list, False, uploaded_ids

    # Fail if no files were successfully uploaded
    if not uploaded_ids:
        err = "All file uploads to sandbox failed"
        logger.error("all_file_uploads_failed", attachment_ids=attachment_ids)
        event_list.append(
            agent_events.error(error_msg=err, name="execute", description=err)
        )
        return event_list, False, []

    if len(uploaded_ids) < len(files):
        logger.warning(
            "partial_file_upload",
            total=len(files),
            uploaded=len(uploaded_ids),
            uploaded_ids=uploaded_ids,
        )

    return event_list, True, uploaded_ids


async def _build_file_context(
    attachment_ids: list[str],
    user_id: str,
) -> str:
    """Query DB for file metadata and return formatted context string."""
    file_info: list[str] = []
    try:
        async with async_session_maker() as session:
            result = await session.execute(
                select(FileModel).where(
                    FileModel.id.in_(attachment_ids),
                    FileModel.user_id == user_id,
                )
            )
            files = result.scalars().all()
            # Fallback without user_id filter
            if not files:
                result = await session.execute(
                    select(FileModel).where(
                        FileModel.id.in_(attachment_ids),
                    )
                )
                files = result.scalars().all()
            for f in files:
                safe_name = _safe_filename(f.id, f.original_filename)
                file_info.append(
                    f"File: '/home/user/{safe_name}' (original: {f.original_filename})"
                )
    except Exception as e:
        logger.error("failed_to_fetch_file_info", error=str(e))

    return "\n".join(file_info) if file_info else "No additional data files."


def _get_data_tools() -> list:
    """Build the tool list for data analysis (no handoff tools)."""
    from app.agents.tools.code_execution import execute_code
    from app.agents.tools.image_generation import generate_image
    from app.agents.tools.vision import analyze_image
    from app.agents.tools.web_search import web_search
    from app.sandbox import sandbox_file

    return [
        execute_code,
        sandbox_file,
        web_search,
        generate_image,
        analyze_image,
    ]


# ---------------------------------------------------------------------------
# Analysis type detection keywords
# ---------------------------------------------------------------------------

_VIZ_WORDS = ["plot", "chart", "graph", "visualiz", "show"]
_STATS_WORDS = [
    "statistic",
    "mean",
    "median",
    "correlation",
    "regression",
]
_PROC_WORDS = ["clean", "transform", "parse", "convert", "filter"]
_ML_WORDS = ["predict", "classify", "cluster", "train", "model"]


# ---------------------------------------------------------------------------
# Skill class
# ---------------------------------------------------------------------------


class DataAnalysisSkill(Skill):
    """Full data analysis: plan, code, summarize."""

    metadata = SkillMetadata(
        id="data_analysis",
        name="Data Analysis",
        version="1.0.0",
        description=(
            "Full data analysis: plans approach, generates/executes "
            "Python code in sandbox, captures visualizations, "
            "summarizes results. Use for CSV/Excel/JSON, statistics, "
            "visualization, ML."
        ),
        category="data",
        parameters=[
            SkillParameter(
                name="query",
                type="string",
                description="The data analysis query",
                required=True,
            ),
            SkillParameter(
                name="attachment_ids",
                type="array",
                description="IDs of attached data files",
                required=False,
                default=[],
            ),
            SkillParameter(
                name="data_source",
                type="string",
                description="Inline data or data source reference",
                required=False,
                default="",
            ),
        ],
        output_schema={
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "Summary of analysis results",
                },
                "code": {
                    "type": "string",
                    "description": "Final Python code executed",
                },
                "execution_result": {
                    "type": "string",
                    "description": "Code execution output",
                },
                "images": {
                    "type": "array",
                    "description": "Generated visualizations",
                },
                "analysis_type": {
                    "type": "string",
                    "description": "Detected analysis type",
                },
            },
        },
        required_tools=[
            "execute_code",
            "sandbox_file",
            "web_search",
            "generate_image",
            "analyze_image",
        ],
        risk_level="high",
        side_effect_level="high",
        data_sensitivity="sensitive",
        network_scope="sandbox_only",
        idempotency_hint=False,
        max_execution_time_seconds=600,
        max_iterations=20,
        tags=[
            "data",
            "analysis",
            "csv",
            "statistics",
            "visualization",
            "pandas",
        ],
    )

    def create_graph(self) -> StateGraph:
        """Build plan -> code_loop -> summarize -> END graph."""
        graph = StateGraph(DataAnalysisSkillState)

        # --------------------------------------------------------------
        # Node: plan
        # --------------------------------------------------------------
        async def plan_analysis(
            state: DataAnalysisSkillState,
        ) -> dict:
            """Plan the data analysis approach."""
            params = state.get("input_params", {})
            query = params.get("query", "")
            attachment_ids = params.get("attachment_ids", [])
            data_source = params.get("data_source", "")
            user_id = state.get("user_id")

            # Read context passed from task agent via input_params
            provider_from_params = params.get("provider", state.get("provider"))
            model_from_params = params.get("model", state.get("model"))
            messages_from_params = params.get("messages", [])

            pending = list(state.get("pending_events", []))
            pending.append(
                agent_events.stage(
                    name="plan",
                    description="Planning data analysis approach",
                    status="running",
                )
            )

            # Fetch attachment info
            attachments_info: list[str] = []
            if attachment_ids and user_id:
                try:
                    async with async_session_maker() as session:
                        result = await session.execute(
                            select(FileModel).where(
                                FileModel.id.in_(attachment_ids),
                                FileModel.user_id == user_id,
                            )
                        )
                        files = result.scalars().all()
                        # Fallback without user_id filter
                        if not files:
                            result = await session.execute(
                                select(FileModel).where(
                                    FileModel.id.in_(attachment_ids),
                                )
                            )
                            files = result.scalars().all()
                        attachments_info = [
                            f"- {f.original_filename} ({f.content_type}, {f.file_size} bytes)"
                            for f in files
                        ]
                        if attachments_info:
                            pending.append(
                                agent_events.stage(
                                    "plan",
                                    f"Found {len(attachments_info)} attachments for analysis",
                                    "running",
                                )
                            )
                except Exception as e:
                    logger.error(
                        "failed_to_fetch_attachments",
                        error=str(e),
                    )

            attachments_ctx = (
                "\n".join(attachments_info) if attachments_info else "No files attached."
            )

            try:
                planning_prompt = get_planning_prompt(
                    query,
                    attachments_ctx,
                )
                provider = provider_from_params
                tier = state.get("tier")
                model = model_from_params
                llm = llm_service.choose_llm_for_task(
                    "data",
                    provider=provider,
                    tier_override=tier,
                    model_override=model,
                )
                messages = [
                    SystemMessage(content=PLANNING_SYSTEM_PROMPT),
                ]
                # Prefer messages passed via input_params (from task agent),
                # fall back to state-level messages field.
                history_msgs = messages_from_params or state.get("messages", [])
                append_history(messages, history_msgs)
                messages.append(
                    HumanMessage(content=planning_prompt),
                )

                # Planning is a pure reasoning step — no sandbox tools needed
                # (files are uploaded later in code_loop).
                plan = ""
                chunks: list[str] = []
                async for chunk in llm.astream(messages):
                    if chunk.content:
                        c = extract_text_from_content(
                            chunk.content,
                        )
                        if c:
                            chunks.append(c)
                plan = "".join(chunks)

                # Detect analysis type
                analysis_type = _detect_analysis_type(query)

                preview = plan[:200].strip()
                if len(plan) > 200:
                    preview += "..."

                desc = f"Analysis type: {analysis_type}"
                if preview:
                    desc += f" - {preview}"
                pending.append(
                    agent_events.stage("plan", desc, "completed"),
                )

                logger.info(
                    "analysis_planned",
                    query=query[:50],
                    analysis_type=analysis_type,
                )

                return {
                    "query": query,
                    "attachment_ids": attachment_ids,
                    "data_source": data_source,
                    "analysis_type": analysis_type,
                    "analysis_plan": plan,
                    "pending_events": pending,
                    "iterations": state.get("iterations", 0) + 1,
                }

            except Exception as e:
                logger.error(
                    "analysis_planning_failed",
                    error=str(e),
                )
                pending.append(
                    agent_events.error(
                        error_msg=str(e),
                        name="plan",
                        description=f"Planning error: {e}",
                    )
                )
                return {
                    "query": query,
                    "attachment_ids": attachment_ids,
                    "data_source": data_source,
                    "analysis_type": "general",
                    "pending_events": pending,
                    "iterations": state.get("iterations", 0) + 1,
                }

        # --------------------------------------------------------------
        # Node: code_loop
        # --------------------------------------------------------------
        async def code_loop(
            state: DataAnalysisSkillState,
        ) -> dict:
            """Generate and execute code in a ReAct loop."""
            params = state.get("input_params", {})
            query = state.get("query", "")
            analysis_plan = state.get("analysis_plan", "")
            attachment_ids = state.get("attachment_ids", [])
            user_id = state.get("user_id")
            task_id = state.get("task_id")

            # Read context passed from task agent via input_params
            locale = params.get("locale", state.get("locale", "en"))
            provider = params.get("provider", state.get("provider"))
            model = params.get("model", state.get("model"))
            messages_from_params = params.get("messages", [])

            pending = list(state.get("pending_events", []))
            pending.append(
                agent_events.stage(
                    name="code_loop",
                    description="Generating and executing code",
                    status="running",
                )
            )

            # Upload attachments
            uploaded_ids: list[str] = []
            if attachment_ids and user_id:
                upload_evts, ok, uploaded_ids = await _upload_attachments_to_sandbox(
                    attachment_ids,
                    user_id,
                    task_id,
                )
                pending.extend(upload_evts)
                if not ok:
                    return {
                        "execution_result": (
                            "Failed to upload files to sandbox."
                            " Please check sandbox configuration."
                        ),
                        "pending_events": pending,
                        "iterations": (state.get("iterations", 0) + 1),
                    }

            # Build context using only files that were actually uploaded to sandbox
            file_ctx = ""
            effective_ids = uploaded_ids if uploaded_ids else attachment_ids
            if effective_ids:
                file_ctx = await _build_file_context(
                    effective_ids,
                    user_id,
                )
            data_ctx = (state.get("data_source", "") or "")[:2000]

            code_prompt = get_code_generation_prompt(
                query=query,
                data_context=data_ctx,
                analysis_type=state.get("analysis_type", "general"),
                analysis_plan=analysis_plan,
                file_context=file_ctx,
            )

            if locale and locale != "en":
                sys_msg = SystemMessage(
                    content=get_data_system_prompt(locale),
                    additional_kwargs={
                        "cache_control": {"type": "ephemeral"},
                    },
                )
            else:
                sys_msg = DATA_ANALYSIS_SYSTEM_MESSAGE
            messages = [sys_msg]
            # Prefer messages passed via input_params (from task agent),
            # fall back to state-level messages field.
            history_msgs = messages_from_params or state.get("messages", [])
            append_history(messages, history_msgs)
            messages.append(HumanMessage(content=code_prompt))

            tier = state.get("tier") or state.get("input_params", {}).get("tier")
            llm = llm_service.choose_llm_for_task(
                "code",
                provider=provider,
                tier_override=tier or ModelTier.MAX,
                model_override=model,
            )
            all_tools = _get_data_tools()
            llm_with_tools = llm.bind_tools(all_tools)
            config = get_react_config("data", tier=tier)

            try:
                result = await execute_react_loop(
                    llm_with_tools=llm_with_tools,
                    messages=messages,
                    tools=all_tools,
                    query=query,
                    config=config,
                    source_agent="data",
                    extra_tool_args={
                        "user_id": user_id,
                        "task_id": task_id,
                    },
                    on_token=lambda t: pending.append(
                        agent_events.token(t),
                    ),
                    on_tool_call=lambda name, args, tid: (
                        pending.append(
                            agent_events.tool_call(name, args, tool_id=tid),
                        )
                    ),
                    on_tool_result=lambda name, res, tid: (
                        pending.append(
                            agent_events.tool_result(name, res, tool_id=tid),
                        )
                    ),
                )

                pending.extend(result.events)

                # Extract artifacts from the message history
                code = ""
                execution_result = ""
                images: list[dict] = []

                for msg in reversed(result.messages):
                    if isinstance(msg, ToolMessage) and msg.name == "execute_code":
                        try:
                            parsed = json.loads(msg.content)
                            execution_result = parsed.get(
                                "stdout",
                                "",
                            )
                            if parsed.get("stderr"):
                                execution_result += f"\nErrors:\n{parsed['stderr']}"
                            images = parsed.get("images", [])
                        except (json.JSONDecodeError, AttributeError):
                            execution_result = (
                                msg.content if isinstance(msg.content, str) else str(msg.content)
                            )
                        break

                for msg in reversed(result.messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        extracted = _extract_code(
                            extract_text_from_content(msg.content),
                        )
                        if extracted:
                            code = extracted
                            break

                # Fallback A: if the LLM responded with text containing code
                # but never called tools, extract and execute the code directly.
                if result.tool_iterations == 0 and code and not execution_result:
                    logger.info(
                        "code_loop_fallback_execution",
                        code_length=len(code),
                    )
                    pending.append(
                        agent_events.stage(
                            "code_loop",
                            "LLM did not call tools — executing extracted code",
                            "running",
                        )
                    )
                    try:
                        from app.agents.tools.code_execution import execute_code
                        exec_result_str = await execute_code.ainvoke({
                            "code": code,
                            "user_id": user_id,
                            "task_id": task_id,
                        })
                        try:
                            parsed = json.loads(exec_result_str)
                            execution_result = parsed.get("stdout", "")
                            if parsed.get("stderr"):
                                execution_result += f"\nErrors:\n{parsed['stderr']}"
                            images = parsed.get("images", [])
                        except (json.JSONDecodeError, AttributeError):
                            execution_result = (
                                exec_result_str
                                if isinstance(exec_result_str, str)
                                else str(exec_result_str)
                            )
                        logger.info(
                            "code_loop_fallback_completed",
                            has_result=bool(execution_result),
                            image_count=len(images),
                        )
                    except Exception as fallback_err:
                        logger.error(
                            "code_loop_fallback_failed",
                            error=str(fallback_err),
                        )
                        execution_result = f"Fallback execution error: {fallback_err}"

                # Fallback B: LLM produced neither tool calls nor code text.
                # Retry without tool binding (plain code generation) so models
                # that struggle with tool-calling can still produce analysis code.
                if result.tool_iterations == 0 and not code and not execution_result:
                    logger.warning(
                        "code_loop_no_output_retrying_without_tools",
                        original_token_count=len([
                            e for e in pending
                            if isinstance(e, dict) and e.get("type") == "token"
                        ]),
                    )
                    pending.append(
                        agent_events.stage(
                            "code_loop",
                            "Retrying code generation without tool binding",
                            "running",
                        )
                    )
                    try:
                        # Use plain LLM (no tools bound) with explicit code instruction
                        retry_prompt = (
                            "You MUST write Python code to analyze the data. "
                            "Output ONLY a single ```python code block. "
                            "The code should read the data file, perform analysis, "
                            "and print results to stdout. "
                            "Use matplotlib to save any charts to /tmp/outputs/."
                        )
                        retry_messages = list(messages) + [
                            HumanMessage(content=retry_prompt),
                        ]
                        retry_chunks: list[str] = []
                        async for chunk in llm.astream(retry_messages):
                            if chunk.content:
                                c = extract_text_from_content(chunk.content)
                                if c:
                                    retry_chunks.append(c)
                                    pending.append(agent_events.token(c))
                        retry_text = "".join(retry_chunks)
                        code = _extract_code(retry_text)

                        if code:
                            logger.info(
                                "code_loop_retry_code_extracted",
                                code_length=len(code),
                            )
                            from app.agents.tools.code_execution import execute_code
                            exec_result_str = await execute_code.ainvoke({
                                "code": code,
                                "user_id": user_id,
                                "task_id": task_id,
                            })
                            try:
                                parsed = json.loads(exec_result_str)
                                execution_result = parsed.get("stdout", "")
                                if parsed.get("stderr"):
                                    execution_result += (
                                        f"\nErrors:\n{parsed['stderr']}"
                                    )
                                images = parsed.get("images", [])
                            except (json.JSONDecodeError, AttributeError):
                                execution_result = (
                                    exec_result_str
                                    if isinstance(exec_result_str, str)
                                    else str(exec_result_str)
                                )
                        else:
                            logger.warning("code_loop_retry_no_code_extracted")
                    except Exception as retry_err:
                        logger.error(
                            "code_loop_retry_failed",
                            error=str(retry_err),
                        )

                # Emit image events
                for img in images:
                    img_data = img.get("data", "")
                    if img_data:
                        pending.append(
                            agent_events.image(
                                data=img_data,
                                mime_type=img.get(
                                    "type",
                                    "image/png",
                                ),
                            )
                        )

                pending.append(
                    agent_events.stage(
                        "code_loop",
                        "Code loop complete",
                        "completed",
                    )
                )

                logger.info(
                    "code_loop_completed",
                    iterations=result.tool_iterations,
                    code_length=len(code),
                    has_execution_result=bool(execution_result),
                )

                return {
                    "code": code,
                    "execution_result": execution_result,
                    "images": images,
                    "pending_events": pending,
                    "iterations": state.get("iterations", 0) + 1,
                }

            except Exception as e:
                logger.error("code_loop_failed", error=str(e))
                pending.append(
                    agent_events.error(
                        error_msg=str(e),
                        name="code_loop",
                        description=f"Code loop failed: {e}",
                    )
                )
                return {
                    "execution_result": f"Error: {e}",
                    "pending_events": pending,
                    "iterations": state.get("iterations", 0) + 1,
                }

        # --------------------------------------------------------------
        # Node: summarize
        # --------------------------------------------------------------
        async def summarize_results(
            state: DataAnalysisSkillState,
        ) -> dict:
            """Summarize the analysis results."""
            params = state.get("input_params", {})
            query = state.get("query", "")
            execution_result = state.get("execution_result", "")
            code = state.get("code", "")
            analysis_type = state.get("analysis_type", "general")
            images = state.get("images", [])

            # Read context passed from task agent via input_params
            provider = params.get("provider", state.get("provider"))
            model = params.get("model", state.get("model"))
            messages_from_params = params.get("messages", [])

            pending = list(state.get("pending_events", []))
            pending.append(
                agent_events.stage(
                    name="summarize",
                    description="Summarizing analysis results",
                    status="running",
                )
            )

            tier = state.get("tier")
            llm = llm_service.choose_llm_for_task(
                "data",
                provider=provider,
                tier_override=tier,
                model_override=model,
            )

            try:
                summary_prompt = get_summary_prompt(
                    query=query,
                    analysis_type=analysis_type,
                    code=code[:1500],
                    execution_result=execution_result[:2000],
                    has_visualization=len(images) > 0,
                    visualization_count=len(images),
                )

                response_chunks: list[str] = []
                messages = [
                    SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                ]
                history_msgs = messages_from_params or state.get("messages", [])
                append_history(messages, history_msgs)
                messages.append(
                    HumanMessage(content=summary_prompt),
                )
                async for chunk in llm.astream(
                    messages,
                    config={"tags": ["summarize"]},
                ):
                    if chunk.content:
                        content = extract_text_from_content(
                            chunk.content,
                        )
                        if content:
                            response_chunks.append(content)
                            pending.append(
                                agent_events.token(content),
                            )

                summary = "".join(response_chunks)

                pending.append(
                    agent_events.stage(
                        "summarize",
                        "Summary complete",
                        "completed",
                    )
                )

                logger.info("analysis_summarized")

                return {
                    "output": {
                        "response": summary,
                        "code": code,
                        "execution_result": execution_result,
                        "images": images,
                        "analysis_type": analysis_type,
                    },
                    "pending_events": pending,
                    "iterations": state.get("iterations", 0) + 1,
                }

            except Exception as e:
                logger.error("summarization_failed", error=str(e))
                return {
                    "output": {
                        "response": (f"Analysis Results:\n\n{execution_result}"),
                        "code": code,
                        "execution_result": execution_result,
                        "images": images,
                        "analysis_type": analysis_type,
                    },
                    "pending_events": pending,
                    "iterations": state.get("iterations", 0) + 1,
                }

        # Build graph
        graph.add_node("plan", plan_analysis)
        graph.add_node("code_loop", code_loop)
        graph.add_node("summarize", summarize_results)

        graph.set_entry_point("plan")
        graph.add_edge("plan", "code_loop")
        graph.add_edge("code_loop", "summarize")
        graph.add_edge("summarize", END)

        return graph.compile()


def _detect_analysis_type(query: str) -> str:
    """Detect the analysis type from the query text."""
    q = query.lower()
    if any(w in q for w in _VIZ_WORDS):
        return "visualization"
    if any(w in q for w in _STATS_WORDS):
        return "statistics"
    if any(w in q for w in _PROC_WORDS):
        return "processing"
    if any(w in q for w in _ML_WORDS):
        return "ml"
    return "general"
