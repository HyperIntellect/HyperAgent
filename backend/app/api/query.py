"""Unified query router for both chat and research modes."""

import asyncio
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import agent_supervisor
from app.ai.model_tiers import ModelTier, resolve_model
from app.api.query_helpers import (
    TASK_SYSTEM_PROMPT,
    _sse_data,
    get_conversation_history,
    get_file_context,
    get_image_attachments,
    trim_duplicate_user_message,
)
from app.config import settings
from app.core.auth import CurrentUser, get_current_user
from app.core.logging import get_logger
from app.db.base import get_db
from app.guardrails.scanners.input_scanner import input_scanner
from app.models.schemas import (
    QueryMode,
    UnifiedQueryRequest,
    UnifiedQueryResponse,
)
from app.repository import deep_research_repository
from app.services.run_ledger import run_ledger_service

logger = get_logger(__name__)

router = APIRouter(prefix="/query")


@router.post("/", response_model=UnifiedQueryResponse)
async def query(
    request: UnifiedQueryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Unified entry point for chat and research modes."""
    try:
        # Get file context if attachments provided
        file_context = await get_file_context(
            db,
            request.attachment_ids,
            current_user.id,
        )

        # Enhance system prompt with file context
        system_prompt = TASK_SYSTEM_PROMPT
        if file_context:
            system_prompt = f"{TASK_SYSTEM_PROMPT}\n\nThe user has attached the following files for context:\n\n{file_context}"

        history = [m.model_dump() for m in request.history]
        if not history and request.conversation_id:
            history = await get_conversation_history(
                db,
                request.conversation_id,
                current_user.id,
            )
            history = trim_duplicate_user_message(history, request.message)

        # Resolve effective provider/model for responses (respects tier config)
        effective_provider, effective_model = resolve_model(ModelTier.PRO, provider=request.provider)

        if request.mode == QueryMode.TASK:
            # Use agent supervisor even for chat to enable tools
            result = await agent_supervisor.invoke(
                query=request.message,
                mode=QueryMode.TASK.value,
                user_id=current_user.id,
                messages=history,
                system_prompt=system_prompt,
                provider=request.provider,
                model=request.model,
            )

            return UnifiedQueryResponse(
                id=str(uuid.uuid4()),
                mode=QueryMode.TASK,
                content=result.get("response", ""),
                model=request.model or effective_model,
                provider=effective_provider,
            )

        elif request.mode in (QueryMode.APP, QueryMode.DATA, QueryMode.IMAGE, QueryMode.SLIDE, QueryMode.RESEARCH):
            # Use agent supervisor for specialized modes (including research)
            result = await agent_supervisor.invoke(
                query=request.message,
                mode=request.mode.value,
                user_id=current_user.id,
                messages=history,
                system_prompt=system_prompt,
                provider=request.provider,
                model=request.model,
                scenario=request.scenario,
                depth=request.depth,
            )

            return UnifiedQueryResponse(
                id=str(uuid.uuid4()),
                mode=request.mode,
                content=result.get("response", ""),
                model=request.model or effective_model,
                provider=effective_provider,
            )
    except ValueError as e:
        logger.error("chat_error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("chat_error", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500, detail="An internal error occurred while processing your message."
        )


@router.post("/stream")
async def stream_query(
    request: UnifiedQueryRequest,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Stream response for chat, research, and other agent modes."""
    _CHAT_MODES = (
        QueryMode.TASK,
        QueryMode.APP,
        QueryMode.DATA,
        QueryMode.IMAGE,
        QueryMode.SLIDE,
        QueryMode.RESEARCH,
    )
    if request.mode in _CHAT_MODES:
        # Phase 1: All DB work in a scoped session (released before streaming)
        from app.db.base import async_session_maker

        async with async_session_maker() as db:
            history = [m.model_dump() for m in request.history]
            if not history and request.conversation_id:
                history = await get_conversation_history(
                    db,
                    request.conversation_id,
                    current_user.id,
                )
                history = trim_duplicate_user_message(history, request.message)

            # Get file context if attachments provided
            file_context = await get_file_context(
                db,
                request.attachment_ids,
                current_user.id,
            )

            # Get image attachments as base64 for vision tool usage
            image_attachments = await get_image_attachments(
                db,
                request.attachment_ids,
                current_user.id,
            )
        # DB session released here - pool slot freed

        # Debug logging for conversation context
        logger.info(
            "query_stream_context",
            conversation_id=request.conversation_id,
            history_from_request=len(request.history),
            history_total=len(history),
            has_history=bool(history),
        )

        # Enhance system prompt with file context
        system_prompt = TASK_SYSTEM_PROMPT
        if file_context:
            system_prompt = f"{TASK_SYSTEM_PROMPT}\n\nThe user has attached the following files for context:\n\n{file_context}"

        # Generate task_id for browser session management if not provided
        # Use request.task_id, conversation_id as fallback, or generate a new one
        chat_task_id = request.task_id or request.conversation_id or str(uuid.uuid4())
        run_id = str(uuid.uuid4())

        # Input guardrails check
        scan_result = await input_scanner.scan(request.message)
        if scan_result.blocked:
            logger.warning(
                "input_guardrail_blocked",
                violations=[v.value for v in scan_result.violations],
                reason=scan_result.reason,
            )

            async def blocked_generator() -> AsyncGenerator[str, None]:
                error_message = (
                    "I'm unable to process this request due to safety concerns. "
                    "Please rephrase your message and try again."
                )
                yield _sse_data({"type": "token", "data": error_message})
                yield _sse_data({"type": "complete", "data": ""})

            return StreamingResponse(
                blocked_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # Phase 2: SSE streaming (no DB dependency)

        if settings.run_ledger_v1:
            await run_ledger_service.create_run(
                run_id=run_id,
                user_id=current_user.id,
                mode=request.mode.value,
                objective=request.message,
                task_id=chat_task_id,
                conversation_id=request.conversation_id,
                execution_mode=request.execution_mode,
                budget=request.budget,
                run_labels=request.run_labels,
            )

        # Declarative field mappings per event type.
        # Each entry maps event_type -> list of (output_key, event_key, default).
        # This replaces the ~180-line if/elif chain with a compact table.
        _EVENT_FIELD_MAP: dict[str, list[tuple[str, str, Any]]] = {
            "token": [("data", "content", "")],
            "tool_call": [("tool", "tool", ""), ("args", "args", {}), ("id", "id", None)],
            "tool_result": [
                ("tool", "tool", ""),
                ("content", "content", ""),
                ("id", "id", None),
            ],
            "routing": [
                ("agent", "agent", ""),
                ("reason", "reason", ""),
                ("confidence", "confidence", None),
                ("low_confidence", "low_confidence", False),
            ],
            "handoff": [("source", "source", ""), ("target", "target", ""), ("task", "task", "")],
            "source": [("title", "title", ""), ("url", "url", ""), ("snippet", "snippet", "")],
            "code_result": [
                ("output", "output", ""),
                ("exit_code", "exit_code", None),
                ("error", "error", None),
            ],
            "browser_action": [
                ("action", "action", ""),
                ("description", "description", ""),
                ("target", "target", None),
                ("status", "status", "running"),
            ],
            "terminal_command": [
                ("command", "command", ""),
                ("cwd", "cwd", "/home/user"),
                ("timestamp", "timestamp", None),
            ],
            "terminal_output": [
                ("content", "content", ""),
                ("stream", "stream", "stdout"),
                ("timestamp", "timestamp", None),
            ],
            "terminal_error": [
                ("content", "content", ""),
                ("exit_code", "exit_code", None),
                ("timestamp", "timestamp", None),
            ],
            "terminal_complete": [("exit_code", "exit_code", 0), ("timestamp", "timestamp", None)],
            "workspace_update": [
                ("operation", "operation", "create"),
                ("path", "path", ""),
                ("name", "name", ""),
                ("is_directory", "is_directory", False),
                ("size", "size", None),
                ("sandbox_type", "sandbox_type", "app"),
                ("sandbox_id", "sandbox_id", ""),
                ("timestamp", "timestamp", None),
            ],
            "skill_output": [("skill_id", "skill_id", ""), ("output", "output", {})],
            "parallel_task": [
                ("task_id", "task_id", ""),
                ("focus_area", "focus_area", ""),
                ("status", "status", "pending"),
                ("query", "query", None),
                ("duration_ms", "duration_ms", None),
            ],
            "reasoning": [
                ("thinking", "thinking", ""),
                ("confidence", "confidence", None),
                ("context", "context", None),
            ],
            "verification": [
                ("step_number", "step_number", None),
                ("status", "status", "running"),
                ("message", "message", ""),
                ("findings", "findings", []),
                ("retry_hint", "retry_hint", None),
            ],
            "usage": [
                ("input_tokens", "input_tokens", 0),
                ("output_tokens", "output_tokens", 0),
                ("cached_tokens", "cached_tokens", 0),
                ("cost_usd", "cost_usd", 0.0),
                ("model", "model", ""),
                ("tier", "tier", ""),
            ],
            "error": [("data", "error", "Unknown error")],
        }
        # Event types that are passed through as-is (no field mapping needed)
        _PASSTHROUGH_EVENTS = {"stage", "complete"}

        def _build_sse_payload(event_type: str, event: dict) -> dict[str, Any] | None:
            """Build SSE payload from event using declarative field map."""
            if event_type in _PASSTHROUGH_EVENTS:
                if event_type == "complete":
                    return {"type": "complete", "data": ""}
                return event
            fields = _EVENT_FIELD_MAP.get(event_type)
            if fields is None:
                return None
            payload: dict[str, Any] = {"type": event_type}
            for out_key, evt_key, default in fields:
                payload[out_key] = event.get(evt_key, default)
            return payload

        # Stream agent response using supervisor
        async def agent_generator() -> AsyncGenerator[str, None]:
            try:
                async for event in agent_supervisor.run(
                    query=request.message,
                    mode=request.mode.value,
                    task_id=chat_task_id,
                    run_id=run_id,
                    user_id=current_user.id,
                    messages=history,
                    system_prompt=system_prompt,
                    provider=request.provider,
                    model=request.model,
                    attachment_ids=request.attachment_ids,
                    image_attachments=image_attachments,
                    locale=request.locale,
                    budget=request.budget,
                    execution_mode=request.execution_mode,
                    skills=request.skills,
                    scenario=request.scenario,
                    depth=request.depth,
                ):
                    event_type = event["type"]

                    # Special handlers for events with conditional logic or logging
                    if event_type == "browser_stream":
                        logger.info(
                            "streaming_browser_stream_event",
                            sandbox_id=event.get("sandbox_id", "")[:8],
                        )
                        yield _sse_data(
                            {
                                "type": "browser_stream",
                                "run_id": run_id,
                                "step_id": event.get("id"),
                                "stream_url": event.get("stream_url", ""),
                                "sandbox_id": event.get("sandbox_id", ""),
                                "auth_key": event.get("auth_key"),
                            }
                        )
                        if settings.run_ledger_v1:
                            await run_ledger_service.record_event(
                                run_id=run_id,
                                event_type="browser_stream",
                                payload={
                                    "run_id": run_id,
                                    "step_id": event.get("id"),
                                    "stream_url": event.get("stream_url", ""),
                                    "sandbox_id": event.get("sandbox_id", ""),
                                    "auth_key": event.get("auth_key"),
                                },
                                step_id=event.get("id"),
                                dedup_key=f"browser_stream:{event.get('sandbox_id')}",
                            )

                    elif event_type == "image":
                        img_data = event.get("data")
                        img_url = event.get("url")
                        if img_data or img_url:
                            payload: dict[str, Any] = {
                                "type": "image",
                                "run_id": run_id,
                                "step_id": event.get("id"),
                                "mime_type": event.get("mime_type", "image/png"),
                                "index": event.get("index"),
                            }
                            if img_data:
                                payload["data"] = img_data
                            if img_url:
                                payload["url"] = img_url
                            if event.get("storage_key"):
                                payload["storage_key"] = event["storage_key"]
                            if event.get("file_id"):
                                payload["file_id"] = event["file_id"]

                            logger.info(
                                "streaming_image_event",
                                mime_type=event.get("mime_type", "image/png"),
                                has_data=bool(img_data),
                                has_url=bool(img_url),
                                index=event.get("index"),
                            )
                            yield _sse_data(payload)
                            if settings.run_ledger_v1:
                                await run_ledger_service.record_event(
                                    run_id=run_id,
                                    event_type="image",
                                    payload=payload,
                                    step_id=payload.get("step_id"),
                                    dedup_key=f"image:{payload.get('index')}",
                                )
                        else:
                            logger.warning("skipping_empty_image_event", event=event)

                    elif event_type == "interrupt":
                        logger.info(
                            "streaming_interrupt_event",
                            interrupt_id=event.get("interrupt_id", "")[:8],
                            interrupt_type=event.get("interrupt_type"),
                        )
                        yield _sse_data(
                            {
                                "type": "interrupt",
                                "run_id": run_id,
                                "step_id": event.get("id"),
                                "interrupt_id": event.get("interrupt_id", ""),
                                "interrupt_type": event.get("interrupt_type", "input"),
                                "title": event.get("title", "Agent Question"),
                                "message": event.get("message", ""),
                                "options": event.get("options"),
                                "tool_info": event.get("tool_info"),
                                "default_action": event.get("default_action"),
                                "timeout_seconds": event.get("timeout_seconds", 120),
                                "timestamp": event.get("timestamp"),
                            }
                        )
                        if settings.run_ledger_v1:
                            await run_ledger_service.record_event(
                                run_id=run_id,
                                event_type="interrupt",
                                payload={
                                    "type": "interrupt",
                                    "run_id": run_id,
                                    "step_id": event.get("id"),
                                    "interrupt_id": event.get("interrupt_id", ""),
                                    "interrupt_type": event.get("interrupt_type", "input"),
                                    "title": event.get("title", "Agent Question"),
                                    "message": event.get("message", ""),
                                },
                                step_id=event.get("id"),
                                dedup_key=f"interrupt:{event.get('interrupt_id')}",
                            )

                    else:
                        # Use declarative mapping for all other event types
                        sse_payload = _build_sse_payload(event_type, event)
                        if sse_payload is not None:
                            sse_payload["run_id"] = run_id
                            sse_payload["step_id"] = event.get("id")
                            for extra_key in ("policy_decision", "budget_state", "verification"):
                                if extra_key in event and event.get(extra_key) is not None:
                                    sse_payload[extra_key] = event.get(extra_key)
                            yield _sse_data(sse_payload)
                            dedup_key = None
                            if event_type in {"tool_call", "tool_result"}:
                                dedup_key = f"{event_type}:{sse_payload.get('id')}"
                            elif event_type == "stage":
                                dedup_key = f"stage:{sse_payload.get('name')}:{sse_payload.get('status')}"
                            if settings.run_ledger_v1:
                                await run_ledger_service.record_event(
                                    run_id=run_id,
                                    event_type=event_type,
                                    payload=sse_payload,
                                    step_id=sse_payload.get("step_id"),
                                    dedup_key=dedup_key,
                                )

            except asyncio.CancelledError:
                # SSE connection was closed by client
                logger.info(
                    "sse_connection_cancelled",
                    task_id=chat_task_id,
                    user_id=current_user.id,
                )
                if settings.run_ledger_v1:
                    try:
                        await asyncio.shield(
                            run_ledger_service.mark_run_status(run_id, "cancelled")
                        )
                    except asyncio.CancelledError:
                        pass
                raise
            except Exception as e:
                logger.error("agent_stream_error", mode=request.mode.value, error=str(e))
                if settings.run_ledger_v1:
                    await run_ledger_service.mark_run_status(run_id, "failed", last_error=str(e))
                yield _sse_data({"type": "error", "data": str(e)})
            finally:
                # Shield cleanup DB work from cancellation to prevent connection leaks
                async def _cleanup():
                    if settings.run_ledger_v1:
                        run = await run_ledger_service.get_run(run_id)
                        if run and run.get("status") == "running":
                            await run_ledger_service.mark_run_status(run_id, "completed")
                    # Sandbox cleanup is handled by timeout-based expiry (30 min idle),
                    # NOT on SSE disconnect, to keep sandboxes alive across conversation messages.
                    #
                    # However, we eagerly save a snapshot now while the sandbox is still
                    # alive, rather than waiting for the cleanup loop (which may run after
                    # the E2B sandbox has already timed out).
                    try:
                        from app.sandbox import get_execution_sandbox_manager
                        exec_mgr = get_execution_sandbox_manager()
                        await exec_mgr.save_snapshot_for_session(
                            user_id=current_user.id,
                            task_id=chat_task_id,
                        )
                    except Exception as snap_err:
                        logger.debug(
                            "eager_snapshot_failed",
                            task_id=chat_task_id,
                            error=str(snap_err),
                        )

                try:
                    await asyncio.shield(_cleanup())
                except (asyncio.CancelledError, Exception) as cleanup_error:
                    if not isinstance(cleanup_error, asyncio.CancelledError):
                        logger.warning(
                            "sse_disconnect_cleanup_failed",
                            task_id=chat_task_id,
                            error=str(cleanup_error),
                        )

        return StreamingResponse(
            agent_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Note: Research mode is now handled by _CHAT_MODES above
    # (flows through supervisor → task agent → deep_research skill)


@router.get("/status/{task_id}")
async def get_query_status(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Get the status of a research task."""
    task_data = await deep_research_repository.get_task_dict(db, task_id)

    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    # Verify the requesting user owns this task
    if task_data.get("user_id") != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    return task_data
