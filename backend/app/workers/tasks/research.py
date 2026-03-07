"""Research task handler for background worker."""

import asyncio
import uuid
from typing import Any

from redis.asyncio import Redis

from app.agents import agent_orchestrator as agent_supervisor
from app.config import settings
from app.core.logging import get_logger
from app.db.base import async_session_maker
from app.models.schemas import ResearchDepth
from app.repository import deep_research_repository
from app.workers.progress import ProgressReporter

logger = get_logger(__name__)

# Progress percentages for each step
STEP_PROGRESS = {
    "search": 25,
    "analyze": 50,
    "synthesize": 75,
    "write": 90,
}


async def run_research_task(
    ctx: dict,
    task_id: str,
    query: str,
    depth: str,
    user_id: str,
    locale: str = "en",
) -> dict[str, Any]:
    """
    Execute a research task in the background worker.

    This is the main task handler that replaces the inline SSE processing.
    It runs the research agent and updates the database with progress and results.

    Args:
        ctx: ARQ context (contains redis connection, job info)
        task_id: Unique task identifier
        query: Research query string
        depth: Research depth (quick, standard, deep)
        user_id: User ID for task ownership (required)
        locale: User's preferred language (e.g., 'en', 'zh-CN')

    Returns:
        Result dictionary with task_id and status
    """
    redis: Redis = ctx["redis"]
    job_id = ctx["job_id"]
    worker_name = ctx.get("worker_name")

    logger.info(
        "research_task_started",
        task_id=task_id,
        job_id=job_id,
        query=query[:50],
        depth=depth,
    )

    # Initialize progress reporter for real-time updates
    progress = ProgressReporter(redis, task_id)

    async with async_session_maker() as db:
        try:
            # Verify task exists in database before proceeding
            task = await deep_research_repository.get_task(db, task_id)
            if task is None:
                raise ValueError(f"Task {task_id} not found in database")

            # Clear any existing data from previous runs (retry scenario)
            await deep_research_repository.clear_task_steps(db, task_id)
            await deep_research_repository.clear_task_sources(db, task_id)
            await deep_research_repository.update_task_report(db, task_id, "")

            # Update task status to running
            await deep_research_repository.update_task_status(db, task_id, "running")
            await deep_research_repository.update_task_worker_info(db, task_id, job_id, worker_name)
            await db.commit()

            # Emit progress: starting
            await progress.emit("task_started", {"task_id": task_id})

            report_content: list[str] = []
            source_count = 0
            step_ids: dict[str, str] = {}
            token_buffer: list[str] = []
            last_flush_time = asyncio.get_event_loop().time()

            # Run the research agent via supervisor with timeout protection
            async with asyncio.timeout(settings.research_task_timeout):
                async for event in agent_supervisor.run(
                    query=query,
                    mode="research",
                    task_id=task_id,
                    user_id=user_id,
                    depth=ResearchDepth(depth),
                    locale=locale,
                ):
                    # Handle stage events (supervisor uses "stage", not "step")
                    if event["type"] == "stage":
                        step_type = event.get("name", "")
                        status = event.get("status", "running")

                        description = event.get("description", step_type)

                        try:
                            if status == "running":
                                # Create new step in database
                                step_id = str(uuid.uuid4())
                                await deep_research_repository.add_step(
                                    db=db,
                                    task_id=task_id,
                                    step_id=step_id,
                                    step_type=step_type,
                                    description=description,
                                    status="running",
                                )
                                step_ids[step_type] = step_id

                                # Emit progress event
                                await progress.emit_step(
                                    step_type=step_type,
                                    description=description,
                                    status="running",
                                    step_id=step_id,
                                )
                            else:
                                # Update existing step status
                                if step_type in step_ids:
                                    await deep_research_repository.update_step_status(
                                        db, step_ids[step_type], status
                                    )
                                    await progress.emit_step(
                                        step_type=step_type,
                                        description=description,
                                        status=status,
                                        step_id=step_ids[step_type],
                                    )

                            # Update progress percentage based on step
                            if step_type in STEP_PROGRESS:
                                await deep_research_repository.update_task_progress(
                                    db, task_id, STEP_PROGRESS[step_type]
                                )
                                await progress.emit_progress(STEP_PROGRESS[step_type], step_type)

                            if status != "running":
                                await db.commit()
                        except Exception as step_error:
                            # Handle FK violation or other DB errors gracefully
                            logger.error(
                                "step_insert_failed",
                                task_id=task_id,
                                step_type=step_type,
                                error=str(step_error),
                            )
                            await db.rollback()
                            # Continue processing - emit progress even if DB insert fails
                            await progress.emit_step(
                                step_type=step_type,
                                description=description,
                                status=status,
                                step_id=step_ids.get(step_type, str(uuid.uuid4())),
                            )

                    elif event["type"] == "source":
                        # Create source in database
                        source_id = str(uuid.uuid4())
                        source_count += 1
                        try:
                            await deep_research_repository.add_source(
                                db=db,
                                task_id=task_id,
                                source_id=source_id,
                                title=event["title"],
                                url=event["url"],
                                snippet=event.get("snippet"),
                                relevance_score=event.get("relevance_score"),
                            )
                            if source_count % 5 == 0:
                                await db.commit()
                        except Exception as source_error:
                            # Handle FK violation or other DB errors gracefully
                            logger.error(
                                "source_insert_failed",
                                task_id=task_id,
                                source_id=source_id,
                                error=str(source_error),
                            )
                            await db.rollback()

                        # Emit source event regardless of DB success
                        await progress.emit_source(
                            source_id=source_id,
                            title=event["title"],
                            url=event["url"],
                            snippet=event.get("snippet"),
                        )

                    elif event["type"] == "tool_call":
                        # Forward tool call events to frontend
                        await progress.emit_tool_call(
                            tool=event.get("tool", ""),
                            args=event.get("args", {}),
                            tool_id=event.get("id"),
                        )

                    elif event["type"] == "tool_result":
                        # Forward tool result events to frontend
                        await progress.emit_tool_result(
                            tool=event.get("tool", ""),
                            output=str(event.get("content", "")),
                            tool_id=event.get("id"),
                        )

                    elif event["type"] == "token":
                        report_content.append(event["content"])
                        token_buffer.append(event["content"])

                        # Time-based flush: emit every 100ms
                        now = asyncio.get_event_loop().time()
                        if now - last_flush_time >= 0.1:
                            await progress.emit_token_batch("".join(token_buffer))
                            token_buffer.clear()
                            last_flush_time = now

            # Flush remaining tokens
            if token_buffer:
                await progress.emit_token_batch("".join(token_buffer))

            # Save final report
            full_report = "".join(report_content)
            await deep_research_repository.update_task_report(db, task_id, full_report)
            await deep_research_repository.update_task_status(db, task_id, "completed")
            await deep_research_repository.update_task_progress(db, task_id, 100)
            await db.commit()

            await progress.emit_complete()

            logger.info(
                "research_task_completed",
                task_id=task_id,
                job_id=job_id,
                report_length=len(full_report),
            )

            return {
                "task_id": task_id,
                "status": "completed",
                "report_length": len(full_report),
            }

        except asyncio.TimeoutError:
            timeout_minutes = settings.research_task_timeout // 60
            error_msg = f"Research task timed out after {timeout_minutes} minutes"
            logger.error(
                "research_task_timeout",
                task_id=task_id,
                job_id=job_id,
                timeout_seconds=settings.research_task_timeout,
            )

            await deep_research_repository.update_task_status(
                db, task_id, "failed", error=error_msg
            )
            await db.commit()

            await progress.emit_error(error_msg)

            raise

        except Exception as e:
            logger.error(
                "research_task_failed",
                task_id=task_id,
                job_id=job_id,
                error=str(e),
            )

            await deep_research_repository.update_task_status(
                db, task_id, "failed", error=str(e)
            )
            await db.commit()

            await progress.emit_error(str(e))

            # Re-raise for ARQ retry mechanism
            raise
