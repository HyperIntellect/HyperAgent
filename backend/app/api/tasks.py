"""
Task management API endpoints.

Provides endpoints for:
- Submitting new background tasks
- Querying task status
- Cancelling pending tasks
- Streaming task progress via SSE
"""

import json
import uuid
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from redis.asyncio import ConnectionPool, Redis
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.core.auth import CurrentUser, get_current_user
from app.core.logging import get_logger
from app.db.base import get_db
from app.db.models import ResearchTask
from app.models.schemas import ResearchDepth
from app.repository import deep_research_repository
from app.workers.task_queue import task_queue

logger = get_logger(__name__)

_redis_pool: ConnectionPool | None = None


def _get_redis_pool() -> ConnectionPool:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = ConnectionPool.from_url(settings.redis_url, decode_responses=True)
    return _redis_pool


router = APIRouter(prefix="/tasks", tags=["tasks"])


# === Request/Response Models ===


class SubmitResearchRequest(BaseModel):
    """Request to submit a research task."""

    query: str = Field(..., min_length=1, max_length=5000)
    depth: ResearchDepth = ResearchDepth.FAST


class TaskStatusResponse(BaseModel):
    """Task status response."""

    id: str
    status: str
    progress: int
    query: str
    depth: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    worker_job_id: str | None = None


class TaskSubmitResponse(BaseModel):
    """Response after submitting a task."""

    task_id: str
    job_id: str
    status: str = "queued"


class TaskListItem(BaseModel):
    """Task list item."""

    id: str
    query: str
    status: str
    progress: int
    created_at: str


class TaskListResponse(BaseModel):
    """Task list response."""

    tasks: list[TaskListItem]
    total: int
    limit: int
    offset: int


# === Endpoints ===


@router.post("/research", response_model=TaskSubmitResponse)
async def submit_research_task(
    request: SubmitResearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Submit a research task for background processing.

    The task is created in the database and enqueued for processing.
    Use the returned task_id to check status or stream progress.
    """
    task_id = str(uuid.uuid4())

    # Create task in database first
    await deep_research_repository.create_task(
        db=db,
        task_id=task_id,
        query=request.query,
        depth=request.depth.value,

        user_id=current_user.id,
    )

    # Update status to queued
    await deep_research_repository.update_task_status(db, task_id, "queued")
    await db.commit()

    # Enqueue for background processing
    job_id = await task_queue.enqueue_research_task(
        task_id=task_id,
        query=request.query,
        depth=request.depth.value,

        user_id=current_user.id,
    )

    logger.info(
        "research_task_submitted",
        task_id=task_id,
        job_id=job_id,
        user_id=current_user.id,
    )

    return TaskSubmitResponse(
        task_id=task_id,
        job_id=job_id,
        status="queued",
    )


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Get the current status of a task."""
    task = await deep_research_repository.get_task(db, task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Verify ownership
    if task.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    return TaskStatusResponse(
        id=task.id,
        status=task.status,
        progress=task.progress,
        query=task.query,
        depth=task.depth,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        error=task.error,
        worker_job_id=task.worker_job_id,
    )


@router.get("/{task_id}/result")
async def get_task_result(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Get the full result of a completed task."""
    task_data = await deep_research_repository.get_task_dict(db, task_id)

    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    # Verify ownership
    if task_data["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    return task_data


@router.delete("/{task_id}")
async def delete_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Delete a task. If running, it will be cancelled first."""
    task = await deep_research_repository.get_task(db, task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # If task is active, try to cancel in queue first
    if task.status in ("pending", "queued", "running"):
        job_id = task.worker_job_id
        if job_id:
            try:
                await task_queue.cancel_job(job_id)
            except Exception as e:
                logger.warning("failed_to_cancel_job", job_id=job_id, error=str(e))

    # Delete from database
    await deep_research_repository.delete_task(db, task_id)
    await db.commit()

    return {"task_id": task_id, "status": "deleted"}


@router.get("/{task_id}/stream")
async def stream_task_progress(
    task_id: str,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Stream real-time task progress via SSE.

    Connects to Redis pub/sub channel for the task and forwards
    all progress events to the client.
    """
    # Verify task ownership before starting stream
    from app.db.base import get_db_session

    async with get_db_session() as db:
        task = await deep_research_repository.get_task(db, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

    async def event_generator() -> AsyncGenerator[dict, None]:
        redis = Redis(connection_pool=_get_redis_pool())
        pubsub = redis.pubsub()
        channel = f"hyperagent:progress:{task_id}"

        try:
            await pubsub.subscribe(channel)

            # Send initial connection event
            yield {
                "event": "connected",
                "data": json.dumps({"task_id": task_id}),
            }

            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    yield {
                        "event": "message",
                        "data": json.dumps(data),
                    }

                    # Close stream on completion
                    if data.get("type") in ("complete", "error"):
                        break

        except Exception as e:
            logger.error("stream_error", task_id=task_id, error=str(e))
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }
        finally:
            await pubsub.unsubscribe(channel)

    return EventSourceResponse(event_generator())


@router.get("/", response_model=TaskListResponse)
async def list_tasks(
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    """List user's tasks with optional filtering."""
    base_filter = ResearchTask.user_id == current_user.id

    # Count total matching tasks
    count_query = select(func.count()).select_from(ResearchTask).where(base_filter)
    if status:
        count_query = count_query.where(ResearchTask.status == status)
    total = (await db.execute(count_query)).scalar_one()

    # Fetch paginated results
    list_query = (
        select(ResearchTask)
        .where(base_filter)
        .order_by(desc(ResearchTask.created_at))
        .limit(limit)
        .offset(offset)
    )
    if status:
        list_query = list_query.where(ResearchTask.status == status)

    result = await db.execute(list_query)
    tasks = result.scalars().all()

    return TaskListResponse(
        tasks=[
            TaskListItem(
                id=t.id,
                query=t.query,
                status=t.status,
                progress=t.progress,
                created_at=t.created_at.isoformat(),
            )
            for t in tasks
        ],
        total=total,
        limit=limit,
        offset=offset,
    )
