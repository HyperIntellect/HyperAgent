"""Task Queue Service - Abstraction layer for submitting background jobs."""

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from arq import ArqRedis, create_pool

from app.core.logging import get_logger
from app.workers.config import get_redis_settings

logger = get_logger(__name__)


class TaskQueueService:
    """Service for enqueueing background tasks."""

    def __init__(self):
        self._pool: ArqRedis | None = None

    async def get_pool(self) -> ArqRedis:
        """Get or create Redis connection pool."""
        if self._pool is None:
            self._pool = await create_pool(get_redis_settings())
        return self._pool

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def enqueue_research_task(
        self,
        task_id: str,
        query: str,
        depth: str,
        user_id: str,
        locale: str = "en",
        priority: int = 0,
        delay: timedelta | None = None,
    ) -> str:
        """
        Enqueue a research task for background processing.

        Args:
            task_id: Database task ID (should already exist in DB)
            query: Research query
            depth: Research depth level
            user_id: User ID (required)
            locale: User's preferred language (e.g., 'en', 'zh-CN')
            priority: Job priority (higher = more urgent)
            delay: Optional delay before processing

        Returns:
            ARQ job ID
        """
        # Check for duplicate query within 5-minute window
        query_hash = hashlib.sha256(f"{user_id}:{query}".encode()).hexdigest()[:16]
        dedup_key = f"hyperagent:dedup:{query_hash}"
        pool = await self.get_pool()

        existing = await pool.get(dedup_key)
        if existing:
            job_id_str = existing.decode() if isinstance(existing, bytes) else existing
            logger.info(
                "duplicate_task_skipped",
                task_id=task_id,
                existing_job_id=job_id_str,
            )
            return job_id_str

        job = await pool.enqueue_job(
            "run_research_task",
            task_id=task_id,
            query=query,
            depth=depth,
            user_id=user_id,
            locale=locale,
            _defer_by=delay or timedelta(seconds=1),
            _job_id=f"research:{task_id}",
        )

        await pool.set(dedup_key, job.job_id, ex=300)  # 5-minute TTL

        logger.info(
            "task_enqueued",
            task_id=task_id,
            job_id=job.job_id,
            function="run_research_task",
        )

        return job.job_id

    async def enqueue_batch_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        priority: int = 0,
    ) -> str:
        """
        Enqueue a generic batch task.

        Args:
            task_type: Type of batch operation (function name)
            payload: Task parameters
            priority: Job priority

        Returns:
            ARQ job ID
        """
        pool = await self.get_pool()

        job_id = f"batch:{task_type}:{uuid.uuid4().hex[:8]}"

        job = await pool.enqueue_job(
            task_type,
            **payload,
            _job_id=job_id,
        )

        logger.info(
            "batch_task_enqueued",
            job_id=job.job_id,
            task_type=task_type,
        )

        return job.job_id

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """
        Get the status of a queued job.

        Args:
            job_id: ARQ job ID

        Returns:
            Job status dict or None if not found
        """
        pool = await self.get_pool()

        job = await pool.job(job_id)
        if job is None:
            return None

        info = await job.info()
        if info is None:
            return {"job_id": job_id, "status": "unknown"}

        return {
            "job_id": job_id,
            "function": info.function,
            "status": info.status.value if info.status else "unknown",
            "enqueue_time": info.enqueue_time.isoformat() if info.enqueue_time else None,
            "start_time": info.start_time.isoformat() if info.start_time else None,
            "finish_time": info.finish_time.isoformat() if info.finish_time else None,
            "success": info.success,
            "result": info.result,
        }

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending job.

        Args:
            job_id: ARQ job ID

        Returns:
            True if cancelled, False if not found or already running
        """
        pool = await self.get_pool()

        job = await pool.job(job_id)
        if job is None:
            return False

        await job.abort()
        logger.info("job_cancelled", job_id=job_id)
        return True

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        pool = await self.get_pool()

        # Get queue length
        queue_length = await pool.queued_jobs()

        return {
            "queued_jobs": len(queue_length) if queue_length else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# Global instance
task_queue = TaskQueueService()
