"""
Worker entry point for ARQ background job processing.

Run with: python -m app.workers.main
Or: arq app.workers.main.WorkerSettings
"""

import random
from typing import Any

from arq import cron

from app.config import settings
from app.core.logging import get_logger, setup_logging
from app.workers.config import get_redis_settings
from app.workers.tasks.research import run_research_task

# Initialize logging
setup_logging(log_level=settings.log_level, log_format=settings.log_format)
logger = get_logger(__name__)


def calculate_retry_delay(retry_count: int) -> int:
    """
    Calculate exponential backoff delay with jitter.

    Uses exponential backoff: base * (2 ^ retry_count) with random jitter (10-30%)
    to prevent thundering herd problems when multiple jobs retry simultaneously.

    Args:
        retry_count: Number of previous retry attempts (0-indexed)

    Returns:
        Delay in seconds before next retry
    """
    base_delay = 5  # Start with 5 seconds
    max_delay = 60  # Cap at 60 seconds

    # Calculate exponential delay: 5s, 10s, 20s, 40s, capped at 60s
    delay = min(base_delay * (2 ** retry_count), max_delay)

    # Add jitter (10-30% of delay) to prevent thundering herd
    jitter = delay * random.uniform(0.1, 0.3)

    total_delay = int(delay + jitter)
    logger.info(
        "retry_delay_calculated",
        retry_count=retry_count,
        base_delay=delay,
        jitter=jitter,
        total_delay=total_delay,
    )
    return total_delay


async def startup(ctx: dict) -> None:
    """Worker startup hook - initialize resources."""
    # Note: worker_name is None during startup - ARQ sets it after initialization
    logger.info("worker_initializing")

    # Initialize database connection pool
    from app.db.base import init_db

    await init_db(create_tables=False)

    logger.info("worker_ready", max_jobs=ctx.get("max_jobs", 10))


async def shutdown(ctx: dict) -> None:
    """Worker shutdown hook - cleanup resources."""
    logger.info("worker_shutting_down")

    # Close database connections
    from app.db.base import close_db

    await close_db()

    logger.info("worker_stopped")


async def cleanup_stale_tasks(ctx: dict) -> int:
    """Mark tasks stuck in 'running' beyond job_timeout as failed."""
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import update

    from app.db.base import async_session_maker
    from app.db.models import ResearchTask

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=WorkerSettings.job_timeout)

    async with async_session_maker() as db:
        result = await db.execute(
            update(ResearchTask)
            .where(
                ResearchTask.status == "running",
                ResearchTask.started_at < cutoff,
            )
            .values(
                status="failed",
                error="Task timed out (stale task cleanup)",
            )
        )
        await db.commit()
        count = result.rowcount
        if count > 0:
            logger.info("stale_tasks_cleaned", count=count)
        return count


class WorkerSettings:
    """ARQ Worker Settings - loaded by arq CLI."""

    # Redis connection
    redis_settings = get_redis_settings()

    # Task functions to register
    functions = [
        run_research_task,
    ]

    # Scheduled/cron jobs
    cron_jobs = [
        cron(cleanup_stale_tasks, hour={0, 6, 12, 18}),  # Run every 6 hours
    ]

    # Lifecycle hooks
    on_startup = startup
    on_shutdown = shutdown

    # Worker settings
    max_jobs = 10
    job_timeout = 1800  # 30 minutes
    keep_result = 86400  # 24 hours
    # 0.5s poll_delay adds up to 500ms latency for time-sensitive jobs.
    # Lower to 0.1s if near-real-time responsiveness is needed.
    poll_delay = 0.5  # Seconds between queue polls

    # Worker name (customize per instance if running multiple workers)
    name = "hyperagent-worker"

    # Retry settings with exponential backoff
    max_tries = 3
    # Use custom retry_jobs function for exponential backoff instead of fixed delay
    # Delays will be approximately: 5s, 10s, 20s (with jitter)

    # Logging
    log_results = True


async def retry_jobs(ctx: dict[str, Any]) -> int:
    """
    Custom retry handler that implements exponential backoff.

    ARQ calls this function to determine the delay before retrying a failed job.
    This replaces the fixed retry_delay with exponential backoff + jitter.

    Args:
        ctx: ARQ job context containing job_try (current attempt number)

    Returns:
        Delay in seconds before the next retry attempt
    """
    job_try = ctx.get("job_try", 1)
    # job_try is 1-indexed, so subtract 1 for retry_count (0-indexed)
    retry_count = job_try - 1
    return calculate_retry_delay(retry_count)


# Register the retry function in WorkerSettings
WorkerSettings.retry_jobs = staticmethod(retry_jobs)


if __name__ == "__main__":
    import asyncio

    from arq import run_worker

    asyncio.run(run_worker(WorkerSettings))
