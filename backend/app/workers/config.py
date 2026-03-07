"""ARQ Worker configuration."""

from urllib.parse import urlparse

from arq.connections import RedisSettings

from app.config import settings


def get_redis_settings() -> RedisSettings:
    """Parse Redis URL into ARQ RedisSettings."""
    parsed = urlparse(settings.redis_url)

    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password,
        database=int(parsed.path.lstrip("/") or 0) if parsed.path else 0,
    )
