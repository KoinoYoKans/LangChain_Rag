from __future__ import annotations

from config.settings import AppSettings

INGEST_QUEUE = "rag:ingest_jobs"


def enqueue_ingest_job(settings: AppSettings, job_id: str) -> None:
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required")
    import redis

    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    client.rpush(INGEST_QUEUE, job_id)


def wait_for_ingest_job(settings: AppSettings, timeout: int = 5) -> str | None:
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required")
    import redis

    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        result = client.blpop(INGEST_QUEUE, timeout=timeout)
    except redis.exceptions.TimeoutError:
        return None
    if result is None:
        return None
    return str(result[1])
