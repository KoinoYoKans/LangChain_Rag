from __future__ import annotations

from datetime import datetime, timezone

from config.settings import AppSettings

INGEST_QUEUE = "rag:ingest_jobs"
WORKER_HEARTBEAT_KEY = "rag:ingest_worker:last_seen"


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


def get_ingest_queue_length(settings: AppSettings) -> int:
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required")
    import redis

    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    return int(client.llen(INGEST_QUEUE))


def record_worker_heartbeat(settings: AppSettings) -> None:
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required")
    import redis

    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    client.set(WORKER_HEARTBEAT_KEY, datetime.now(timezone.utc).isoformat(), ex=120)


def get_worker_heartbeat(settings: AppSettings) -> str | None:
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required")
    import redis

    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    value = client.get(WORKER_HEARTBEAT_KEY)
    return str(value) if value else None
