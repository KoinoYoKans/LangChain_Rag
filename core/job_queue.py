from __future__ import annotations

from datetime import datetime, timezone

from config.settings import AppSettings

INGEST_QUEUE = "rag:ingest_jobs"
WORKER_HEARTBEAT_KEY = "rag:ingest_worker:last_seen"


def _redis_client(settings: AppSettings):
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required")
    import redis

    return redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=8,
    )


def enqueue_ingest_job(settings: AppSettings, job_id: str) -> None:
    _redis_client(settings).rpush(INGEST_QUEUE, job_id)


def wait_for_ingest_job(settings: AppSettings, timeout: int = 5) -> str | None:
    import redis

    client = _redis_client(settings)
    try:
        result = client.blpop(INGEST_QUEUE, timeout=timeout)
    except redis.exceptions.TimeoutError:
        return None
    if result is None:
        return None
    return str(result[1])


def get_ingest_queue_length(settings: AppSettings) -> int:
    return int(_redis_client(settings).llen(INGEST_QUEUE))


def record_worker_heartbeat(settings: AppSettings) -> None:
    _redis_client(settings).set(WORKER_HEARTBEAT_KEY, datetime.now(timezone.utc).isoformat(), ex=120)


def get_worker_heartbeat(settings: AppSettings) -> str | None:
    value = _redis_client(settings).get(WORKER_HEARTBEAT_KEY)
    return str(value) if value else None
