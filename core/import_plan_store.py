from __future__ import annotations

import json
from typing import Any

from config.settings import AppSettings

IMPORT_PLAN_PREFIX = "rag:import_plan:"
IMPORT_PLAN_LOCK_PREFIX = "rag:import_plan_lock:"
IMPORT_PLAN_TTL_SECONDS = 1800
IMPORT_PLAN_LOCK_SECONDS = 30


def save_import_plan(settings: AppSettings, plan_id: str, payload: dict[str, Any]) -> None:
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required")
    import redis

    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    client.set(
        f"{IMPORT_PLAN_PREFIX}{plan_id}",
        json.dumps(payload, ensure_ascii=False),
        ex=IMPORT_PLAN_TTL_SECONDS,
    )


def load_import_plan(settings: AppSettings, plan_id: str) -> dict[str, Any] | None:
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required")
    import redis

    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    value = client.get(f"{IMPORT_PLAN_PREFIX}{plan_id}")
    if not value:
        return None
    return json.loads(value)


def refresh_import_plan(settings: AppSettings, plan_id: str, payload: dict[str, Any]) -> None:
    save_import_plan(settings, plan_id, payload)


def acquire_import_plan_lock(settings: AppSettings, plan_id: str, token: str) -> bool:
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required")
    import redis

    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    return bool(client.set(f"{IMPORT_PLAN_LOCK_PREFIX}{plan_id}", token, nx=True, ex=IMPORT_PLAN_LOCK_SECONDS))


def release_import_plan_lock(settings: AppSettings, plan_id: str, token: str) -> None:
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required")
    import redis

    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    client.eval(
        """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        end
        return 0
        """,
        1,
        f"{IMPORT_PLAN_LOCK_PREFIX}{plan_id}",
        token,
    )
