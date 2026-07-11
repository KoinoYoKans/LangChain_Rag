from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass

from fastapi import HTTPException

from config.settings import AppSettings


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    retry_after_seconds: int


def rate_limit_key(scope: str, identity: str) -> str:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"rag:rate_limit:{scope}:{digest}"


def _consume(settings: AppSettings, key: str, limit: int, window_seconds: int) -> RateLimitResult:
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required for rate limiting")
    import redis

    client = redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )
    pipeline = client.pipeline()
    pipeline.incr(key)
    pipeline.ttl(key)
    count, ttl = pipeline.execute()
    if int(count) == 1:
        client.expire(key, window_seconds)
        ttl = window_seconds
    retry_after = max(1, int(ttl) if int(ttl) > 0 else window_seconds)
    return RateLimitResult(allowed=int(count) <= limit, retry_after_seconds=retry_after)


async def enforce_rate_limit(
    settings: AppSettings,
    *,
    scope: str,
    identity: str,
    limit: int,
    window_seconds: int,
) -> None:
    """Enforce a Redis-backed fixed-window limit without storing raw identities."""
    try:
        result = await asyncio.to_thread(
            _consume,
            settings,
            rate_limit_key(scope, identity),
            limit,
            window_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        if settings.rate_limit_fail_open:
            return
        raise HTTPException(status_code=503, detail="Rate limiting service is unavailable") from exc
    if not result.allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please retry later.",
            headers={"Retry-After": str(result.retry_after_seconds)},
        )
