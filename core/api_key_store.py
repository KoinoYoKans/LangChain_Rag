from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import select, update

from config.database import ApiKeyModel, build_async_session_maker
from config.settings import AppSettings


@dataclass(frozen=True)
class ApiKey:
    id: str
    org_id: str
    user_id: str
    knowledge_base_id: str
    name: str
    key_prefix: str
    is_active: bool
    last_used_at: datetime | None
    created_at: datetime


@dataclass(frozen=True)
class CreatedApiKey:
    record: ApiKey
    secret: str


async def create_api_key(
    settings: AppSettings,
    *,
    org_id: str,
    user_id: str,
    knowledge_base_id: str,
    name: str,
) -> CreatedApiKey:
    secret = f"rag-{secrets.token_urlsafe(32)}"
    prefix = secret[:12]
    model = ApiKeyModel(
        id=uuid4(),
        org_id=UUID(org_id),
        user_id=UUID(user_id),
        knowledge_base_id=UUID(knowledge_base_id),
        name=name,
        key_prefix=prefix,
        key_hash=_hash_key(secret),
        is_active=True,
    )
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            session.add(model)
        await session.refresh(model)
        return CreatedApiKey(record=_to_dataclass(model), secret=secret)


async def list_api_keys(settings: AppSettings, org_id: str) -> list[ApiKey]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        result = await session.scalars(
            select(ApiKeyModel)
            .where(ApiKeyModel.org_id == UUID(org_id))
            .order_by(ApiKeyModel.created_at.desc())
        )
        return [_to_dataclass(item) for item in result]


async def verify_api_key(settings: AppSettings, secret: str) -> ApiKey | None:
    prefix = secret[:12]
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        model = await session.scalar(
            select(ApiKeyModel).where(ApiKeyModel.key_prefix == prefix, ApiKeyModel.is_active.is_(True))
        )
        if model is None or model.key_hash != _hash_key(secret):
            return None
        await session.execute(
            update(ApiKeyModel)
            .where(ApiKeyModel.id == model.id)
            .values(last_used_at=datetime.utcnow())
        )
        await session.commit()
        return _to_dataclass(model)


def _hash_key(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _to_dataclass(model: ApiKeyModel) -> ApiKey:
    return ApiKey(
        id=str(model.id),
        org_id=str(model.org_id),
        user_id=str(model.user_id),
        knowledge_base_id=str(model.knowledge_base_id),
        name=model.name,
        key_prefix=model.key_prefix,
        is_active=bool(model.is_active),
        last_used_at=model.last_used_at,
        created_at=model.created_at,
    )
