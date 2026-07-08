from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select, update

from config.database import ApiKeyModel, KnowledgeBaseMemberModel, KnowledgeBaseModel, UserModel, build_async_session_maker
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


async def list_api_keys(settings: AppSettings, org_id: str, knowledge_base_ids: list[str] | None = None) -> list[ApiKey]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        conditions = [ApiKeyModel.org_id == UUID(org_id), ApiKeyModel.is_active.is_(True)]
        if knowledge_base_ids is not None:
            if not knowledge_base_ids:
                return []
            conditions.append(ApiKeyModel.knowledge_base_id.in_([UUID(item) for item in knowledge_base_ids]))
        result = await session.scalars(
            select(ApiKeyModel)
            .where(*conditions)
            .order_by(ApiKeyModel.created_at.desc())
        )
        return [_to_dataclass(item) for item in result]


async def get_api_key(settings: AppSettings, org_id: str, api_key_id: str) -> ApiKey | None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        model = await session.scalar(
            select(ApiKeyModel).where(
                ApiKeyModel.id == UUID(api_key_id),
                ApiKeyModel.org_id == UUID(org_id),
                ApiKeyModel.is_active.is_(True),
            )
        )
        return _to_dataclass(model) if model else None


async def revoke_api_key(settings: AppSettings, org_id: str, api_key_id: str) -> ApiKey | None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            model = await session.scalar(
                select(ApiKeyModel).where(
                    ApiKeyModel.id == UUID(api_key_id),
                    ApiKeyModel.org_id == UUID(org_id),
                    ApiKeyModel.is_active.is_(True),
                )
            )
            if model is None:
                return None
            model.is_active = False
        await session.refresh(model)
        return _to_dataclass(model)


async def revoke_api_keys_for_kb_user(settings: AppSettings, *, kb_id: str, user_id: str) -> int:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            result = await session.execute(
                update(ApiKeyModel)
                .where(
                    ApiKeyModel.knowledge_base_id == UUID(kb_id),
                    ApiKeyModel.user_id == UUID(user_id),
                    ApiKeyModel.is_active.is_(True),
                )
                .values(is_active=False)
            )
            return int(result.rowcount or 0)


async def verify_api_key(settings: AppSettings, secret: str) -> ApiKey | None:
    prefix = secret[:12]
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        row = await session.execute(
            select(ApiKeyModel, KnowledgeBaseModel, UserModel)
            .join(KnowledgeBaseModel, ApiKeyModel.knowledge_base_id == KnowledgeBaseModel.id)
            .join(UserModel, ApiKeyModel.user_id == UserModel.id)
            .where(
                ApiKeyModel.key_prefix == prefix,
                ApiKeyModel.is_active.is_(True),
                KnowledgeBaseModel.status != "deleted",
                UserModel.is_active.is_(True),
            )
        )
        result = row.one_or_none()
        if result is None:
            return None
        model, knowledge_base, user = result
        if model is None or model.key_hash != _hash_key(secret):
            return None
        if not await _api_key_user_can_manage_bound_kb(session, knowledge_base, user):
            return None
        await session.execute(
            update(ApiKeyModel)
            .where(ApiKeyModel.id == model.id)
            .values(last_used_at=datetime.utcnow())
        )
        await session.commit()
        return _to_dataclass(model)


async def _api_key_user_can_manage_bound_kb(session: Any, knowledge_base: KnowledgeBaseModel, user: UserModel) -> bool:
    if user.role == "admin":
        return True
    if knowledge_base.owner_user_id == user.id:
        return True
    member = await session.scalar(
        select(KnowledgeBaseMemberModel).where(
            KnowledgeBaseMemberModel.knowledge_base_id == knowledge_base.id,
            KnowledgeBaseMemberModel.user_id == user.id,
            KnowledgeBaseMemberModel.role == "owner",
        )
    )
    return member is not None


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
