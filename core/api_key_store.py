from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select, update

from config.database import (
    ApiKeyModel,
    KnowledgeBaseGrantModel,
    KnowledgeBaseMemberModel,
    KnowledgeBaseModel,
    UserModel,
    build_async_session_maker,
)
from config.settings import AppSettings


@dataclass(frozen=True)
class ApiKey:
    id: str
    org_id: str
    user_id: str
    knowledge_base_id: str
    name: str
    purpose: str | None
    key_prefix: str
    is_active: bool
    expires_at: datetime | None
    daily_request_limit: int | None
    daily_token_limit: int | None
    daily_request_count: int
    daily_token_count: int
    quota_reset_date: Any | None
    last_used_at: datetime | None
    created_at: datetime


@dataclass(frozen=True)
class CreatedApiKey:
    record: ApiKey
    secret: str


@dataclass(frozen=True)
class ApiKeyVerificationResult:
    api_key: ApiKey | None
    reason: str | None
    key_prefix: str | None = None
    org_id: str | None = None
    user_id: str | None = None
    knowledge_base_id: str | None = None


async def create_api_key(
    settings: AppSettings,
    *,
    org_id: str,
    user_id: str,
    knowledge_base_id: str,
    name: str,
    purpose: str | None = None,
    expires_at: datetime | None = None,
    daily_request_limit: int | None = None,
    daily_token_limit: int | None = None,
) -> CreatedApiKey:
    secret = f"rag-{secrets.token_urlsafe(32)}"
    prefix = secret[:12]
    model = ApiKeyModel(
        id=uuid4(),
        org_id=UUID(org_id),
        user_id=UUID(user_id),
        knowledge_base_id=UUID(knowledge_base_id),
        name=name,
        purpose=purpose,
        key_prefix=prefix,
        key_hash=_hash_key(secret),
        is_active=True,
        expires_at=expires_at,
        daily_request_limit=daily_request_limit,
        daily_token_limit=daily_token_limit,
        quota_reset_date=datetime.now(timezone.utc).date(),
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
        conditions = [ApiKeyModel.org_id == UUID(org_id)]
        if knowledge_base_ids is not None:
            if not knowledge_base_ids:
                return []
            conditions.append(ApiKeyModel.knowledge_base_id.in_([UUID(item) for item in knowledge_base_ids]))
        result = await session.scalars(
            select(ApiKeyModel)
            .where(*conditions)
            .order_by(ApiKeyModel.created_at.desc())
        )
        items = list(result)
        for item in items:
            _reset_quota_if_needed(item)
        await session.commit()
        return [_to_dataclass(item) for item in items]


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
    result = await verify_api_key_detailed(settings, secret)
    return result.api_key


async def verify_api_key_detailed(settings: AppSettings, secret: str) -> ApiKeyVerificationResult:
    prefix = secret[:12]
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        row = await session.execute(
            select(ApiKeyModel, KnowledgeBaseModel, UserModel)
            .join(KnowledgeBaseModel, ApiKeyModel.knowledge_base_id == KnowledgeBaseModel.id)
            .join(UserModel, ApiKeyModel.user_id == UserModel.id)
            .where(
                ApiKeyModel.key_prefix == prefix,
            )
        )
        result = row.one_or_none()
        if result is None:
            return ApiKeyVerificationResult(api_key=None, reason="invalid", key_prefix=prefix)
        model, knowledge_base, user = result
        if model is None or model.key_hash != _hash_key(secret):
            return ApiKeyVerificationResult(api_key=None, reason="invalid", key_prefix=prefix)
        if not model.is_active:
            return _verification_failure(model, "disabled")
        if knowledge_base.status == "deleted" or not user.is_active:
            return _verification_failure(model, "inactive_binding")
        if _is_expired(model.expires_at):
            return _verification_failure(model, "expired")
        if not await _api_key_user_can_manage_bound_kb(session, knowledge_base, user):
            return _verification_failure(model, "forbidden")
        _reset_quota_if_needed(model)
        await session.commit()
        return ApiKeyVerificationResult(api_key=_to_dataclass(model), reason=None, key_prefix=prefix)


async def reserve_api_key_usage(settings: AppSettings, api_key_id: str, estimated_tokens: int = 0) -> ApiKeyVerificationResult:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            model = await session.scalar(
                select(ApiKeyModel)
                .where(ApiKeyModel.id == UUID(api_key_id))
                .with_for_update()
            )
            if model is None:
                return ApiKeyVerificationResult(api_key=None, reason="invalid")
            _reset_quota_if_needed(model)
            reserved_tokens = max(0, estimated_tokens)
            if not model.is_active:
                return _verification_failure(model, "disabled")
            if _is_expired(model.expires_at):
                return _verification_failure(model, "expired")
            if model.daily_request_limit is not None and model.daily_request_count + 1 > model.daily_request_limit:
                return _verification_failure(model, "request_quota_exceeded")
            if model.daily_token_limit is not None and model.daily_token_count + reserved_tokens > model.daily_token_limit:
                return _verification_failure(model, "token_quota_exceeded")
            model.daily_request_count += 1
            model.daily_token_count += reserved_tokens
            model.last_used_at = datetime.now(timezone.utc)
        await session.refresh(model)
        return ApiKeyVerificationResult(api_key=_to_dataclass(model), reason=None, key_prefix=model.key_prefix)


async def adjust_api_key_token_usage(settings: AppSettings, api_key_id: str, reserved_tokens: int, actual_tokens: int) -> ApiKey:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            model = await session.scalar(
                select(ApiKeyModel)
                .where(ApiKeyModel.id == UUID(api_key_id))
                .with_for_update()
            )
            if model is None:
                raise ValueError("API key not found")
            today = datetime.now(timezone.utc).date()
            if model.quota_reset_date == today:
                delta = max(0, actual_tokens) - max(0, reserved_tokens)
                model.daily_token_count = max(0, model.daily_token_count + delta)
            model.last_used_at = datetime.now(timezone.utc)
        await session.refresh(model)
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
    if member is not None:
        return True
    direct_grant = await session.scalar(
        select(KnowledgeBaseGrantModel).where(
            KnowledgeBaseGrantModel.knowledge_base_id == knowledge_base.id,
            KnowledgeBaseGrantModel.subject_type == "user",
            KnowledgeBaseGrantModel.subject_id == user.id,
            KnowledgeBaseGrantModel.role == "admin",
        )
    )
    if direct_grant is not None:
        return True
    if user.department_id is None:
        return False
    department_grant = await session.scalar(
        select(KnowledgeBaseGrantModel).where(
            KnowledgeBaseGrantModel.knowledge_base_id == knowledge_base.id,
            KnowledgeBaseGrantModel.subject_type == "department",
            KnowledgeBaseGrantModel.subject_id == user.department_id,
            KnowledgeBaseGrantModel.role == "admin",
        )
    )
    return department_grant is not None


def _hash_key(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _verification_failure(model: ApiKeyModel, reason: str) -> ApiKeyVerificationResult:
    return ApiKeyVerificationResult(
        api_key=None,
        reason=reason,
        key_prefix=model.key_prefix,
        org_id=str(model.org_id),
        user_id=str(model.user_id),
        knowledge_base_id=str(model.knowledge_base_id),
    )


def _is_expired(expires_at: datetime | None) -> bool:
    if expires_at is None:
        return False
    now = datetime.now(timezone.utc)
    if expires_at.tzinfo is None:
        return expires_at <= now.replace(tzinfo=None)
    return expires_at <= now


def _reset_quota_if_needed(model: ApiKeyModel) -> None:
    today = datetime.now(timezone.utc).date()
    if model.quota_reset_date != today:
        model.daily_request_count = 0
        model.daily_token_count = 0
        model.quota_reset_date = today


def _to_dataclass(model: ApiKeyModel) -> ApiKey:
    return ApiKey(
        id=str(model.id),
        org_id=str(model.org_id),
        user_id=str(model.user_id),
        knowledge_base_id=str(model.knowledge_base_id),
        name=model.name,
        purpose=model.purpose,
        key_prefix=model.key_prefix,
        is_active=bool(model.is_active),
        expires_at=model.expires_at,
        daily_request_limit=model.daily_request_limit,
        daily_token_limit=model.daily_token_limit,
        daily_request_count=int(model.daily_request_count or 0),
        daily_token_count=int(model.daily_token_count or 0),
        quota_reset_date=model.quota_reset_date,
        last_used_at=model.last_used_at,
        created_at=model.created_at,
    )
