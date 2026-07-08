from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert

from config.database import (
    ApiKeyModel,
    AuditLogModel,
    ChatLogModel,
    DepartmentModel,
    IngestJobModel,
    KnowledgeBaseMemberModel,
    KnowledgeBaseModel,
    OrganizationModel,
    RagFileModel,
    UserModel,
    build_async_session_maker,
)
from config.settings import AppSettings


ADMIN_ROLES = {"admin"}
WRITE_ROLES = {"admin", "manager"}


@dataclass(frozen=True)
class Organization:
    id: str
    name: str
    created_at: datetime


@dataclass(frozen=True)
class Department:
    id: str
    org_id: str
    name: str
    parent_id: str | None
    created_at: datetime


@dataclass(frozen=True)
class EnterpriseUser:
    id: str
    org_id: str
    department_id: str | None
    email: str
    display_name: str
    password_hash: str
    role: str
    is_active: bool
    last_login_at: datetime | None
    created_at: datetime


@dataclass(frozen=True)
class KnowledgeBase:
    id: str
    org_id: str
    owner_user_id: str
    name: str
    description: str | None
    visibility: str
    department_ids: list[str]
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class KnowledgeBaseStats:
    file_count: int
    completed_file_count: int
    failed_job_count: int


@dataclass(frozen=True)
class KnowledgeBaseMember:
    id: str
    knowledge_base_id: str
    user_id: str
    role: str
    email: str | None
    display_name: str | None
    department_id: str | None
    created_at: datetime


async def bootstrap_default_org(settings: AppSettings) -> EnterpriseUser:
    from core.auth import hash_password

    session_maker = build_async_session_maker(settings)
    org_id = uuid4()
    department_id = uuid4()
    user_id = uuid4()
    async with session_maker() as session:
        async with session.begin():
            org = await session.scalar(select(OrganizationModel).where(OrganizationModel.name == settings.default_org_name))
            if org is None:
                org = OrganizationModel(id=org_id, name=settings.default_org_name)
                session.add(org)
            dept = await session.scalar(
                select(DepartmentModel).where(
                    DepartmentModel.org_id == org.id,
                    DepartmentModel.name == settings.default_department_name,
                )
            )
            if dept is None:
                dept = DepartmentModel(id=department_id, org_id=org.id, name=settings.default_department_name)
                session.add(dept)
            user = await session.scalar(
                select(UserModel).where(
                    UserModel.org_id == org.id,
                    UserModel.email == settings.default_admin_email,
                )
            )
            if user is None:
                user = UserModel(
                    id=user_id,
                    org_id=org.id,
                    department_id=dept.id,
                    email=settings.default_admin_email,
                    display_name="Administrator",
                    password_hash=hash_password(settings.default_admin_password),
                    role="admin",
                    is_active=True,
                )
                session.add(user)
        await session.refresh(user)
        return _user_to_dataclass(user)


async def authenticate_user(settings: AppSettings, email: str, password: str) -> EnterpriseUser | None:
    from core.auth import verify_password

    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        user = await session.scalar(select(UserModel).where(UserModel.email == email, UserModel.is_active.is_(True)))
        if user is None or not verify_password(password, user.password_hash):
            return None
        await session.execute(update(UserModel).where(UserModel.id == user.id).values(last_login_at=func.now()))
        await session.commit()
        await session.refresh(user)
        return _user_to_dataclass(user)


async def get_user_by_id(settings: AppSettings, user_id: str) -> EnterpriseUser | None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        user = await session.get(UserModel, UUID(user_id))
        return _user_to_dataclass(user) if user else None


async def list_users(settings: AppSettings, org_id: str) -> list[EnterpriseUser]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        result = await session.scalars(select(UserModel).where(UserModel.org_id == UUID(org_id)).order_by(UserModel.created_at.desc()))
        return [_user_to_dataclass(item) for item in result]


async def create_user(
    settings: AppSettings,
    *,
    org_id: str,
    department_id: str | None,
    email: str,
    display_name: str,
    password: str,
    role: str,
) -> EnterpriseUser:
    from core.auth import hash_password

    user = UserModel(
        id=uuid4(),
        org_id=UUID(org_id),
        department_id=UUID(department_id) if department_id else None,
        email=email,
        display_name=display_name,
        password_hash=hash_password(password),
        role=role,
        is_active=True,
    )
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            session.add(user)
        await session.refresh(user)
        return _user_to_dataclass(user)


async def update_user(
    settings: AppSettings,
    *,
    org_id: str,
    user_id: str,
    display_name: str,
    role: str,
    department_id: str | None,
    is_active: bool,
) -> EnterpriseUser:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            user = await session.get(UserModel, UUID(user_id))
            if user is None or user.org_id != UUID(org_id):
                from fastapi import HTTPException

                raise HTTPException(status_code=404, detail="User not found")
            user.display_name = display_name
            user.role = role
            user.department_id = UUID(department_id) if department_id else None
            user.is_active = is_active
        await session.refresh(user)
        return _user_to_dataclass(user)


async def reset_user_password(settings: AppSettings, *, org_id: str, user_id: str, password: str) -> EnterpriseUser:
    from core.auth import hash_password

    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            user = await session.get(UserModel, UUID(user_id))
            if user is None or user.org_id != UUID(org_id):
                from fastapi import HTTPException

                raise HTTPException(status_code=404, detail="User not found")
            user.password_hash = hash_password(password)
        await session.refresh(user)
        return _user_to_dataclass(user)


async def deactivate_user(settings: AppSettings, *, org_id: str, user_id: str) -> EnterpriseUser:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            user = await session.get(UserModel, UUID(user_id))
            if user is None or user.org_id != UUID(org_id):
                from fastapi import HTTPException

                raise HTTPException(status_code=404, detail="User not found")
            user.is_active = False
            await session.execute(
                update(ApiKeyModel)
                .where(ApiKeyModel.user_id == user.id)
                .values(is_active=False)
            )
        await session.refresh(user)
        return _user_to_dataclass(user)


async def list_departments(settings: AppSettings, org_id: str) -> list[Department]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        result = await session.scalars(
            select(DepartmentModel).where(DepartmentModel.org_id == UUID(org_id)).order_by(DepartmentModel.name.asc())
        )
        return [_department_to_dataclass(item) for item in result]


async def create_department(settings: AppSettings, *, org_id: str, name: str, parent_id: str | None = None) -> Department:
    department = DepartmentModel(
        id=uuid4(),
        org_id=UUID(org_id),
        name=name,
        parent_id=UUID(parent_id) if parent_id else None,
    )
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            session.add(department)
        await session.refresh(department)
        return _department_to_dataclass(department)


async def list_knowledge_bases(settings: AppSettings, user: Any) -> list[KnowledgeBase]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        result = await session.scalars(
            select(KnowledgeBaseModel)
            .where(
                KnowledgeBaseModel.org_id == UUID(user.org_id),
                KnowledgeBaseModel.status != "deleted",
            )
            .order_by(KnowledgeBaseModel.updated_at.desc())
        )
        return [
            _kb_to_dataclass(item)
            for item in result
            if await _can_access_kb_model(session, item, user, write=False)
        ]


async def create_knowledge_base(
    settings: AppSettings,
    *,
    user: Any,
    name: str,
    description: str | None,
    visibility: str,
    department_ids: list[str],
) -> KnowledgeBase:
    kb = KnowledgeBaseModel(
        id=uuid4(),
        org_id=UUID(user.org_id),
        owner_user_id=UUID(user.id),
        name=name,
        description=description,
        visibility=visibility,
        department_ids=department_ids,
        status="active",
    )
    member = KnowledgeBaseMemberModel(id=uuid4(), knowledge_base_id=kb.id, user_id=UUID(user.id), role="owner")
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            session.add(kb)
            session.add(member)
        await session.refresh(kb)
        return _kb_to_dataclass(kb)


async def get_knowledge_base(settings: AppSettings, kb_id: str) -> KnowledgeBase | None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        kb = await session.get(KnowledgeBaseModel, UUID(kb_id))
        return _kb_to_dataclass(kb) if kb else None


async def update_knowledge_base(
    settings: AppSettings,
    *,
    kb_id: str,
    user: Any,
    name: str,
    description: str | None,
    visibility: str,
    department_ids: list[str],
) -> KnowledgeBase:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            kb = await session.get(KnowledgeBaseModel, UUID(kb_id))
            if kb is None or kb.org_id != UUID(user.org_id) or kb.status == "deleted":
                from fastapi import HTTPException

                raise HTTPException(status_code=404, detail="Knowledge base not found")
            if not await _can_access_kb_model(session, kb, user, write=True):
                from fastapi import HTTPException

                raise HTTPException(status_code=403, detail="Knowledge base access denied")
            kb.name = name
            kb.description = description
            kb.visibility = visibility
            kb.department_ids = department_ids
        await session.refresh(kb)
        return _kb_to_dataclass(kb)


async def soft_delete_knowledge_base(settings: AppSettings, *, kb_id: str, user: Any) -> KnowledgeBase:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            kb = await session.get(KnowledgeBaseModel, UUID(kb_id))
            if kb is None or kb.org_id != UUID(user.org_id) or kb.status == "deleted":
                from fastapi import HTTPException

                raise HTTPException(status_code=404, detail="Knowledge base not found")
            if not await _can_access_kb_model(session, kb, user, write=True):
                from fastapi import HTTPException

                raise HTTPException(status_code=403, detail="Knowledge base access denied")
            kb.status = "deleted"
            kb.deleted_at = datetime.utcnow()
            await session.execute(
                update(RagFileModel)
                .where(RagFileModel.knowledge_base_id == kb.id, RagFileModel.status != "deleted")
                .values(status="deleted", deleted_at=func.now())
            )
            await session.execute(
                update(ApiKeyModel)
                .where(ApiKeyModel.knowledge_base_id == kb.id)
                .values(is_active=False)
            )
        await session.refresh(kb)
        return _kb_to_dataclass(kb)


async def get_knowledge_base_stats(settings: AppSettings, knowledge_base_id: str) -> KnowledgeBaseStats:
    kb_uuid = UUID(knowledge_base_id)
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        file_count = await session.scalar(
            select(func.count())
            .select_from(RagFileModel)
            .where(RagFileModel.knowledge_base_id == kb_uuid, RagFileModel.status != "deleted")
        )
        completed_file_count = await session.scalar(
            select(func.count())
            .select_from(RagFileModel)
            .where(RagFileModel.knowledge_base_id == kb_uuid, RagFileModel.status == "completed")
        )
        failed_job_count = await session.scalar(
            select(func.count())
            .select_from(IngestJobModel)
            .where(IngestJobModel.knowledge_base_id == kb_uuid, IngestJobModel.status == "failed")
        )
        return KnowledgeBaseStats(
            file_count=int(file_count or 0),
            completed_file_count=int(completed_file_count or 0),
            failed_job_count=int(failed_job_count or 0),
        )


async def list_knowledge_base_members(settings: AppSettings, kb_id: str) -> list[KnowledgeBaseMember]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        rows = await session.execute(
            select(KnowledgeBaseMemberModel, UserModel)
            .join(UserModel, KnowledgeBaseMemberModel.user_id == UserModel.id)
            .where(KnowledgeBaseMemberModel.knowledge_base_id == UUID(kb_id))
            .order_by(KnowledgeBaseMemberModel.created_at.asc())
        )
        return [_member_to_dataclass(member, user) for member, user in rows]


async def is_knowledge_base_owner(settings: AppSettings, *, kb_id: str, user_id: str) -> bool:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        kb = await session.get(KnowledgeBaseModel, UUID(kb_id))
        if kb is None:
            return False
        if str(kb.owner_user_id) == user_id:
            return True
        member = await session.scalar(
            select(KnowledgeBaseMemberModel).where(
                KnowledgeBaseMemberModel.knowledge_base_id == UUID(kb_id),
                KnowledgeBaseMemberModel.user_id == UUID(user_id),
                KnowledgeBaseMemberModel.role == "owner",
            )
        )
        return member is not None


async def upsert_knowledge_base_member(
    settings: AppSettings,
    *,
    kb_id: str,
    user_id: str,
    role: str,
) -> KnowledgeBaseMember:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        kb = await session.get(KnowledgeBaseModel, UUID(kb_id))
        user = await session.get(UserModel, UUID(user_id))
        if kb is None or user is None or user.org_id != kb.org_id or not user.is_active:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Knowledge base member user not found")
    stmt = insert(KnowledgeBaseMemberModel).values(
        id=uuid4(),
        knowledge_base_id=UUID(kb_id),
        user_id=UUID(user_id),
        role=role,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[
            KnowledgeBaseMemberModel.knowledge_base_id,
            KnowledgeBaseMemberModel.user_id,
        ],
        set_={"role": stmt.excluded.role},
    )
    async with session_maker() as session:
        async with session.begin():
            await session.execute(stmt)
        row = await session.execute(
            select(KnowledgeBaseMemberModel, UserModel)
            .join(UserModel, KnowledgeBaseMemberModel.user_id == UserModel.id)
            .where(
                KnowledgeBaseMemberModel.knowledge_base_id == UUID(kb_id),
                KnowledgeBaseMemberModel.user_id == UUID(user_id),
            )
        )
        member, user = row.one()
        return _member_to_dataclass(member, user)


async def remove_knowledge_base_member(settings: AppSettings, *, kb_id: str, user_id: str) -> None:
    from sqlalchemy import delete

    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            kb = await session.get(KnowledgeBaseModel, UUID(kb_id))
            if kb is not None and str(kb.owner_user_id) == user_id:
                from fastapi import HTTPException

                raise HTTPException(status_code=400, detail="Knowledge base owner cannot be removed")
            await session.execute(
                delete(KnowledgeBaseMemberModel).where(
                    KnowledgeBaseMemberModel.knowledge_base_id == UUID(kb_id),
                    KnowledgeBaseMemberModel.user_id == UUID(user_id),
                )
            )


async def require_knowledge_base_access(settings: AppSettings, kb_id: str, user: Any, write: bool = False) -> KnowledgeBase:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        kb = await session.get(KnowledgeBaseModel, UUID(kb_id))
        if kb is None or kb.org_id != UUID(user.org_id) or kb.status == "deleted":
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Knowledge base not found")
        if not await _can_access_kb_model(session, kb, user, write=write):
            from fastapi import HTTPException

            raise HTTPException(status_code=403, detail="Knowledge base access denied")
        return _kb_to_dataclass(kb)


async def add_audit_log(
    settings: AppSettings,
    *,
    org_id: str,
    actor_user_id: str | None,
    action: str,
    target_type: str | None = None,
    target_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    actor_department_id: str | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
    result: str = "success",
    latency_ms: int | None = None,
    error_message: str | None = None,
) -> None:
    row = AuditLogModel(
        id=uuid4(),
        org_id=UUID(org_id),
        actor_user_id=UUID(actor_user_id) if actor_user_id else None,
        actor_department_id=UUID(actor_department_id) if actor_department_id else None,
        action=action,
        target_type=target_type,
        target_id=target_id,
        ip_address=ip_address,
        user_agent=user_agent,
        request_id=request_id,
        result=result,
        latency_ms=latency_ms,
        error_message=error_message,
        metadata_=metadata or {},
    )
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            session.add(row)


async def list_audit_logs(settings: AppSettings, org_id: str, limit: int = 100) -> list[dict[str, Any]]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        result = await session.scalars(
            select(AuditLogModel)
            .where(AuditLogModel.org_id == UUID(org_id))
            .order_by(AuditLogModel.created_at.desc())
            .limit(limit)
        )
        return [
            {
                "id": str(item.id),
                "actor_user_id": str(item.actor_user_id) if item.actor_user_id else None,
                "actor_department_id": str(item.actor_department_id) if item.actor_department_id else None,
                "action": item.action,
                "target_type": item.target_type,
                "target_id": item.target_id,
                "ip_address": item.ip_address,
                "user_agent": item.user_agent,
                "request_id": item.request_id,
                "result": item.result,
                "latency_ms": item.latency_ms,
                "error_message": item.error_message,
                "metadata": dict(item.metadata_ or {}),
                "created_at": item.created_at.isoformat(),
            }
            for item in result
        ]


async def add_chat_log(
    settings: AppSettings,
    *,
    org_id: str,
    knowledge_base_id: str,
    user_id: str,
    conversation_id: str,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    latency_ms: int | None,
) -> None:
    row = ChatLogModel(
        id=uuid4(),
        org_id=UUID(org_id),
        knowledge_base_id=UUID(knowledge_base_id),
        user_id=UUID(user_id),
        conversation_id=UUID(conversation_id),
        question=question,
        answer=answer,
        sources=sources,
        latency_ms=latency_ms,
    )
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            session.add(row)


async def _can_access_kb_model(session: Any, kb: KnowledgeBaseModel, user: Any, write: bool) -> bool:
    if user.role == "admin":
        return True
    if str(kb.owner_user_id) == user.id:
        return True
    member = await session.scalar(
        select(KnowledgeBaseMemberModel).where(
            KnowledgeBaseMemberModel.knowledge_base_id == kb.id,
            KnowledgeBaseMemberModel.user_id == UUID(user.id),
        )
    )
    if member is None:
        if write:
            return False
        if kb.visibility == "org":
            return True
        if kb.visibility == "department" and user.department_id and user.department_id in list(kb.department_ids or []):
            return True
        return False
    if write:
        return member.role in {"owner", "editor"}
    return member.role in {"owner", "editor", "viewer"}


def _user_to_dataclass(model: UserModel) -> EnterpriseUser:
    return EnterpriseUser(
        id=str(model.id),
        org_id=str(model.org_id),
        department_id=str(model.department_id) if model.department_id else None,
        email=model.email,
        display_name=model.display_name,
        password_hash=model.password_hash,
        role=model.role,
        is_active=bool(model.is_active),
        last_login_at=model.last_login_at,
        created_at=model.created_at,
    )


def _department_to_dataclass(model: DepartmentModel) -> Department:
    return Department(
        id=str(model.id),
        org_id=str(model.org_id),
        name=model.name,
        parent_id=str(model.parent_id) if model.parent_id else None,
        created_at=model.created_at,
    )


def _kb_to_dataclass(model: KnowledgeBaseModel) -> KnowledgeBase:
    return KnowledgeBase(
        id=str(model.id),
        org_id=str(model.org_id),
        owner_user_id=str(model.owner_user_id),
        name=model.name,
        description=model.description,
        visibility=model.visibility,
        department_ids=list(model.department_ids or []),
        status=model.status,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _member_to_dataclass(member: KnowledgeBaseMemberModel, user: UserModel) -> KnowledgeBaseMember:
    return KnowledgeBaseMember(
        id=str(member.id),
        knowledge_base_id=str(member.knowledge_base_id),
        user_id=str(member.user_id),
        role=member.role,
        email=user.email,
        display_name=user.display_name,
        department_id=str(user.department_id) if user.department_id else None,
        created_at=member.created_at,
    )
