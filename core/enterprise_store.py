from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.dialects.postgresql import array, insert

from config.database import (
    ApiKeyModel,
    AuditLogModel,
    ChatLogModel,
    DepartmentModel,
    FeedbackModel,
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
    retrieval_top_k: int | None
    rerank_top_n: int | None
    low_confidence_threshold: float
    low_confidence_max_retries: int
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


@dataclass(frozen=True)
class KnowledgeBaseCapabilities:
    current_user_role: str
    can_read: bool
    can_write: bool
    can_manage_members: bool
    can_manage_settings: bool
    can_manage_api_keys: bool


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
    retrieval_top_k: int | None = None,
    rerank_top_n: int | None = None,
    low_confidence_threshold: float = 0.35,
    low_confidence_max_retries: int = 1,
) -> KnowledgeBase:
    kb = KnowledgeBaseModel(
        id=uuid4(),
        org_id=UUID(user.org_id),
        owner_user_id=UUID(user.id),
        name=name,
        description=description,
        visibility=visibility,
        department_ids=department_ids,
        retrieval_top_k=retrieval_top_k,
        rerank_top_n=rerank_top_n,
        low_confidence_threshold=low_confidence_threshold,
        low_confidence_max_retries=low_confidence_max_retries,
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
    retrieval_top_k: int | None = None,
    rerank_top_n: int | None = None,
    low_confidence_threshold: float = 0.35,
    low_confidence_max_retries: int = 1,
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
            kb.retrieval_top_k = retrieval_top_k
            kb.rerank_top_n = rerank_top_n
            kb.low_confidence_threshold = low_confidence_threshold
            kb.low_confidence_max_retries = low_confidence_max_retries
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


async def get_knowledge_base_capabilities(settings: AppSettings, kb_id: str, user: Any) -> KnowledgeBaseCapabilities:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        kb = await session.get(KnowledgeBaseModel, UUID(kb_id))
        if kb is None or kb.org_id != UUID(user.org_id) or kb.status == "deleted":
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Knowledge base not found")
        return await _kb_capabilities(session, kb, user)


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


async def list_audit_logs(
    settings: AppSettings,
    org_id: str,
    limit: int = 100,
    knowledge_base_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        query = select(AuditLogModel).where(AuditLogModel.org_id == UUID(org_id))
        allowed_ids = list(knowledge_base_ids or [])
        if knowledge_base_ids is not None:
            if not allowed_ids:
                return []
            query = query.where(
                or_(
                    AuditLogModel.target_id.in_(allowed_ids),
                    AuditLogModel.metadata_["knowledge_base_id"].as_string().in_(allowed_ids),
                    AuditLogModel.metadata_["bound_knowledge_base_id"].as_string().in_(allowed_ids),
                    AuditLogModel.metadata_["knowledge_base_ids"].op("?|")(array(allowed_ids)),
                )
            )
        result = await session.scalars(
            query
            .order_by(AuditLogModel.created_at.desc())
            .limit(limit)
        )
        items: list[dict[str, Any]] = []
        for item in result:
            metadata = dict(item.metadata_ or {})
            items.append(
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
                    "metadata": metadata,
                    "created_at": item.created_at.isoformat(),
                }
            )
            if len(items) >= limit:
                break
        return items


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
    api_key_id: str | None = None,
    assistant_message_id: str | None = None,
    request_id: str | None = None,
    model_name: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    citation_count: int = 0,
    citation_coverage: float | None = None,
    answer_status: str | None = None,
    confidence: str | None = None,
    confidence_score: float | None = None,
    retry_count: int = 0,
    retry_trace: list[dict[str, Any]] | None = None,
    auto_retry_triggered: bool = False,
    final_low_confidence: bool = False,
) -> None:
    row = ChatLogModel(
        id=uuid4(),
        org_id=UUID(org_id),
        knowledge_base_id=UUID(knowledge_base_id),
        user_id=UUID(user_id),
        api_key_id=UUID(api_key_id) if api_key_id else None,
        conversation_id=UUID(conversation_id),
        assistant_message_id=UUID(assistant_message_id) if assistant_message_id else None,
        request_id=request_id,
        question=question,
        answer=answer,
        sources=sources,
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        source_count=len(sources),
        citation_count=citation_count,
        citation_coverage=citation_coverage,
        answer_status=answer_status,
        confidence=confidence,
        confidence_score=confidence_score,
        retry_count=retry_count,
        retry_trace=retry_trace or [],
        auto_retry_triggered=auto_retry_triggered,
        final_low_confidence=final_low_confidence,
        latency_ms=latency_ms,
    )
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            session.add(row)


async def list_chat_operations(
    settings: AppSettings,
    *,
    org_id: str,
    knowledge_base_ids: list[str],
    limit: int = 100,
    knowledge_base_id: str | None = None,
    feedback_rating: str | None = None,
    answer_status: str | None = None,
    low_confidence: bool = False,
    no_citations: bool = False,
) -> list[dict[str, Any]]:
    if not knowledge_base_ids:
        return []
    allowed_ids = {UUID(item) for item in knowledge_base_ids}
    selected_ids = allowed_ids
    if knowledge_base_id:
        requested_id = UUID(knowledge_base_id)
        if requested_id not in allowed_ids:
            return []
        selected_ids = {requested_id}
    conditions = [
        ChatLogModel.org_id == UUID(org_id),
        ChatLogModel.knowledge_base_id.in_(list(selected_ids)),
    ]
    if answer_status:
        conditions.append(ChatLogModel.answer_status == answer_status)
    if low_confidence:
        conditions.append(or_(ChatLogModel.confidence == "low", ChatLogModel.final_low_confidence.is_(True)))
    if no_citations:
        conditions.append(ChatLogModel.citation_count == 0)
    join_condition = and_(
        FeedbackModel.org_id == ChatLogModel.org_id,
        FeedbackModel.knowledge_base_id == ChatLogModel.knowledge_base_id,
        FeedbackModel.conversation_id == ChatLogModel.conversation_id,
        FeedbackModel.user_id == ChatLogModel.user_id,
        FeedbackModel.assistant_message_id == ChatLogModel.assistant_message_id,
    )
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        rows = await session.execute(
            select(ChatLogModel, FeedbackModel)
            .outerjoin(FeedbackModel, join_condition)
            .where(*conditions)
            .order_by(ChatLogModel.created_at.desc())
            .limit(limit if feedback_rating is None else min(max(limit * 5, limit), 1000))
        )
        items = []
        for chat_log, feedback in rows:
            if feedback_rating:
                if feedback_rating == "unrated" and feedback is not None:
                    continue
                if feedback_rating in {"up", "down"} and (feedback is None or feedback.rating != feedback_rating):
                    continue
            items.append(chat_operation_to_dict(chat_log, feedback))
            if len(items) >= limit:
                break
        return items


def chat_operation_to_dict(chat_log: ChatLogModel, feedback: FeedbackModel | None = None) -> dict[str, Any]:
    return {
        "id": str(chat_log.id),
        "org_id": str(chat_log.org_id),
        "knowledge_base_id": str(chat_log.knowledge_base_id),
        "user_id": str(chat_log.user_id),
        "api_key_id": str(chat_log.api_key_id) if chat_log.api_key_id else None,
        "conversation_id": str(chat_log.conversation_id),
        "assistant_message_id": str(chat_log.assistant_message_id) if chat_log.assistant_message_id else None,
        "request_id": chat_log.request_id,
        "question": chat_log.question,
        "answer": chat_log.answer,
        "sources": list(chat_log.sources or []),
        "source_count": int(chat_log.source_count or 0),
        "citation_count": int(chat_log.citation_count or 0),
        "citation_coverage": float(chat_log.citation_coverage) if chat_log.citation_coverage is not None else None,
        "answer_status": chat_log.answer_status,
        "confidence": chat_log.confidence,
        "confidence_score": float(chat_log.confidence_score) if chat_log.confidence_score is not None else None,
        "retry_count": int(chat_log.retry_count or 0),
        "retry_trace": list(chat_log.retry_trace or []),
        "auto_retry_triggered": bool(chat_log.auto_retry_triggered),
        "final_low_confidence": bool(chat_log.final_low_confidence),
        "model_name": chat_log.model_name,
        "latency_ms": chat_log.latency_ms,
        "prompt_tokens": chat_log.prompt_tokens,
        "completion_tokens": chat_log.completion_tokens,
        "total_tokens": chat_log.total_tokens,
        "feedback_rating": feedback.rating if feedback else None,
        "feedback_reason": feedback.reason if feedback else None,
        "feedback_comment": feedback.comment if feedback else None,
        "created_at": chat_log.created_at.isoformat(),
    }


async def _can_access_kb_model(session: Any, kb: KnowledgeBaseModel, user: Any, write: bool) -> bool:
    capabilities = await _kb_capabilities(session, kb, user)
    return capabilities.can_write if write else capabilities.can_read


async def _kb_capabilities(session: Any, kb: KnowledgeBaseModel, user: Any) -> KnowledgeBaseCapabilities:
    if user.role == "admin":
        return KnowledgeBaseCapabilities(
            current_user_role="admin",
            can_read=True,
            can_write=True,
            can_manage_members=True,
            can_manage_settings=True,
            can_manage_api_keys=True,
        )
    is_owner = str(kb.owner_user_id) == user.id
    member = await session.scalar(
        select(KnowledgeBaseMemberModel).where(
            KnowledgeBaseMemberModel.knowledge_base_id == kb.id,
            KnowledgeBaseMemberModel.user_id == UUID(user.id),
        )
    )
    member_role = member.role if member is not None else None
    if is_owner or member_role == "owner":
        return KnowledgeBaseCapabilities(
            current_user_role="owner",
            can_read=True,
            can_write=True,
            can_manage_members=True,
            can_manage_settings=True,
            can_manage_api_keys=True,
        )
    if member_role == "editor":
        return KnowledgeBaseCapabilities(
            current_user_role="editor",
            can_read=True,
            can_write=True,
            can_manage_members=False,
            can_manage_settings=False,
            can_manage_api_keys=False,
        )
    if member_role == "viewer":
        return KnowledgeBaseCapabilities(
            current_user_role="viewer",
            can_read=True,
            can_write=False,
            can_manage_members=False,
            can_manage_settings=False,
            can_manage_api_keys=False,
        )
    implicit_read = False
    if member is None:
        if kb.visibility == "org":
            implicit_read = True
        if kb.visibility == "department" and user.department_id and user.department_id in list(kb.department_ids or []):
            implicit_read = True
    return KnowledgeBaseCapabilities(
        current_user_role="implicit_viewer" if implicit_read else "none",
        can_read=implicit_read,
        can_write=False,
        can_manage_members=False,
        can_manage_settings=False,
        can_manage_api_keys=False,
    )


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
        retrieval_top_k=model.retrieval_top_k,
        rerank_top_n=model.rerank_top_n,
        low_confidence_threshold=float(
            model.low_confidence_threshold
            if model.low_confidence_threshold is not None
            else 0.35
        ),
        low_confidence_max_retries=int(model.low_confidence_max_retries or 0),
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
