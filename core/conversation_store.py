from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert

from config.database import (
    RagConversationModel,
    RagMessageModel,
    build_async_session_maker,
    initialize_orm_tables,
)
from config.settings import AppSettings


@dataclass(frozen=True)
class Conversation:
    id: str
    org_id: str | None
    knowledge_base_id: str | None
    user_id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ChatMessage:
    id: str
    org_id: str | None
    knowledge_base_id: str | None
    conversation_id: str
    user_id: str
    role: str
    content: str
    metadata: dict[str, Any]
    created_at: datetime


async def initialize_conversation_tables(settings: AppSettings) -> None:
    await initialize_orm_tables(settings)


async def ensure_conversation(
    settings: AppSettings,
    *,
    conversation_id: str,
    org_id: str | None = None,
    knowledge_base_id: str | None = None,
    user_id: str,
    title: str | None = None,
) -> Conversation:
    stmt = insert(RagConversationModel).values(
        id=UUID(conversation_id),
        org_id=UUID(org_id) if org_id else None,
        knowledge_base_id=UUID(knowledge_base_id) if knowledge_base_id else None,
        user_id=user_id,
        title=title,
    )
    same_scope = RagConversationModel.user_id == user_id
    if knowledge_base_id:
        same_scope = same_scope & (RagConversationModel.knowledge_base_id == UUID(knowledge_base_id))
    else:
        same_scope = same_scope & RagConversationModel.knowledge_base_id.is_(None)
    stmt = stmt.on_conflict_do_update(
        index_elements=[RagConversationModel.id],
        set_={"updated_at": func.now()},
        where=same_scope,
    ).returning(RagConversationModel)
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            result = await session.scalar(stmt)
            if result is None:
                raise ValueError("Conversation belongs to another user or knowledge base")
            return _model_to_conversation(result)


async def add_message(
    settings: AppSettings,
    *,
    message_id: str,
    conversation_id: str,
    org_id: str | None = None,
    knowledge_base_id: str | None = None,
    user_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> ChatMessage:
    message = RagMessageModel(
        id=UUID(message_id),
        org_id=UUID(org_id) if org_id else None,
        knowledge_base_id=UUID(knowledge_base_id) if knowledge_base_id else None,
        conversation_id=UUID(conversation_id),
        user_id=user_id,
        role=role,
        content=content,
        metadata_=metadata or {},
    )
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            session.add(message)
        await session.refresh(message)
        return _model_to_message(message)


async def get_conversation(
    settings: AppSettings,
    *,
    conversation_id: str,
    user_id: str,
    knowledge_base_id: str,
) -> Conversation | None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        result = await session.scalar(
            select(RagConversationModel).where(
                RagConversationModel.id == UUID(conversation_id),
                RagConversationModel.user_id == user_id,
                RagConversationModel.knowledge_base_id == UUID(knowledge_base_id),
            )
        )
        return _model_to_conversation(result) if result else None


async def get_recent_messages(
    settings: AppSettings,
    *,
    conversation_id: str,
    user_id: str,
    limit: int,
    knowledge_base_id: str | None = None,
) -> list[ChatMessage]:
    if limit <= 0:
        return []
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        conditions = [
            RagMessageModel.user_id == user_id,
            RagMessageModel.conversation_id == UUID(conversation_id),
        ]
        if knowledge_base_id:
            conditions.append(RagMessageModel.knowledge_base_id == UUID(knowledge_base_id))
        recent = (
            select(RagMessageModel)
            .where(*conditions)
            .order_by(RagMessageModel.created_at.desc())
            .limit(limit)
            .subquery()
        )
        result = await session.execute(select(recent).order_by(recent.c.created_at.asc()))
        return [_row_to_message(row._mapping) for row in result]


async def update_conversation_title(
    settings: AppSettings,
    *,
    conversation_id: str,
    user_id: str,
    knowledge_base_id: str,
    title: str,
) -> Conversation | None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            result = await session.scalar(
                update(RagConversationModel)
                .where(
                    RagConversationModel.id == UUID(conversation_id),
                    RagConversationModel.user_id == user_id,
                    RagConversationModel.knowledge_base_id == UUID(knowledge_base_id),
                )
                .values(title=title, updated_at=func.now())
                .returning(RagConversationModel)
            )
            return _model_to_conversation(result) if result else None


async def delete_conversation(
    settings: AppSettings,
    *,
    conversation_id: str,
    user_id: str,
    knowledge_base_id: str,
) -> bool:
    conversation_uuid = UUID(conversation_id)
    kb_uuid = UUID(knowledge_base_id)
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            exists = await session.scalar(
                select(RagConversationModel.id).where(
                    RagConversationModel.id == conversation_uuid,
                    RagConversationModel.user_id == user_id,
                    RagConversationModel.knowledge_base_id == kb_uuid,
                )
            )
            if exists is None:
                return False
            await session.execute(
                delete(RagMessageModel).where(
                    RagMessageModel.user_id == user_id,
                    RagMessageModel.conversation_id == conversation_uuid,
                    RagMessageModel.knowledge_base_id == kb_uuid,
                )
            )
            await session.execute(
                delete(RagConversationModel).where(
                    RagConversationModel.id == conversation_uuid,
                    RagConversationModel.user_id == user_id,
                    RagConversationModel.knowledge_base_id == kb_uuid,
                )
            )
            return True


async def list_conversation_messages(
    settings: AppSettings,
    *,
    conversation_id: str,
    user_id: str,
    knowledge_base_id: str,
    limit: int = 200,
) -> list[ChatMessage]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        result = await session.scalars(
            select(RagMessageModel)
            .where(
                RagMessageModel.user_id == user_id,
                RagMessageModel.conversation_id == UUID(conversation_id),
                RagMessageModel.knowledge_base_id == UUID(knowledge_base_id),
            )
            .order_by(RagMessageModel.created_at.asc())
            .limit(limit)
        )
        return [_model_to_message(item) for item in result]


async def list_conversations(
    settings: AppSettings,
    user_id: str,
    limit: int,
    offset: int,
    knowledge_base_id: str | None = None,
) -> list[Conversation]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        conditions = [RagConversationModel.user_id == user_id]
        if knowledge_base_id:
            conditions.append(RagConversationModel.knowledge_base_id == UUID(knowledge_base_id))
        result = await session.scalars(
            select(RagConversationModel)
            .where(*conditions)
            .order_by(RagConversationModel.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return [_model_to_conversation(item) for item in result]


def _model_to_conversation(model: RagConversationModel) -> Conversation:
    return Conversation(
        id=str(model.id),
        org_id=str(model.org_id) if model.org_id else None,
        knowledge_base_id=str(model.knowledge_base_id) if model.knowledge_base_id else None,
        user_id=model.user_id,
        title=model.title,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _model_to_message(model: RagMessageModel) -> ChatMessage:
    return ChatMessage(
        id=str(model.id),
        org_id=str(model.org_id) if model.org_id else None,
        knowledge_base_id=str(model.knowledge_base_id) if model.knowledge_base_id else None,
        conversation_id=str(model.conversation_id),
        user_id=model.user_id,
        role=model.role,
        content=model.content,
        metadata=dict(model.metadata_ or {}),
        created_at=model.created_at,
    )


def _row_to_message(row: Any) -> ChatMessage:
    return ChatMessage(
        id=str(row["id"]),
        org_id=str(row["org_id"]) if row.get("org_id") else None,
        knowledge_base_id=str(row["knowledge_base_id"]) if row.get("knowledge_base_id") else None,
        conversation_id=str(row["conversation_id"]),
        user_id=row["user_id"],
        role=row["role"],
        content=row["content"],
        metadata=dict(row["metadata"] or {}),
        created_at=row["created_at"],
    )
