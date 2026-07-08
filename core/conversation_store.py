from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
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
    user_id: str
    title: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ChatMessage:
    id: str
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
    user_id: str,
    title: str | None = None,
) -> Conversation:
    stmt = insert(RagConversationModel).values(
        id=UUID(conversation_id),
        user_id=user_id,
        title=title,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[RagConversationModel.id],
        set_={"updated_at": func.now()},
        where=RagConversationModel.user_id == user_id,
    ).returning(RagConversationModel)
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            result = await session.scalar(stmt)
            if result is None:
                raise ValueError("Conversation belongs to another user")
            return _model_to_conversation(result)


async def add_message(
    settings: AppSettings,
    *,
    message_id: str,
    conversation_id: str,
    user_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> ChatMessage:
    message = RagMessageModel(
        id=UUID(message_id),
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


async def get_recent_messages(
    settings: AppSettings,
    *,
    conversation_id: str,
    user_id: str,
    limit: int,
) -> list[ChatMessage]:
    if limit <= 0:
        return []
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        recent = (
            select(RagMessageModel)
            .where(
                RagMessageModel.user_id == user_id,
                RagMessageModel.conversation_id == UUID(conversation_id),
            )
            .order_by(RagMessageModel.created_at.desc())
            .limit(limit)
            .subquery()
        )
        result = await session.execute(select(recent).order_by(recent.c.created_at.asc()))
        return [_row_to_message(row._mapping) for row in result]


async def list_conversations(
    settings: AppSettings,
    user_id: str,
    limit: int,
    offset: int,
) -> list[Conversation]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        result = await session.scalars(
            select(RagConversationModel)
            .where(RagConversationModel.user_id == user_id)
            .order_by(RagConversationModel.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return [_model_to_conversation(item) for item in result]


def _model_to_conversation(model: RagConversationModel) -> Conversation:
    return Conversation(
        id=str(model.id),
        user_id=model.user_id,
        title=model.title,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _model_to_message(model: RagMessageModel) -> ChatMessage:
    return ChatMessage(
        id=str(model.id),
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
        conversation_id=str(row["conversation_id"]),
        user_id=row["user_id"],
        role=row["role"],
        content=row["content"],
        metadata=dict(row["metadata"] or {}),
        created_at=row["created_at"],
    )
