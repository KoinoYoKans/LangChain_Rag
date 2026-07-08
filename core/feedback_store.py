from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select

from config.database import FeedbackModel, build_async_session_maker, initialize_orm_tables
from config.settings import AppSettings


@dataclass(frozen=True)
class Feedback:
    id: str
    org_id: str
    knowledge_base_id: str
    user_id: str
    conversation_id: str
    assistant_message_id: str
    rating: str
    reason: str | None
    comment: str | None
    question: str
    answer: str
    sources_snapshot: list[dict[str, Any]]
    created_at: datetime


async def initialize_feedback_tables(settings: AppSettings) -> None:
    await initialize_orm_tables(settings)


async def create_feedback(
    settings: AppSettings,
    *,
    org_id: str,
    knowledge_base_id: str,
    user_id: str,
    conversation_id: str,
    assistant_message_id: str,
    rating: str,
    reason: str | None,
    comment: str | None,
    question: str,
    answer: str,
    sources_snapshot: list[dict[str, Any]],
) -> Feedback:
    row = FeedbackModel(
        id=uuid4(),
        org_id=UUID(org_id),
        knowledge_base_id=UUID(knowledge_base_id),
        user_id=UUID(user_id),
        conversation_id=UUID(conversation_id),
        assistant_message_id=UUID(assistant_message_id),
        rating=rating,
        reason=reason,
        comment=comment,
        question=question,
        answer=answer,
        sources_snapshot=sources_snapshot,
    )
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            session.add(row)
        await session.refresh(row)
        return _model_to_feedback(row)


async def list_feedback(
    settings: AppSettings,
    *,
    org_id: str,
    knowledge_base_id: str | None = None,
    knowledge_base_ids: list[str] | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Feedback]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        conditions = [FeedbackModel.org_id == UUID(org_id)]
        if knowledge_base_id:
            conditions.append(FeedbackModel.knowledge_base_id == UUID(knowledge_base_id))
        elif knowledge_base_ids is not None:
            if not knowledge_base_ids:
                return []
            conditions.append(FeedbackModel.knowledge_base_id.in_([UUID(item) for item in knowledge_base_ids]))
        rows = await session.scalars(
            select(FeedbackModel)
            .where(*conditions)
            .order_by(FeedbackModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return [_model_to_feedback(row) for row in rows]


def _model_to_feedback(row: FeedbackModel) -> Feedback:
    return Feedback(
        id=str(row.id),
        org_id=str(row.org_id),
        knowledge_base_id=str(row.knowledge_base_id),
        user_id=str(row.user_id),
        conversation_id=str(row.conversation_id),
        assistant_message_id=str(row.assistant_message_id),
        rating=row.rating,
        reason=row.reason,
        comment=row.comment,
        question=row.question,
        answer=row.answer,
        sources_snapshot=list(row.sources_snapshot or []),
        created_at=row.created_at,
    )
