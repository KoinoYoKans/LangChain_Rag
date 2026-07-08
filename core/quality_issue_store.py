from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from config.database import (
    ChatLogModel,
    FeedbackModel,
    QualityIssueModel,
    UserModel,
    build_async_session_maker,
)
from config.settings import AppSettings


QUALITY_ISSUE_TYPES = {
    "wrong_answer",
    "missing_source",
    "source_mismatch",
    "outdated_content",
    "permission_risk",
    "other",
}
QUALITY_ISSUE_PRIORITIES = {"low", "medium", "high", "urgent"}
QUALITY_ISSUE_STATUSES = {"open", "in_progress", "resolved", "ignored"}


@dataclass(frozen=True)
class QualityIssue:
    id: str
    org_id: str
    knowledge_base_id: str
    chat_log_id: str | None
    conversation_id: str | None
    assistant_message_id: str
    user_id: str | None
    question: str
    answer_snapshot: str
    sources_snapshot: list[dict[str, Any]]
    issue_type: str
    priority: str
    status: str
    assignee_user_id: str | None
    resolution_note: str | None
    created_by_user_id: str
    resolved_at: datetime | None
    feedback_id: str | None
    feedback_rating: str | None
    feedback_reason: str | None
    feedback_comment: str | None
    created_at: datetime
    updated_at: datetime


async def create_quality_issue_from_chat(
    settings: AppSettings,
    *,
    org_id: str,
    knowledge_base_id: str,
    created_by_user_id: str,
    assistant_message_id: str | None = None,
    chat_log_id: str | None = None,
    issue_type: str,
    priority: str,
    assignee_user_id: str | None = None,
    resolution_note: str | None = None,
) -> tuple[QualityIssue, bool]:
    _validate_issue_fields(issue_type=issue_type, priority=priority, status="open")
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        chat_log = await _find_chat_log(
            session,
            org_id=org_id,
            knowledge_base_id=knowledge_base_id,
            assistant_message_id=assistant_message_id,
            chat_log_id=chat_log_id,
        )
        if chat_log is None or chat_log.assistant_message_id is None:
            raise ValueError("Chat operation not found")
        existing = await session.scalar(
            select(QualityIssueModel).where(
                QualityIssueModel.knowledge_base_id == UUID(knowledge_base_id),
                QualityIssueModel.assistant_message_id == chat_log.assistant_message_id,
            )
        )
        if existing is not None:
            return _model_to_quality_issue(existing), False
        if assignee_user_id:
            await _ensure_active_org_user(session, org_id=org_id, user_id=assignee_user_id)
        feedback = await session.scalar(
            select(FeedbackModel)
            .where(
                FeedbackModel.org_id == UUID(org_id),
                FeedbackModel.knowledge_base_id == UUID(knowledge_base_id),
                FeedbackModel.assistant_message_id == chat_log.assistant_message_id,
            )
            .order_by(FeedbackModel.created_at.desc())
            .limit(1)
        )
        issue = QualityIssueModel(
            id=uuid4(),
            org_id=UUID(org_id),
            knowledge_base_id=UUID(knowledge_base_id),
            chat_log_id=chat_log.id,
            conversation_id=chat_log.conversation_id,
            assistant_message_id=chat_log.assistant_message_id,
            user_id=chat_log.user_id,
            question=chat_log.question,
            answer_snapshot=chat_log.answer,
            sources_snapshot=list(chat_log.sources or []),
            issue_type=issue_type,
            priority=priority,
            status="open",
            assignee_user_id=UUID(assignee_user_id) if assignee_user_id else None,
            resolution_note=resolution_note,
            created_by_user_id=UUID(created_by_user_id),
            feedback_id=feedback.id if feedback else None,
            feedback_rating=feedback.rating if feedback else None,
            feedback_reason=feedback.reason if feedback else None,
            feedback_comment=feedback.comment if feedback else None,
        )
        session.add(issue)
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            existing = await session.scalar(
                select(QualityIssueModel).where(
                    QualityIssueModel.knowledge_base_id == UUID(knowledge_base_id),
                    QualityIssueModel.assistant_message_id == chat_log.assistant_message_id,
                )
            )
            if existing is not None:
                return _model_to_quality_issue(existing), False
            raise
        await session.refresh(issue)
        return _model_to_quality_issue(issue), True


async def list_quality_issues(
    settings: AppSettings,
    *,
    org_id: str,
    knowledge_base_ids: list[str],
    knowledge_base_id: str | None = None,
    status: str | None = None,
    assignee_user_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[QualityIssue]:
    if not knowledge_base_ids:
        return []
    selected_ids = {UUID(item) for item in knowledge_base_ids}
    if knowledge_base_id:
        requested_id = UUID(knowledge_base_id)
        if requested_id not in selected_ids:
            return []
        selected_ids = {requested_id}
    conditions = [
        QualityIssueModel.org_id == UUID(org_id),
        QualityIssueModel.knowledge_base_id.in_(list(selected_ids)),
    ]
    if status:
        conditions.append(QualityIssueModel.status == status)
    if assignee_user_id:
        conditions.append(QualityIssueModel.assignee_user_id == UUID(assignee_user_id))
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        rows = await session.scalars(
            select(QualityIssueModel)
            .where(*conditions)
            .order_by(QualityIssueModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return [_model_to_quality_issue(row) for row in rows]


async def get_quality_issue(settings: AppSettings, issue_id: str) -> QualityIssue | None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        issue = await session.get(QualityIssueModel, UUID(issue_id))
        return _model_to_quality_issue(issue) if issue else None


async def update_quality_issue(
    settings: AppSettings,
    *,
    issue_id: str,
    org_id: str,
    values: dict[str, Any],
) -> tuple[QualityIssue, dict[str, Any]]:
    if not values:
        raise ValueError("No quality issue fields to update")
    status = values.get("status")
    priority = values.get("priority")
    if status is not None and status not in QUALITY_ISSUE_STATUSES:
        raise ValueError("Unsupported quality issue status")
    if priority is not None and priority not in QUALITY_ISSUE_PRIORITIES:
        raise ValueError("Unsupported quality issue priority")
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            issue = await session.get(QualityIssueModel, UUID(issue_id))
            if issue is None or issue.org_id != UUID(org_id):
                raise ValueError("Quality issue not found")
            if values.get("assignee_user_id"):
                await _ensure_active_org_user(session, org_id=org_id, user_id=values["assignee_user_id"])
            before = {
                "status": issue.status,
                "priority": issue.priority,
                "assignee_user_id": str(issue.assignee_user_id) if issue.assignee_user_id else None,
                "resolution_note": issue.resolution_note,
            }
            if "status" in values:
                issue.status = values["status"]
                issue.resolved_at = datetime.now(timezone.utc) if values["status"] in {"resolved", "ignored"} else None
            if "priority" in values:
                issue.priority = values["priority"]
            if "assignee_user_id" in values:
                issue.assignee_user_id = UUID(values["assignee_user_id"]) if values["assignee_user_id"] else None
            if "resolution_note" in values:
                issue.resolution_note = values["resolution_note"]
        await session.refresh(issue)
        return _model_to_quality_issue(issue), before


async def list_quality_issue_links(
    settings: AppSettings,
    *,
    org_id: str,
    knowledge_base_ids: list[str],
    assistant_message_ids: list[str],
) -> dict[str, QualityIssue]:
    if not knowledge_base_ids or not assistant_message_ids:
        return {}
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        rows = await session.scalars(
            select(QualityIssueModel).where(
                QualityIssueModel.org_id == UUID(org_id),
                QualityIssueModel.knowledge_base_id.in_([UUID(item) for item in knowledge_base_ids]),
                QualityIssueModel.assistant_message_id.in_([UUID(item) for item in assistant_message_ids]),
            )
        )
        return {str(row.assistant_message_id): _model_to_quality_issue(row) for row in rows}


async def _find_chat_log(
    session: Any,
    *,
    org_id: str,
    knowledge_base_id: str,
    assistant_message_id: str | None,
    chat_log_id: str | None,
) -> ChatLogModel | None:
    conditions = [
        ChatLogModel.org_id == UUID(org_id),
        ChatLogModel.knowledge_base_id == UUID(knowledge_base_id),
    ]
    if chat_log_id:
        conditions.append(ChatLogModel.id == UUID(chat_log_id))
    elif assistant_message_id:
        conditions.append(ChatLogModel.assistant_message_id == UUID(assistant_message_id))
    else:
        raise ValueError("assistant_message_id or chat_log_id is required")
    return await session.scalar(select(ChatLogModel).where(*conditions).limit(1))


async def _ensure_active_org_user(session: Any, *, org_id: str, user_id: str) -> None:
    user = await session.get(UserModel, UUID(user_id))
    if user is None or user.org_id != UUID(org_id) or not user.is_active:
        raise ValueError("Assignee user not found")


def _validate_issue_fields(*, issue_type: str, priority: str, status: str) -> None:
    if issue_type not in QUALITY_ISSUE_TYPES:
        raise ValueError("Unsupported quality issue type")
    if priority not in QUALITY_ISSUE_PRIORITIES:
        raise ValueError("Unsupported quality issue priority")
    if status not in QUALITY_ISSUE_STATUSES:
        raise ValueError("Unsupported quality issue status")


def _model_to_quality_issue(row: QualityIssueModel) -> QualityIssue:
    return QualityIssue(
        id=str(row.id),
        org_id=str(row.org_id),
        knowledge_base_id=str(row.knowledge_base_id),
        chat_log_id=str(row.chat_log_id) if row.chat_log_id else None,
        conversation_id=str(row.conversation_id) if row.conversation_id else None,
        assistant_message_id=str(row.assistant_message_id),
        user_id=str(row.user_id) if row.user_id else None,
        question=row.question,
        answer_snapshot=row.answer_snapshot,
        sources_snapshot=list(row.sources_snapshot or []),
        issue_type=row.issue_type,
        priority=row.priority,
        status=row.status,
        assignee_user_id=str(row.assignee_user_id) if row.assignee_user_id else None,
        resolution_note=row.resolution_note,
        created_by_user_id=str(row.created_by_user_id),
        resolved_at=row.resolved_at,
        feedback_id=str(row.feedback_id) if row.feedback_id else None,
        feedback_rating=row.feedback_rating,
        feedback_reason=row.feedback_reason,
        feedback_comment=row.feedback_comment,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
