from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select

from config.database import IngestJobModel, build_async_session_maker
from config.settings import AppSettings


@dataclass(frozen=True)
class IngestJob:
    id: str
    org_id: str
    knowledge_base_id: str
    created_by_user_id: str
    source_type: str
    source_uri: str | None
    filename: str | None
    status: str
    progress: int
    error_message: str | None
    retry_count: int
    payload: dict[str, Any]
    file_id: str | None
    created_at: datetime
    updated_at: datetime


async def create_ingest_job(
    settings: AppSettings,
    *,
    org_id: str,
    knowledge_base_id: str,
    created_by_user_id: str,
    source_type: str,
    source_uri: str | None,
    filename: str | None,
    payload: dict[str, Any],
) -> IngestJob:
    job = IngestJobModel(
        id=uuid4(),
        org_id=UUID(org_id),
        knowledge_base_id=UUID(knowledge_base_id),
        created_by_user_id=UUID(created_by_user_id),
        source_type=source_type,
        source_uri=source_uri,
        filename=filename,
        status="pending",
        progress=0,
        payload=payload,
    )
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            session.add(job)
        await session.refresh(job)
        return _job_to_dataclass(job)


async def get_ingest_job(settings: AppSettings, job_id: str) -> IngestJob | None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        job = await session.get(IngestJobModel, UUID(job_id))
        return _job_to_dataclass(job) if job else None


async def list_ingest_jobs(settings: AppSettings, knowledge_base_id: str, limit: int = 100) -> list[IngestJob]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        result = await session.scalars(
            select(IngestJobModel)
            .where(IngestJobModel.knowledge_base_id == UUID(knowledge_base_id))
            .order_by(IngestJobModel.created_at.desc())
            .limit(limit)
        )
        return [_job_to_dataclass(item) for item in result]


async def list_recoverable_ingest_jobs(settings: AppSettings, limit: int = 50) -> list[IngestJob]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        result = await session.scalars(
            select(IngestJobModel)
            .where(IngestJobModel.status.in_(("pending", "running")))
            .order_by(IngestJobModel.created_at.asc())
            .limit(limit)
        )
        return [_job_to_dataclass(item) for item in result]


async def mark_job_running(settings: AppSettings, job_id: str) -> None:
    await _update_job(settings, job_id, status="running", progress=10, error_message=None)


async def mark_job_progress(settings: AppSettings, job_id: str, progress: int) -> None:
    await _update_job(settings, job_id, progress=max(0, min(progress, 99)))


async def mark_job_succeeded(settings: AppSettings, job_id: str, *, file_id: str, progress: int = 100) -> None:
    await _update_job(settings, job_id, status="succeeded", progress=progress, file_id=UUID(file_id))


async def mark_job_failed(settings: AppSettings, job_id: str, error_message: str) -> None:
    job = await get_ingest_job(settings, job_id)
    retry_count = (job.retry_count + 1) if job else 1
    await _update_job(
        settings,
        job_id,
        status="failed",
        progress=100,
        error_message=error_message[:4000],
        retry_count=retry_count,
    )


async def _update_job(settings: AppSettings, job_id: str, **values: Any) -> None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            job = await session.get(IngestJobModel, UUID(job_id))
            if job is None:
                raise ValueError(f"Ingest job not found: {job_id}")
            for key, value in values.items():
                setattr(job, key, value)


def _job_to_dataclass(model: IngestJobModel) -> IngestJob:
    return IngestJob(
        id=str(model.id),
        org_id=str(model.org_id),
        knowledge_base_id=str(model.knowledge_base_id),
        created_by_user_id=str(model.created_by_user_id),
        source_type=model.source_type,
        source_uri=model.source_uri,
        filename=model.filename,
        status=model.status,
        progress=int(model.progress),
        error_message=model.error_message,
        retry_count=int(model.retry_count),
        payload=dict(model.payload or {}),
        file_id=str(model.file_id) if model.file_id else None,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )
