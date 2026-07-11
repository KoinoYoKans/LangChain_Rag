from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, select, update

from config.database import IngestJobModel, RagFileModel, build_async_session_maker
from config.settings import AppSettings


class IngestJobCancelledError(Exception):
    pass


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
    duration_ms: int | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class IngestQueueHealth:
    pending_count: int
    running_count: int
    succeeded_count: int
    failed_count: int
    cancelled_count: int
    oldest_pending_at: datetime | None
    oldest_running_at: datetime | None


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


async def list_ingest_jobs(
    settings: AppSettings,
    knowledge_base_id: str,
    limit: int = 100,
    status: str = "all",
) -> list[IngestJob]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        conditions = [IngestJobModel.knowledge_base_id == UUID(knowledge_base_id)]
        if status == "active":
            conditions.append(IngestJobModel.status.in_(("pending", "running")))
        elif status == "history":
            conditions.append(IngestJobModel.status.in_(("succeeded", "failed", "cancelled")))
        elif status != "all":
            conditions.append(IngestJobModel.status == status)
        result = await session.scalars(
            select(IngestJobModel)
            .where(*conditions)
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


async def list_pending_ingest_jobs(settings: AppSettings, limit: int = 50) -> list[IngestJob]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        result = await session.scalars(
            select(IngestJobModel)
            .where(IngestJobModel.status == "pending")
            .order_by(IngestJobModel.created_at.asc())
            .limit(limit)
        )
        return [_job_to_dataclass(item) for item in result]


async def requeue_stale_running_ingest_jobs(settings: AppSettings, stale_after_seconds: int) -> int:
    """Return jobs abandoned by a terminated worker to the durable pending queue."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            result = await session.execute(
                update(IngestJobModel)
                .where(
                    IngestJobModel.status == "running",
                    IngestJobModel.updated_at < cutoff,
                )
                .values(
                    status="pending",
                    progress=0,
                    retry_count=IngestJobModel.retry_count + 1,
                    error_message="Recovered after worker lease expired.",
                    updated_at=func.now(),
                )
            )
        return int(result.rowcount or 0)


async def get_ingest_queue_health(settings: AppSettings, knowledge_base_id: str) -> IngestQueueHealth:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        counts: dict[str, int] = {}
        for status in ("pending", "running", "succeeded", "failed", "cancelled"):
            value = await session.scalar(
                select(func.count())
                .select_from(IngestJobModel)
                .where(
                    IngestJobModel.knowledge_base_id == UUID(knowledge_base_id),
                    IngestJobModel.status == status,
                )
            )
            counts[status] = int(value or 0)
        oldest_pending_at = await session.scalar(
            select(func.min(IngestJobModel.created_at)).where(
                IngestJobModel.knowledge_base_id == UUID(knowledge_base_id),
                IngestJobModel.status == "pending",
            )
        )
        oldest_running_at = await session.scalar(
            select(func.min(IngestJobModel.updated_at)).where(
                IngestJobModel.knowledge_base_id == UUID(knowledge_base_id),
                IngestJobModel.status == "running",
            )
        )
        return IngestQueueHealth(
            pending_count=counts["pending"],
            running_count=counts["running"],
            succeeded_count=counts["succeeded"],
            failed_count=counts["failed"],
            cancelled_count=counts["cancelled"],
            oldest_pending_at=oldest_pending_at,
            oldest_running_at=oldest_running_at,
        )


async def mark_job_running(settings: AppSettings, job_id: str) -> None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            result = await session.execute(
                update(IngestJobModel)
                .where(IngestJobModel.id == UUID(job_id), IngestJobModel.status == "pending")
                .values(status="running", progress=10, error_message=None, updated_at=func.now())
                .returning(IngestJobModel.id)
            )
            if result.first() is not None:
                return
            current_status = await session.scalar(select(IngestJobModel.status).where(IngestJobModel.id == UUID(job_id)))
            if current_status is None:
                raise ValueError(f"Ingest job not found: {job_id}")
            if current_status == "cancelled":
                raise IngestJobCancelledError("Ingest job has been cancelled")
            raise ValueError(f"Ingest job is not pending: {current_status}")


async def mark_stale_pending_jobs_failed(
    settings: AppSettings,
    knowledge_base_id: str,
    stale_after_seconds: int = 1800,
) -> list[IngestJob]:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)
    failed_at = datetime.now(timezone.utc)
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            result = await session.scalars(
                update(IngestJobModel)
                .where(
                    IngestJobModel.knowledge_base_id == UUID(knowledge_base_id),
                    IngestJobModel.status == "pending",
                    IngestJobModel.created_at < cutoff,
                )
                .values(
                    status="failed",
                    progress=100,
                    error_message="Pending timed out before worker pickup. Check worker/Redis health, then retry this job.",
                    updated_at=failed_at,
                )
                .returning(IngestJobModel)
            )
            stale_jobs = list(result)
        return [_job_to_dataclass(job) for job in stale_jobs]


async def mark_job_progress(settings: AppSettings, job_id: str, progress: int) -> None:
    await _update_job(settings, job_id, allowed_statuses={"running"}, progress=max(0, min(progress, 99)))


async def mark_job_file_id(settings: AppSettings, job_id: str, file_id: str) -> None:
    await _update_job(settings, job_id, allowed_statuses={"running"}, file_id=UUID(file_id))


async def mark_job_succeeded(settings: AppSettings, job_id: str, *, file_id: str, progress: int = 100) -> None:
    await _update_job(settings, job_id, allowed_statuses={"running"}, status="succeeded", progress=progress, file_id=UUID(file_id))


async def mark_job_failed(settings: AppSettings, job_id: str, error_message: str) -> None:
    await _update_job(
        settings,
        job_id,
        allowed_statuses={"pending", "running"},
        status="failed",
        progress=100,
        error_message=error_message[:4000],
    )


async def retry_failed_ingest_job(settings: AppSettings, job_id: str) -> IngestJob:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            job = await session.get(IngestJobModel, UUID(job_id))
            if job is None:
                raise ValueError(f"Ingest job not found: {job_id}")
            if job.status != "failed":
                raise ValueError("Only failed ingest jobs can be retried")
            if job.file_id:
                await session.execute(
                    update(RagFileModel)
                    .where(RagFileModel.id == job.file_id, RagFileModel.status == "failed")
                    .values(status="deleted", deleted_at=func.now())
                )
            job.status = "pending"
            job.progress = 0
            job.error_message = None
            job.retry_count = int(job.retry_count or 0) + 1
            job.file_id = None
        await session.refresh(job)
        return _job_to_dataclass(job)


async def cancel_ingest_job(settings: AppSettings, job_id: str) -> IngestJob:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            result = await session.scalars(
                update(IngestJobModel)
                .where(IngestJobModel.id == UUID(job_id), IngestJobModel.status.in_(("pending", "running")))
                .values(status="cancelled", progress=100, error_message="Cancelled by user", updated_at=func.now())
                .returning(IngestJobModel)
            )
            job = result.one_or_none()
            if job is None:
                current = await session.get(IngestJobModel, UUID(job_id))
                if current is None:
                    raise ValueError(f"Ingest job not found: {job_id}")
                raise ValueError("Only pending or running ingest jobs can be cancelled")
            if job.file_id:
                await session.execute(
                    update(RagFileModel)
                    .where(RagFileModel.id == job.file_id, RagFileModel.status.in_(("processing", "failed")))
                    .values(status="deleted", deleted_at=func.now())
                )
        return _job_to_dataclass(job)


async def _update_job(settings: AppSettings, job_id: str, allowed_statuses: set[str] | None = None, **values: Any) -> None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            update_values = {**values, "updated_at": func.now()}
            statement = update(IngestJobModel).where(IngestJobModel.id == UUID(job_id))
            if allowed_statuses is not None:
                statement = statement.where(IngestJobModel.status.in_(tuple(allowed_statuses)))
            result = await session.execute(statement.values(**update_values).returning(IngestJobModel.status))
            if result.first() is not None:
                return
            current_status = await session.scalar(select(IngestJobModel.status).where(IngestJobModel.id == UUID(job_id)))
            if current_status is None:
                raise ValueError(f"Ingest job not found: {job_id}")
            if current_status == "cancelled" and values.get("status") != "cancelled":
                raise IngestJobCancelledError("Ingest job has been cancelled")
            raise ValueError(f"Ingest job is not in an updatable state: {current_status}")


def _job_to_dataclass(model: IngestJobModel) -> IngestJob:
    duration_ms = None
    if model.created_at and model.updated_at and model.status in {"succeeded", "failed", "cancelled"}:
        duration_ms = max(0, int((model.updated_at - model.created_at).total_seconds() * 1000))
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
        duration_ms=duration_ms,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )
