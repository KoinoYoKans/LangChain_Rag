from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from config.database import RagFileChunkModel, RagFileModel, build_async_session_maker, initialize_orm_tables
from config.settings import AppSettings


@dataclass(frozen=True)
class RagFile:
    id: str
    user_id: str
    filename: str
    content_type: str
    file_size: int
    file_sha256: str
    content_sha256: str
    chunk_count: int
    chunk_ids: list[str]
    vector_ids: list[str]
    status: str
    error_message: str | None
    created_at: datetime
    updated_at: datetime


async def initialize_file_table(settings: AppSettings) -> None:
    await initialize_orm_tables(settings)


async def find_file_by_content_hash(
    settings: AppSettings,
    user_id: str,
    content_sha256: str,
) -> RagFile | None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        result = await session.scalar(
            select(RagFileModel).where(
                RagFileModel.user_id == user_id,
                RagFileModel.content_sha256 == content_sha256,
                RagFileModel.status.in_(("processing", "completed")),
            )
        )
        return _model_to_file(result) if result else None


async def create_processing_file(
    settings: AppSettings,
    *,
    file_id: str,
    user_id: str,
    filename: str,
    content_type: str,
    file_size: int,
    file_sha256: str,
    content_sha256: str,
) -> RagFile:
    file_model = RagFileModel(
        id=UUID(file_id),
        user_id=user_id,
        filename=filename,
        content_type=content_type,
        file_size=file_size,
        file_sha256=file_sha256,
        content_sha256=content_sha256,
        chunk_count=0,
        chunk_ids=[],
        vector_ids=[],
        status="processing",
    )
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            session.add(file_model)
        await session.refresh(file_model)
        return _model_to_file(file_model)


async def mark_file_completed(
    settings: AppSettings,
    file_id: str,
    *,
    chunk_count: int,
    chunk_ids: list[str],
    vector_ids: list[str],
) -> RagFile:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        file_model = await session.get(RagFileModel, UUID(file_id))
        if file_model is None:
            raise ValueError(f"File record not found: {file_id}")
        file_model.chunk_count = chunk_count
        file_model.chunk_ids = chunk_ids
        file_model.vector_ids = vector_ids
        file_model.status = "completed"
        file_model.error_message = None
        await session.commit()
        await session.refresh(file_model)
        return _model_to_file(file_model)


async def mark_file_failed(settings: AppSettings, file_id: str, error_message: str) -> RagFile | None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        file_model = await session.get(RagFileModel, UUID(file_id))
        if file_model is None:
            return None
        file_model.status = "failed"
        file_model.error_message = error_message[:4000]
        await session.commit()
        await session.refresh(file_model)
        return _model_to_file(file_model)


async def save_file_chunks(
    settings: AppSettings,
    *,
    user_id: str,
    file_id: str,
    chunks: list[tuple[str, int, str, str]],
) -> None:
    if not chunks:
        return
    values = [
        {
            "id": UUID(chunk_id),
            "user_id": user_id,
            "file_id": UUID(file_id),
            "vector_id": UUID(chunk_id),
            "chunk_index": chunk_index,
            "content": content,
            "content_sha256": content_sha256,
        }
        for chunk_id, chunk_index, content, content_sha256 in chunks
    ]
    stmt = insert(RagFileChunkModel).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[RagFileChunkModel.id],
        set_={
            "content": stmt.excluded.content,
            "content_sha256": stmt.excluded.content_sha256,
        },
    )
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            await session.execute(stmt)


async def delete_file_chunks(settings: AppSettings, user_id: str, file_id: str) -> None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            await session.execute(
                delete(RagFileChunkModel).where(
                    RagFileChunkModel.user_id == user_id,
                    RagFileChunkModel.file_id == UUID(file_id),
                )
            )


async def list_files(settings: AppSettings, user_id: str, limit: int = 50, offset: int = 0) -> list[RagFile]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        result = await session.scalars(
            select(RagFileModel)
            .where(RagFileModel.user_id == user_id)
            .order_by(RagFileModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return [_model_to_file(item) for item in result]


async def get_file(settings: AppSettings, user_id: str, file_id: str) -> RagFile | None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        result = await session.scalar(
            select(RagFileModel).where(
                RagFileModel.id == UUID(file_id),
                RagFileModel.user_id == user_id,
            )
        )
        return _model_to_file(result) if result else None


async def delete_file_record(settings: AppSettings, user_id: str, file_id: str) -> RagFile | None:
    existing = await get_file(settings, user_id, file_id)
    if existing is None:
        return None
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            await session.execute(
                delete(RagFileModel).where(
                    RagFileModel.id == UUID(file_id),
                    RagFileModel.user_id == user_id,
                )
            )
    return existing


def _model_to_file(model: RagFileModel) -> RagFile:
    return RagFile(
        id=str(model.id),
        user_id=model.user_id,
        filename=model.filename,
        content_type=model.content_type,
        file_size=int(model.file_size),
        file_sha256=model.file_sha256,
        content_sha256=model.content_sha256,
        chunk_count=int(model.chunk_count),
        chunk_ids=list(model.chunk_ids or []),
        vector_ids=list(model.vector_ids or []),
        status=model.status,
        error_message=model.error_message,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )
