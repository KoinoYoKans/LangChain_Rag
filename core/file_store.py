from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy import update as sqlalchemy_update
from sqlalchemy.dialects.postgresql import insert

from config.database import RagFileChunkModel, RagFileModel, build_async_session_maker, initialize_orm_tables
from config.settings import AppSettings


@dataclass(frozen=True)
class RagFile:
    id: str
    org_id: str | None
    knowledge_base_id: str | None
    owner_user_id: str | None
    user_id: str
    filename: str
    content_type: str
    source_type: str
    source_uri: str | None
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
    knowledge_base_id: str | None = None,
) -> RagFile | None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        conditions = [
            RagFileModel.user_id == user_id,
            RagFileModel.content_sha256 == content_sha256,
            RagFileModel.status.in_(("processing", "completed")),
        ]
        if knowledge_base_id:
            conditions.append(RagFileModel.knowledge_base_id == UUID(knowledge_base_id))
        result = await session.scalar(select(RagFileModel).where(*conditions))
        return _model_to_file(result) if result else None


async def create_processing_file(
    settings: AppSettings,
    *,
    file_id: str,
    org_id: str | None = None,
    knowledge_base_id: str | None = None,
    owner_user_id: str | None = None,
    user_id: str,
    filename: str,
    content_type: str,
    source_type: str = "file",
    source_uri: str | None = None,
    file_size: int,
    file_sha256: str,
    content_sha256: str,
) -> RagFile:
    file_model = RagFileModel(
        id=UUID(file_id),
        org_id=UUID(org_id) if org_id else None,
        knowledge_base_id=UUID(knowledge_base_id) if knowledge_base_id else None,
        owner_user_id=UUID(owner_user_id) if owner_user_id else None,
        user_id=user_id,
        filename=filename,
        content_type=content_type,
        source_type=source_type,
        source_uri=source_uri,
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
    org_id: str | None = None,
    knowledge_base_id: str | None = None,
    file_id: str,
    chunks: list[tuple[str, int, str, str]],
) -> None:
    if not chunks:
        return
    values = [
        {
            "id": UUID(chunk_id),
            "org_id": UUID(org_id) if org_id else None,
            "knowledge_base_id": UUID(knowledge_base_id) if knowledge_base_id else None,
            "user_id": user_id,
            "file_id": UUID(file_id),
            "vector_id": UUID(chunk_id),
            "chunk_index": chunk_index,
            "content": content,
            "content_sha256": content_sha256,
            "keywords": extract_keywords(content),
        }
        for chunk_id, chunk_index, content, content_sha256 in chunks
    ]
    stmt = insert(RagFileChunkModel).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[RagFileChunkModel.id],
        set_={
            "content": stmt.excluded.content,
            "content_sha256": stmt.excluded.content_sha256,
            "org_id": stmt.excluded.org_id,
            "knowledge_base_id": stmt.excluded.knowledge_base_id,
            "keywords": stmt.excluded.keywords,
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


async def list_knowledge_base_files(
    settings: AppSettings,
    knowledge_base_id: str,
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    include_deleted: bool = False,
) -> list[RagFile]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        conditions = [RagFileModel.knowledge_base_id == UUID(knowledge_base_id)]
        if status:
            conditions.append(RagFileModel.status == status)
        elif not include_deleted:
            conditions.append(RagFileModel.status != "deleted")
        result = await session.scalars(
            select(RagFileModel)
            .where(*conditions)
            .order_by(RagFileModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return [_model_to_file(item) for item in result]


async def count_completed_knowledge_base_files(settings: AppSettings, knowledge_base_id: str) -> int:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(RagFileModel)
            .where(RagFileModel.knowledge_base_id == UUID(knowledge_base_id), RagFileModel.status == "completed")
        )
        return int(count or 0)


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


async def get_knowledge_base_file(settings: AppSettings, knowledge_base_id: str, file_id: str) -> RagFile | None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        result = await session.scalar(
            select(RagFileModel).where(
                RagFileModel.id == UUID(file_id),
                RagFileModel.knowledge_base_id == UUID(knowledge_base_id),
                RagFileModel.status != "deleted",
            )
        )
        return _model_to_file(result) if result else None


async def delete_file_chunks_by_file_id(settings: AppSettings, file_id: str) -> None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            await session.execute(delete(RagFileChunkModel).where(RagFileChunkModel.file_id == UUID(file_id)))


async def mark_file_deleted(settings: AppSettings, file_id: str) -> None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            await session.execute(
                sqlalchemy_update(RagFileModel)
                .where(RagFileModel.id == UUID(file_id))
                .values(status="deleted", deleted_at=datetime.utcnow())
            )


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
        org_id=str(model.org_id) if model.org_id else None,
        knowledge_base_id=str(model.knowledge_base_id) if model.knowledge_base_id else None,
        owner_user_id=str(model.owner_user_id) if model.owner_user_id else None,
        user_id=model.user_id,
        filename=model.filename,
        content_type=model.content_type,
        source_type=model.source_type,
        source_uri=model.source_uri,
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


def extract_keywords(text: str, limit: int = 24) -> list[str]:
    try:
        import jieba

        words = jieba.lcut(text)
    except Exception:  # noqa: BLE001
        words = text.split()
    seen: set[str] = set()
    result: list[str] = []
    for word in words:
        value = word.strip().lower()
        if len(value) < 2 or value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= limit:
            break
    return result
