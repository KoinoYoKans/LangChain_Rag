from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from config.database import ChunkLocationModel, DocumentPageModel, RagFileChunkModel, build_async_session_maker
from config.settings import AppSettings
from core.document_parser import ParsedPage


async def replace_document_pages(
    settings: AppSettings,
    *,
    org_id: str,
    knowledge_base_id: str,
    file_id: str,
    pages: list[ParsedPage],
) -> None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            await session.execute(delete(DocumentPageModel).where(DocumentPageModel.file_id == UUID(file_id)))
            if pages:
                await session.execute(
                    insert(DocumentPageModel).values(
                        [
                            {
                                "id": uuid4(),
                                "org_id": UUID(org_id),
                                "knowledge_base_id": UUID(knowledge_base_id),
                                "file_id": UUID(file_id),
                                "page_number": page.page_number,
                                "text": page.text,
                                "width": page.width,
                                "height": page.height,
                                "ocr_status": page.ocr_status,
                                "blocks": page.blocks or [],
                            }
                            for page in pages
                        ]
                    )
                )


async def replace_chunk_locations(
    settings: AppSettings,
    *,
    org_id: str,
    knowledge_base_id: str,
    file_id: str,
    locations: list[dict[str, Any]],
) -> None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            await session.execute(delete(ChunkLocationModel).where(ChunkLocationModel.file_id == UUID(file_id)))
            if locations:
                await session.execute(
                    insert(ChunkLocationModel).values(
                        [
                            {
                                "id": uuid4(),
                                "org_id": UUID(org_id),
                                "knowledge_base_id": UUID(knowledge_base_id),
                                "file_id": UUID(file_id),
                                "chunk_id": UUID(item["chunk_id"]),
                                "page_number": int(item["page_number"]),
                                "bbox": item.get("bbox") or {},
                                "char_start": item.get("char_start"),
                                "char_end": item.get("char_end"),
                            }
                            for item in locations
                        ]
                    )
                )


async def list_document_pages(settings: AppSettings, file_id: str) -> list[dict[str, Any]]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        result = await session.scalars(
            select(DocumentPageModel)
            .where(DocumentPageModel.file_id == UUID(file_id))
            .order_by(DocumentPageModel.page_number.asc())
        )
        return [
            {
                "id": str(page.id),
                "page_number": page.page_number,
                "text": page.text,
                "width": page.width,
                "height": page.height,
                "ocr_status": page.ocr_status,
                "blocks": list(page.blocks or []),
            }
            for page in result
        ]


async def list_document_chunks(settings: AppSettings, file_id: str) -> list[dict[str, Any]]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        rows = await session.execute(
            select(RagFileChunkModel, ChunkLocationModel)
            .outerjoin(ChunkLocationModel, RagFileChunkModel.id == ChunkLocationModel.chunk_id)
            .where(RagFileChunkModel.file_id == UUID(file_id))
            .order_by(RagFileChunkModel.chunk_index.asc())
        )
        chunks: list[dict[str, Any]] = []
        for chunk, location in rows:
            chunks.append(
                {
                    "id": str(chunk.id),
                    "vector_id": str(chunk.vector_id),
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "content_sha256": chunk.content_sha256,
                    "keywords": list(chunk.keywords or []),
                    "location": {
                        "page_number": location.page_number,
                        "bbox": dict(location.bbox or {}),
                        "char_start": location.char_start,
                        "char_end": location.char_end,
                    }
                    if location
                    else None,
                    "created_at": chunk.created_at.isoformat(),
                }
            )
        return chunks


async def delete_document_artifacts(settings: AppSettings, file_id: str) -> None:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        async with session.begin():
            await session.execute(delete(DocumentPageModel).where(DocumentPageModel.file_id == UUID(file_id)))
            await session.execute(delete(ChunkLocationModel).where(ChunkLocationModel.file_id == UUID(file_id)))
