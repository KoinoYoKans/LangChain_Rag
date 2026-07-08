from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import Any
from uuid import UUID as PyUUID

from dotenv import load_dotenv
from sqlalchemy import BigInteger, CheckConstraint, DateTime, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config.settings import AppSettings

load_dotenv(".env")


class Base(AsyncAttrs, DeclarativeBase):
    pass


class RagFileModel(Base):
    __tablename__ = os.getenv("RAG_FILE_TABLE", "rag_files")
    __table_args__ = (
        CheckConstraint(
            "status in ('processing', 'completed', 'failed')",
            name="rag_files_status_check",
        ),
        Index(
            f"{__tablename__}_content_sha256_uidx",
            "user_id",
            "content_sha256",
            unique=True,
            postgresql_where=text("status in ('processing', 'completed')"),
        ),
        Index(f"{__tablename__}_status_created_idx", "user_id", "status", "created_at"),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False, default="default")
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(Text, nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    file_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_ids: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    vector_ids: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class RagFileChunkModel(Base):
    __tablename__ = os.getenv("RAG_CHUNK_TABLE", "rag_file_chunks")
    __table_args__ = (
        Index(f"{__tablename__}_user_file_idx", "user_id", "file_id", "chunk_index"),
        Index(f"{__tablename__}_file_chunk_uidx", "file_id", "chunk_index", unique=True),
        Index(f"{__tablename__}_vector_uidx", "vector_id", unique=True),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    file_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    vector_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RagConversationModel(Base):
    __tablename__ = os.getenv("RAG_CONVERSATION_TABLE", "rag_conversations")
    __table_args__ = (
        Index(f"{__tablename__}_user_updated_idx", "user_id", "updated_at"),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class RagMessageModel(Base):
    __tablename__ = os.getenv("RAG_MESSAGE_TABLE", "rag_messages")
    __table_args__ = (
        CheckConstraint(
            "role in ('user', 'assistant', 'system')",
            name="rag_messages_role_check",
        ),
        Index(f"{__tablename__}_conversation_idx", "user_id", "conversation_id", "created_at"),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


def build_async_database_url(settings: AppSettings) -> str:
    return settings.asyncpg_postgres_dsn


@lru_cache(maxsize=8)
def _build_async_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_pre_ping=True)


def build_async_session_maker(settings: AppSettings) -> async_sessionmaker:
    engine = _build_async_engine(build_async_database_url(settings))
    return async_sessionmaker(engine, expire_on_commit=False)


async def initialize_orm_tables(settings: AppSettings) -> None:
    engine = _build_async_engine(build_async_database_url(settings))
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.execute(
            text(
                f'alter table "{settings.rag_file_table}" '
                "add column if not exists user_id text not null default 'default'"
            )
        )
        await connection.execute(
            text(
                f'alter table "{settings.rag_file_table}" '
                "add column if not exists chunk_ids text[] not null default '{}'"
            )
        )
        await connection.execute(
            text(f'drop index if exists "{settings.rag_file_table}_content_sha256_uidx"')
        )
        await connection.execute(
            text(
                f'create unique index if not exists "{settings.rag_file_table}_content_sha256_uidx" '
                f'on "{settings.rag_file_table}" (user_id, content_sha256) '
                "where status in ('processing', 'completed')"
            )
        )
