from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import Any
from uuid import UUID as PyUUID

from dotenv import load_dotenv
from sqlalchemy import BigInteger, CheckConstraint, DateTime, Float, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config.settings import AppSettings

load_dotenv(".env")


class Base(AsyncAttrs, DeclarativeBase):
    pass


class OrganizationModel(Base):
    __tablename__ = os.getenv("ORG_TABLE", "rag_organizations")

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DepartmentModel(Base):
    __tablename__ = os.getenv("DEPARTMENT_TABLE", "rag_departments")
    __table_args__ = (
        Index("rag_departments_org_name_uidx", "org_id", "name", unique=True),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    org_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    parent_id: Mapped[PyUUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class UserModel(Base):
    __tablename__ = os.getenv("USER_TABLE", "rag_users")
    __table_args__ = (
        Index("rag_users_org_email_uidx", "org_id", "email", unique=True),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    org_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    department_id: Mapped[PyUUID | None] = mapped_column(UUID(as_uuid=True))
    email: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False, default="member")
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class KnowledgeBaseModel(Base):
    __tablename__ = os.getenv("KNOWLEDGE_BASE_TABLE", "rag_knowledge_bases")
    __table_args__ = (
        Index(
            "rag_knowledge_bases_org_name_uidx",
            "org_id",
            "name",
            unique=True,
            postgresql_where=text("status != 'deleted'"),
        ),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    org_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    owner_user_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    visibility: Mapped[str] = mapped_column(String, nullable=False, default="department")
    department_ids: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class KnowledgeBaseMemberModel(Base):
    __tablename__ = os.getenv("KNOWLEDGE_BASE_MEMBER_TABLE", "rag_knowledge_base_members")
    __table_args__ = (
        Index("rag_kb_members_uidx", "knowledge_base_id", "user_id", unique=True),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    knowledge_base_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    user_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False, default="member")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class IngestJobModel(Base):
    __tablename__ = os.getenv("INGEST_JOB_TABLE", "rag_ingest_jobs")
    __table_args__ = (
        Index("rag_ingest_jobs_kb_status_idx", "knowledge_base_id", "status", "created_at"),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    org_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    knowledge_base_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    created_by_user_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text)
    filename: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    file_id: Mapped[PyUUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class AuditLogModel(Base):
    __tablename__ = os.getenv("AUDIT_LOG_TABLE", "rag_audit_logs")
    __table_args__ = (
        Index("rag_audit_logs_org_created_idx", "org_id", "created_at"),
        Index("rag_audit_logs_actor_created_idx", "actor_user_id", "created_at"),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    org_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    actor_user_id: Mapped[PyUUID | None] = mapped_column(UUID(as_uuid=True))
    action: Mapped[str] = mapped_column(Text, nullable=False)
    target_type: Mapped[str | None] = mapped_column(Text)
    target_id: Mapped[str | None] = mapped_column(Text)
    actor_department_id: Mapped[PyUUID | None] = mapped_column(UUID(as_uuid=True))
    ip_address: Mapped[str | None] = mapped_column(Text)
    user_agent: Mapped[str | None] = mapped_column(Text)
    request_id: Mapped[str | None] = mapped_column(Text)
    result: Mapped[str] = mapped_column(String, nullable=False, default="success")
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChatLogModel(Base):
    __tablename__ = os.getenv("CHAT_LOG_TABLE", "rag_chat_logs")
    __table_args__ = (
        Index("rag_chat_logs_kb_created_idx", "knowledge_base_id", "created_at"),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    org_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    knowledge_base_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    user_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    conversation_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FeedbackModel(Base):
    __tablename__ = os.getenv("FEEDBACK_TABLE", "rag_feedback")
    __table_args__ = (
        CheckConstraint("rating in ('up', 'down')", name="rag_feedback_rating_check"),
        Index("rag_feedback_org_created_idx", "org_id", "created_at"),
        Index("rag_feedback_kb_created_idx", "knowledge_base_id", "created_at"),
        Index("rag_feedback_message_idx", "assistant_message_id"),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    org_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    knowledge_base_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    user_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    conversation_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    assistant_message_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    rating: Mapped[str] = mapped_column(String, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    comment: Mapped[str | None] = mapped_column(Text)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    sources_snapshot: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RagFileModel(Base):
    __tablename__ = os.getenv("RAG_FILE_TABLE", "rag_files")
    __table_args__ = (
        CheckConstraint(
            "status in ('processing', 'completed', 'failed', 'deleted')",
            name="rag_files_status_check",
        ),
        Index(
            f"{__tablename__}_content_sha256_uidx",
            "user_id",
            "knowledge_base_id",
            "content_sha256",
            unique=True,
            postgresql_where=text("status in ('processing', 'completed')"),
        ),
        Index(f"{__tablename__}_kb_status_created_idx", "knowledge_base_id", "status", "created_at"),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    org_id: Mapped[PyUUID | None] = mapped_column(UUID(as_uuid=True))
    knowledge_base_id: Mapped[PyUUID | None] = mapped_column(UUID(as_uuid=True))
    owner_user_id: Mapped[PyUUID | None] = mapped_column(UUID(as_uuid=True))
    user_id: Mapped[str] = mapped_column(String, nullable=False, default="default")
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False, default="file")
    source_uri: Mapped[str | None] = mapped_column(Text)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    file_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_ids: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    vector_ids: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    status: Mapped[str] = mapped_column(String, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class RagFileChunkModel(Base):
    __tablename__ = os.getenv("RAG_CHUNK_TABLE", "rag_file_chunks")
    __table_args__ = (
        Index(f"{__tablename__}_kb_file_idx", "knowledge_base_id", "file_id", "chunk_index"),
        Index(f"{__tablename__}_file_chunk_uidx", "file_id", "chunk_index", unique=True),
        Index(f"{__tablename__}_vector_uidx", "vector_id", unique=True),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    org_id: Mapped[PyUUID | None] = mapped_column(UUID(as_uuid=True))
    knowledge_base_id: Mapped[PyUUID | None] = mapped_column(UUID(as_uuid=True))
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    file_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    vector_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    keywords: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DocumentPageModel(Base):
    __tablename__ = os.getenv("DOCUMENT_PAGE_TABLE", "rag_document_pages")
    __table_args__ = (
        Index("rag_document_pages_file_page_uidx", "file_id", "page_number", unique=True),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    org_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    knowledge_base_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    file_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    width: Mapped[float | None] = mapped_column(Float)
    height: Mapped[float | None] = mapped_column(Float)
    ocr_status: Mapped[str] = mapped_column(String, nullable=False, default="not_required")
    blocks: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChunkLocationModel(Base):
    __tablename__ = os.getenv("CHUNK_LOCATION_TABLE", "rag_chunk_locations")
    __table_args__ = (
        Index("rag_chunk_locations_chunk_idx", "chunk_id"),
        Index("rag_chunk_locations_file_page_idx", "file_id", "page_number"),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    org_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    knowledge_base_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    file_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    chunk_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False, default=dict)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ApiKeyModel(Base):
    __tablename__ = os.getenv("API_KEY_TABLE", "rag_api_keys")
    __table_args__ = (
        Index("rag_api_keys_prefix_uidx", "key_prefix", unique=True),
        Index("rag_api_keys_user_kb_idx", "user_id", "knowledge_base_id"),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    org_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    user_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    knowledge_base_id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String, nullable=False)
    key_hash: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RagConversationModel(Base):
    __tablename__ = os.getenv("RAG_CONVERSATION_TABLE", "rag_conversations")
    __table_args__ = (
        Index(f"{__tablename__}_kb_user_updated_idx", "knowledge_base_id", "user_id", "updated_at"),
    )

    id: Mapped[PyUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    org_id: Mapped[PyUUID | None] = mapped_column(UUID(as_uuid=True))
    knowledge_base_id: Mapped[PyUUID | None] = mapped_column(UUID(as_uuid=True))
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
    org_id: Mapped[PyUUID | None] = mapped_column(UUID(as_uuid=True))
    knowledge_base_id: Mapped[PyUUID | None] = mapped_column(UUID(as_uuid=True))
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
        await _migrate_existing_tables(connection, settings)


async def _migrate_existing_tables(connection: Any, settings: AppSettings) -> None:
    await connection.execute(text("set lock_timeout = '5s'"))
    await connection.execute(text("set statement_timeout = '30s'"))
    file_table = settings.rag_file_table
    chunk_table = settings.rag_chunk_table
    conversation_table = settings.rag_conversation_table
    message_table = settings.rag_message_table

    for table_name in (file_table, chunk_table, conversation_table, message_table):
        await connection.execute(text(f'alter table "{table_name}" add column if not exists org_id uuid'))
        await connection.execute(text(f'alter table "{table_name}" add column if not exists knowledge_base_id uuid'))

    await connection.execute(text('alter table "rag_knowledge_bases" add column if not exists status text not null default \'active\''))
    await connection.execute(text('alter table "rag_knowledge_bases" add column if not exists deleted_at timestamp with time zone'))
    await connection.execute(text('drop index if exists "rag_knowledge_bases_org_name_uidx"'))
    await connection.execute(
        text(
            'create unique index if not exists "rag_knowledge_bases_org_name_uidx" '
            'on "rag_knowledge_bases" (org_id, name) '
            "where status != 'deleted'"
        )
    )

    await connection.execute(text(f'alter table "{file_table}" add column if not exists owner_user_id uuid'))
    await connection.execute(text(f'alter table "{file_table}" add column if not exists chunk_ids text[] not null default \'{{}}\''))
    await connection.execute(text(f'alter table "{file_table}" add column if not exists vector_ids text[] not null default \'{{}}\''))
    await connection.execute(text(f'alter table "{file_table}" add column if not exists source_type text not null default \'file\''))
    await connection.execute(text(f'alter table "{file_table}" add column if not exists source_uri text'))
    await connection.execute(text(f'alter table "{file_table}" add column if not exists deleted_at timestamp with time zone'))

    await connection.execute(text(f'alter table "{chunk_table}" add column if not exists user_id text not null default \'default\''))
    await connection.execute(text(f'alter table "{chunk_table}" add column if not exists keywords text[] not null default \'{{}}\''))
    await connection.execute(text(f'alter table "{message_table}" add column if not exists org_id uuid'))

    await connection.execute(text('alter table "rag_users" add column if not exists last_login_at timestamp with time zone'))
    audit_columns = {
        "actor_department_id": "uuid",
        "ip_address": "text",
        "user_agent": "text",
        "request_id": "text",
        "result": "text not null default 'success'",
        "latency_ms": "integer",
        "error_message": "text",
    }
    for column, column_type in audit_columns.items():
        await connection.execute(text(f'alter table "rag_audit_logs" add column if not exists {column} {column_type}'))

    await connection.execute(text(f'drop index if exists "{file_table}_content_sha256_uidx"'))
    await connection.execute(
        text(
            f'create unique index if not exists "{file_table}_content_sha256_uidx" '
            f'on "{file_table}" (user_id, knowledge_base_id, content_sha256) '
            "where status in ('processing', 'completed')"
        )
    )
