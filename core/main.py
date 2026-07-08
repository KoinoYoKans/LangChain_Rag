from __future__ import annotations

import asyncio
import ipaddress
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from config.model import build_chat_model
from config.settings import AppSettings
from core.auth import CurrentUser, create_access_token, require_current_user
from core.api_key_store import ApiKey, create_api_key, get_api_key, list_api_keys, revoke_api_key, revoke_api_keys_for_kb_user, verify_api_key
from core.conversation_store import (
    Conversation,
    add_message,
    delete_conversation,
    ensure_conversation,
    get_conversation,
    list_conversation_messages,
    get_recent_messages,
    list_conversations,
    update_conversation_title,
)
from core.document_loader import (
    SUPPORTED_EXTENSIONS,
    build_chunks_from_text,
    extract_text,
    hash_bytes,
    hash_text,
)
from core.document_store import delete_document_artifacts, list_document_chunks, list_document_pages
from core.document_parser import parse_html
from core.embeddings import build_embeddings
from core.enterprise_store import (
    add_audit_log,
    add_chat_log,
    authenticate_user,
    bootstrap_default_org,
    create_department,
    create_knowledge_base,
    create_user,
    deactivate_user,
    get_user_by_id,
    get_knowledge_base_capabilities,
    get_knowledge_base_stats,
    list_knowledge_base_members,
    list_audit_logs,
    list_departments,
    list_knowledge_bases,
    list_users,
    remove_knowledge_base_member,
    require_knowledge_base_access,
    reset_user_password,
    soft_delete_knowledge_base,
    update_knowledge_base,
    update_user,
    upsert_knowledge_base_member,
)
from core.feedback_store import Feedback, create_feedback, list_feedback
from core.file_store import (
    RagFile,
    create_processing_file,
    count_completed_knowledge_base_files,
    delete_file_chunks,
    delete_file_chunks_by_file_id,
    delete_file_record,
    find_file_by_content_hash,
    get_file,
    get_knowledge_base_file,
    list_knowledge_base_files,
    list_files,
    mark_file_deleted,
    mark_file_completed,
    mark_file_failed,
    save_file_chunks,
)
from core.ingest_store import (
    IngestJob,
    IngestQueueHealth,
    cancel_ingest_job,
    create_ingest_job,
    get_ingest_job,
    get_ingest_queue_health,
    list_ingest_jobs,
    mark_job_failed,
    retry_failed_ingest_job,
)
from core.import_plan_store import (
    IMPORT_PLAN_TTL_SECONDS,
    acquire_import_plan_lock,
    load_import_plan,
    refresh_import_plan,
    release_import_plan_lock,
    save_import_plan,
)
from core.ingestion import _extract_title, _fetch_url, _slugify
from core.job_queue import enqueue_ingest_job, get_ingest_queue_length, get_worker_heartbeat
from core.rerank import RerankedDocument, Reranker, build_reranker
from core.retrieval import hybrid_search
from core.vector_store import (
    add_documents,
    build_vector_store,
    check_database,
    initialize_database,
    delete_documents,
    initialize_database_async,
    similarity_search,
)


class ChatRequest(BaseModel):
    knowledge_base_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank_top_n: int | None = Field(default=None, ge=0, le=50)


class Source(BaseModel):
    file_id: str | None
    chunk_id: str | None
    document_id: str | None
    filename: str | None
    chunk_index: int | None
    content_type: str | None
    upload_time: str | None
    vector_score: float | None = None
    bm25_score: float | None = None
    hybrid_score: float | None = None
    rerank_score: float | None = None
    page_number: int | None = None
    bbox: dict[str, float] | None = None
    snippet: str


class ChatResponse(BaseModel):
    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    answer: str
    sources: list[Source]
    confidence: str
    confidence_score: float | None = None


class FeedbackRequest(BaseModel):
    knowledge_base_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    assistant_message_id: str = Field(min_length=1)
    rating: str = Field(pattern="^(up|down)$")
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    reason: str | None = Field(default=None, max_length=120)
    comment: str | None = Field(default=None, max_length=1000)
    sources_snapshot: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    id: str
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
    created_at: str


class FeedbackListResponse(BaseModel):
    items: list[FeedbackResponse]


class DocumentUploadResponse(BaseModel):
    file_id: str
    document_id: str
    user_id: str
    filename: str
    content_sha256: str
    chunks: int
    chunk_ids: list[str]
    ids: list[str]


class FileRecordResponse(BaseModel):
    id: str
    org_id: str | None = None
    knowledge_base_id: str | None = None
    owner_user_id: str | None = None
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
    created_at: str
    updated_at: str


class FileListResponse(BaseModel):
    items: list[FileRecordResponse]


class DeleteDocumentResponse(BaseModel):
    file_id: str
    deleted_vectors: int
    deleted_chunks: int


class ConversationResponse(BaseModel):
    id: str
    user_id: str
    knowledge_base_id: str | None = None
    title: str | None
    created_at: str
    updated_at: str


class ConversationListResponse(BaseModel):
    items: list[ConversationResponse]


class ConversationUpdateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)


class DeleteConversationResponse(BaseModel):
    conversation_id: str
    deleted: bool


class ChatMessageResponse(BaseModel):
    id: str
    conversation_id: str
    knowledge_base_id: str | None = None
    role: str
    content: str
    metadata: dict[str, Any]
    created_at: str


class ChatMessageListResponse(BaseModel):
    items: list[ChatMessageResponse]


class LoginRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, Any]


class DepartmentCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    parent_id: str | None = None


class DepartmentResponse(BaseModel):
    id: str
    org_id: str
    name: str
    parent_id: str | None
    created_at: str


class UserCreateRequest(BaseModel):
    email: str = Field(min_length=3)
    display_name: str = Field(min_length=1)
    password: str = Field(min_length=8)
    role: str = Field(default="member", pattern="^(admin|manager|member)$")
    department_id: str | None = None


class UserUpdateRequest(BaseModel):
    display_name: str = Field(min_length=1)
    role: str = Field(default="member", pattern="^(admin|manager|member)$")
    department_id: str | None = None
    is_active: bool = True


class PasswordResetRequest(BaseModel):
    password: str = Field(min_length=8)


class UserResponse(BaseModel):
    id: str
    org_id: str
    department_id: str | None
    email: str
    display_name: str
    role: str
    is_active: bool
    last_login_at: str | None = None
    created_at: str


class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None
    visibility: str = Field(default="department", pattern="^(private|department|org)$")
    department_ids: list[str] = Field(default_factory=list)


class KnowledgeBaseUpdateRequest(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None
    visibility: str = Field(default="department", pattern="^(private|department|org)$")
    department_ids: list[str] = Field(default_factory=list)


class KnowledgeBaseResponse(BaseModel):
    id: str
    org_id: str
    owner_user_id: str
    name: str
    description: str | None
    visibility: str
    department_ids: list[str]
    status: str = "active"
    file_count: int = 0
    completed_file_count: int = 0
    failed_job_count: int = 0
    current_user_role: str = "none"
    can_read: bool = False
    can_write: bool = False
    can_manage_members: bool = False
    can_manage_settings: bool = False
    can_manage_api_keys: bool = False
    created_at: str
    updated_at: str


class KnowledgeBaseListResponse(BaseModel):
    items: list[KnowledgeBaseResponse]


class KnowledgeBaseMemberUpsertRequest(BaseModel):
    user_id: str = Field(min_length=1)
    role: str = Field(pattern="^(owner|editor|viewer)$")


class KnowledgeBaseMemberResponse(BaseModel):
    id: str
    knowledge_base_id: str
    user_id: str
    role: str
    email: str | None
    display_name: str | None
    department_id: str | None
    created_at: str


class KnowledgeBaseMemberListResponse(BaseModel):
    items: list[KnowledgeBaseMemberResponse]


class UrlIngestRequest(BaseModel):
    url: str = Field(min_length=8)
    filename: str | None = None


class UrlBatchPlanRequest(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=20)
    skip_duplicates: bool = True


class UrlBatchCommitRequest(BaseModel):
    plan_id: str = Field(min_length=1)
    client_item_ids: list[str] = Field(min_length=1, max_length=100)


class UrlImportPlanItem(BaseModel):
    index: int
    client_item_id: str
    source_type: str = "url"
    url: str
    filename: str | None = None
    content_type: str | None = None
    file_size: int | None = None
    status: str
    severity: str
    can_enqueue: bool
    reason_code: str | None = None
    reason: str | None = None
    content_sha256: str | None = None
    content_length: int | None = None
    estimated_chunks: int | None = None
    duplicate_file_id: str | None = None
    duplicate_of: str | None = None
    job_id: str | None = None
    confirmed_at: str | None = None


class UrlImportPlanResponse(BaseModel):
    plan_id: str
    knowledge_base_id: str
    created_at: str
    expires_at: str
    total: int
    ready_count: int
    warning_count: int
    blocked_count: int
    duplicate_count: int
    invalid_count: int
    error_count: int
    items: list[UrlImportPlanItem]


class IngestJobResponse(BaseModel):
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
    duration_ms: int | None = None
    created_at: str
    updated_at: str


class IngestJobListResponse(BaseModel):
    items: list[IngestJobResponse]


class QueueHealthResponse(BaseModel):
    pending_count: int
    running_count: int
    succeeded_count: int
    failed_count: int
    cancelled_count: int
    redis_queue_length: int
    oldest_pending_at: str | None
    oldest_pending_wait_seconds: int | None
    oldest_running_at: str | None
    oldest_running_seconds: int | None
    worker_last_seen_at: str | None
    worker_stale: bool


class DocumentBatchRequest(BaseModel):
    file_ids: list[str] = Field(min_length=1, max_length=100)


class IngestJobBatchRequest(BaseModel):
    job_ids: list[str] = Field(min_length=1, max_length=100)


class BatchItemResult(BaseModel):
    id: str
    status: str
    message: str | None = None
    job_id: str | None = None


class BatchOperationResponse(BaseModel):
    succeeded: int
    failed: int
    skipped: int = 0
    items: list[BatchItemResult]


class AuditLogListResponse(BaseModel):
    items: list[dict[str, Any]]


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    knowledge_base_id: str = Field(min_length=1)


class ApiKeyResponse(BaseModel):
    id: str
    org_id: str
    user_id: str
    knowledge_base_id: str
    name: str
    key_prefix: str
    is_active: bool
    last_used_at: str | None = None
    created_at: str


class ApiKeyCreateResponse(ApiKeyResponse):
    secret: str


class ApiKeyListResponse(BaseModel):
    items: list[ApiKeyResponse]


class DocumentPreviewResponse(BaseModel):
    file: FileRecordResponse
    pages: list[dict[str, Any]]
    chunks: list[dict[str, Any]]


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)


class RetrievalResponse(BaseModel):
    items: list[Source]


class OpenAIChatMessage(BaseModel):
    role: str
    content: str


class OpenAIChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[OpenAIChatMessage] = Field(min_length=1)
    temperature: float | None = None
    stream: bool = False
    top_k: int | None = Field(default=None, ge=1, le=50)
    conversation_id: str | None = None


@dataclass
class RagState:
    settings: AppSettings
    chat_model: Any | None = None
    embeddings: Any | None = None
    vector_store: Any | None = None
    reranker: Reranker | None = None
    init_errors: list[str] | None = None

    @property
    def ready(self) -> bool:
        return (
            not self.init_errors
            and self.chat_model is not None
            and self.vector_store is not None
            and self.reranker is not None
        )


def create_app() -> FastAPI:
    app = FastAPI(
        title="LangChain RAG Agent",
        version="0.1.0",
        description="FastAPI RAG service with OpenAI-compatible chat, local embeddings/rerank, and pgvector.",
    )

    @app.on_event("startup")
    async def startup() -> None:
        app.state.rag = await initialize_state_async()

    @app.get("/health")
    def health() -> dict[str, Any]:
        state = get_state(app)
        database_ok = False
        database_error = None
        try:
            check_database(state.settings)
            database_ok = True
        except Exception as exc:  # noqa: BLE001
            database_error = str(exc)

        errors = list(state.init_errors or [])
        if database_error:
            errors.append(f"database check failed: {database_error}")
        return {
            "status": "ok" if not errors else "error",
            "ready": state.ready and database_ok,
            "errors": errors,
            "config": {
                "app_env": state.settings.app_env,
                "chat_model": state.settings.openai_model,
                "embedding_provider": state.settings.embedding_provider,
                "embedding_dimension": state.settings.embedding_dimension,
                "embedding_model": state.settings.local_embedding_model_path,
                "rerank_provider": state.settings.rerank_provider,
                "rerank_model": state.settings.local_rerank_model_path,
                "pgvector_table": state.settings.pgvector_table,
                "rag_file_table": state.settings.rag_file_table,
                "rag_chunk_table": state.settings.rag_chunk_table,
                "rag_conversation_table": state.settings.rag_conversation_table,
                "rag_message_table": state.settings.rag_message_table,
                "rag_top_k": state.settings.rag_top_k,
                "rerank_top_n": state.settings.rerank_top_n,
                "chat_history_limit": state.settings.chat_history_limit,
                "openai_timeout_seconds": state.settings.openai_timeout_seconds,
                "retrieval_timeout_seconds": state.settings.retrieval_timeout_seconds,
                "rerank_timeout_seconds": state.settings.rerank_timeout_seconds,
            },
        }

    @app.post("/auth/login", response_model=TokenResponse)
    async def login(request: Request, payload: LoginRequest) -> TokenResponse:
        settings = get_state(app).settings
        user = await authenticate_user(settings, payload.email, payload.password)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        token = create_access_token(settings, user)
        await add_audit_log(
            settings,
            org_id=user.org_id,
            actor_user_id=user.id,
            actor_department_id=user.department_id,
            action="auth.login",
            target_type="user",
            target_id=user.id,
            **audit_context(request),
        )
        return TokenResponse(access_token=token, user=user_response(user).model_dump())

    @app.post("/auth/logout")
    async def logout(
        request: Request,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> dict[str, str]:
        settings = get_state(app).settings
        await add_audit_log(
            settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="auth.logout",
            target_type="user",
            target_id=current_user.id,
            **audit_context(request),
        )
        return {"status": "ok"}

    @app.get("/auth/me", response_model=UserResponse)
    async def me(current_user: CurrentUser = Depends(require_current_user)) -> UserResponse:
        return UserResponse(
            id=current_user.id,
            org_id=current_user.org_id,
            department_id=current_user.department_id,
            email=current_user.email,
            display_name=current_user.display_name,
            role=current_user.role,
            is_active=True,
            last_login_at=None,
            created_at="",
        )

    @app.get("/departments")
    async def departments(current_user: CurrentUser = Depends(require_current_user)) -> dict[str, list[DepartmentResponse]]:
        settings = get_state(app).settings
        return {
            "items": [
                department_response(item)
                for item in await list_departments(settings, current_user.org_id)
            ]
        }

    @app.post("/departments", response_model=DepartmentResponse)
    async def create_department_endpoint(
        http_request: Request,
        request: DepartmentCreateRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> DepartmentResponse:
        require_manager(current_user)
        settings = get_state(app).settings
        department = await create_department(
            settings,
            org_id=current_user.org_id,
            name=request.name,
            parent_id=request.parent_id,
        )
        await add_audit_log(
            settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="department.create",
            target_type="department",
            target_id=department.id,
            metadata={"name": department.name, "parent_id": department.parent_id},
            **audit_context(http_request),
        )
        return department_response(department)

    @app.get("/users")
    async def users(current_user: CurrentUser = Depends(require_current_user)) -> dict[str, list[UserResponse]]:
        require_manager(current_user)
        settings = get_state(app).settings
        return {"items": [user_response(item) for item in await list_users(settings, current_user.org_id)]}

    @app.post("/users", response_model=UserResponse)
    async def create_user_endpoint(
        http_request: Request,
        request: UserCreateRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> UserResponse:
        require_manager(current_user)
        if request.role in {"admin", "manager"} and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Only admins can create admin or manager users")
        settings = get_state(app).settings
        user = await create_user(
            settings,
            org_id=current_user.org_id,
            department_id=request.department_id,
            email=request.email,
            display_name=request.display_name,
            password=request.password,
            role=request.role,
        )
        await add_audit_log(
            settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="user.create",
            target_type="user",
            target_id=user.id,
            metadata={"role": user.role, "email": user.email, "department_id": user.department_id},
            **audit_context(http_request),
        )
        return user_response(user)

    @app.patch("/users/{user_id}", response_model=UserResponse)
    async def update_user_endpoint(
        http_request: Request,
        user_id: str,
        request: UserUpdateRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> UserResponse:
        require_manager(current_user)
        if request.role in {"admin", "manager"} and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Only admins can grant admin or manager role")
        existing_user = await get_user_by_id(get_state(app).settings, user_id)
        if existing_user and existing_user.role in {"admin", "manager"} and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Only admins can manage admin or manager users")
        if user_id == current_user.id and not request.is_active:
            raise HTTPException(status_code=400, detail="You cannot deactivate yourself")
        settings = get_state(app).settings
        user = await update_user(
            settings,
            org_id=current_user.org_id,
            user_id=user_id,
            display_name=request.display_name,
            role=request.role,
            department_id=request.department_id,
            is_active=request.is_active,
        )
        await add_audit_log(
            settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="user.update",
            target_type="user",
            target_id=user.id,
            metadata={
                "role": user.role,
                "department_id": user.department_id,
                "is_active": user.is_active,
            },
            **audit_context(http_request),
        )
        return user_response(user)

    @app.post("/users/{user_id}/reset-password", response_model=UserResponse)
    async def reset_user_password_endpoint(
        http_request: Request,
        user_id: str,
        request: PasswordResetRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> UserResponse:
        require_manager(current_user)
        existing_user = await get_user_by_id(get_state(app).settings, user_id)
        if existing_user and existing_user.role in {"admin", "manager"} and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Only admins can manage admin or manager users")
        settings = get_state(app).settings
        user = await reset_user_password(
            settings,
            org_id=current_user.org_id,
            user_id=user_id,
            password=request.password,
        )
        await add_audit_log(
            settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="user.reset_password",
            target_type="user",
            target_id=user.id,
            **audit_context(http_request),
        )
        return user_response(user)

    @app.delete("/users/{user_id}", response_model=UserResponse)
    async def deactivate_user_endpoint(
        http_request: Request,
        user_id: str,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> UserResponse:
        require_manager(current_user)
        if user_id == current_user.id:
            raise HTTPException(status_code=400, detail="You cannot deactivate yourself")
        existing_user = await get_user_by_id(get_state(app).settings, user_id)
        if existing_user and existing_user.role in {"admin", "manager"} and not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Only admins can manage admin or manager users")
        settings = get_state(app).settings
        user = await deactivate_user(settings, org_id=current_user.org_id, user_id=user_id)
        await add_audit_log(
            settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="user.deactivate",
            target_type="user",
            target_id=user.id,
            **audit_context(http_request),
        )
        return user_response(user)

    @app.get("/knowledge-bases", response_model=KnowledgeBaseListResponse)
    async def knowledge_bases(current_user: CurrentUser = Depends(require_current_user)) -> KnowledgeBaseListResponse:
        settings = get_state(app).settings
        knowledge_bases = await list_knowledge_bases(settings, current_user)
        return KnowledgeBaseListResponse(
            items=[
                kb_response(
                    item,
                    await get_knowledge_base_stats(settings, item.id),
                    await get_knowledge_base_capabilities(settings, item.id, current_user),
                )
                for item in knowledge_bases
            ]
        )

    @app.post("/knowledge-bases", response_model=KnowledgeBaseResponse)
    async def create_knowledge_base_endpoint(
        http_request: Request,
        request: KnowledgeBaseCreateRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> KnowledgeBaseResponse:
        require_manager(current_user)
        settings = get_state(app).settings
        kb = await create_knowledge_base(
            settings,
            user=current_user,
            name=request.name,
            description=request.description,
            visibility=request.visibility,
            department_ids=request.department_ids or ([current_user.department_id] if current_user.department_id else []),
        )
        await add_audit_log(
            settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="knowledge_base.create",
            target_type="knowledge_base",
            target_id=kb.id,
            metadata={"name": kb.name, "visibility": kb.visibility, "department_ids": kb.department_ids},
            **audit_context(http_request),
        )
        return kb_response(
            kb,
            await get_knowledge_base_stats(settings, kb.id),
            await get_knowledge_base_capabilities(settings, kb.id, current_user),
        )

    @app.patch("/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
    async def update_knowledge_base_endpoint(
        http_request: Request,
        knowledge_base_id: str,
        request: KnowledgeBaseUpdateRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> KnowledgeBaseResponse:
        settings = get_state(app).settings
        await require_kb_settings_admin(settings, knowledge_base_id, current_user)
        kb = await update_knowledge_base(
            settings,
            kb_id=knowledge_base_id,
            user=current_user,
            name=request.name,
            description=request.description,
            visibility=request.visibility,
            department_ids=request.department_ids or ([current_user.department_id] if current_user.department_id else []),
        )
        await add_audit_log(
            settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="knowledge_base.update",
            target_type="knowledge_base",
            target_id=kb.id,
            metadata={"name": kb.name, "visibility": kb.visibility, "department_ids": kb.department_ids},
            **audit_context(http_request),
        )
        return kb_response(
            kb,
            await get_knowledge_base_stats(settings, kb.id),
            await get_knowledge_base_capabilities(settings, kb.id, current_user),
        )

    @app.delete("/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
    async def delete_knowledge_base_endpoint(
        http_request: Request,
        knowledge_base_id: str,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> KnowledgeBaseResponse:
        settings = get_state(app).settings
        await require_kb_settings_admin(settings, knowledge_base_id, current_user)
        kb = await soft_delete_knowledge_base(settings, kb_id=knowledge_base_id, user=current_user)
        await add_audit_log(
            settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="knowledge_base.delete",
            target_type="knowledge_base",
            target_id=kb.id,
            metadata={"name": kb.name},
            **audit_context(http_request),
        )
        return kb_response(kb, await get_knowledge_base_stats(settings, kb.id))

    @app.get("/knowledge-bases/{knowledge_base_id}/stats")
    async def knowledge_base_stats_endpoint(
        knowledge_base_id: str,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> dict[str, int]:
        settings = get_state(app).settings
        await require_knowledge_base_access(settings, knowledge_base_id, current_user)
        stats = await get_knowledge_base_stats(settings, knowledge_base_id)
        return {
            "file_count": stats.file_count,
            "completed_file_count": stats.completed_file_count,
            "failed_job_count": stats.failed_job_count,
        }

    @app.get("/knowledge-bases/{knowledge_base_id}/members", response_model=KnowledgeBaseMemberListResponse)
    async def knowledge_base_members_endpoint(
        knowledge_base_id: str,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> KnowledgeBaseMemberListResponse:
        settings = get_state(app).settings
        await require_knowledge_base_access(settings, knowledge_base_id, current_user)
        await require_kb_member_admin(settings, knowledge_base_id, current_user)
        return KnowledgeBaseMemberListResponse(
            items=[
                kb_member_response(item)
                for item in await list_knowledge_base_members(settings, knowledge_base_id)
            ]
        )

    @app.get("/knowledge-bases/{knowledge_base_id}/member-candidates")
    async def knowledge_base_member_candidates_endpoint(
        knowledge_base_id: str,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> dict[str, list[UserResponse]]:
        settings = get_state(app).settings
        await require_kb_member_admin(settings, knowledge_base_id, current_user)
        return {"items": [user_response(item) for item in await list_users(settings, current_user.org_id) if item.is_active]}

    @app.put("/knowledge-bases/{knowledge_base_id}/members", response_model=KnowledgeBaseMemberResponse)
    async def upsert_knowledge_base_member_endpoint(
        http_request: Request,
        knowledge_base_id: str,
        request: KnowledgeBaseMemberUpsertRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> KnowledgeBaseMemberResponse:
        settings = get_state(app).settings
        await require_knowledge_base_access(settings, knowledge_base_id, current_user, write=True)
        await require_kb_member_admin(settings, knowledge_base_id, current_user)
        member = await upsert_knowledge_base_member(
            settings,
            kb_id=knowledge_base_id,
            user_id=request.user_id,
            role=request.role,
        )
        revoked_key_count = 0
        if request.role != "owner":
            revoked_key_count = await revoke_api_keys_for_kb_user(settings, kb_id=knowledge_base_id, user_id=request.user_id)
        await add_audit_log(
            settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="knowledge_base.member_upsert",
            target_type="knowledge_base",
            target_id=knowledge_base_id,
            metadata={"user_id": request.user_id, "role": request.role, "revoked_api_key_count": revoked_key_count},
            **audit_context(http_request),
        )
        return kb_member_response(member)

    @app.delete("/knowledge-bases/{knowledge_base_id}/members/{member_user_id}")
    async def delete_knowledge_base_member_endpoint(
        http_request: Request,
        knowledge_base_id: str,
        member_user_id: str,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> dict[str, str]:
        settings = get_state(app).settings
        await require_knowledge_base_access(settings, knowledge_base_id, current_user, write=True)
        await require_kb_member_admin(settings, knowledge_base_id, current_user)
        revoked_key_count = await revoke_api_keys_for_kb_user(settings, kb_id=knowledge_base_id, user_id=member_user_id)
        await remove_knowledge_base_member(settings, kb_id=knowledge_base_id, user_id=member_user_id)
        await add_audit_log(
            settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="knowledge_base.member_remove",
            target_type="knowledge_base",
            target_id=knowledge_base_id,
            metadata={"user_id": member_user_id, "revoked_api_key_count": revoked_key_count},
            **audit_context(http_request),
        )
        return {"status": "ok"}

    @app.get("/audit-logs", response_model=AuditLogListResponse)
    async def audit_logs(
        limit: int = Query(default=100, ge=1, le=500),
        current_user: CurrentUser = Depends(require_current_user),
    ) -> AuditLogListResponse:
        require_manager(current_user)
        settings = get_state(app).settings
        knowledge_base_ids = None
        if not current_user.is_admin:
            knowledge_base_ids = [item.id for item in await list_knowledge_bases(settings, current_user)]
        return AuditLogListResponse(
            items=await list_audit_logs(
                settings,
                current_user.org_id,
                limit=limit,
                knowledge_base_ids=knowledge_base_ids,
            )
        )

    @app.get("/api-keys", response_model=ApiKeyListResponse)
    async def api_keys(current_user: CurrentUser = Depends(require_current_user)) -> ApiKeyListResponse:
        settings = get_state(app).settings
        manageable_kb_ids = [
            kb.id
            for kb in await list_knowledge_bases(settings, current_user)
            if (await get_knowledge_base_capabilities(settings, kb.id, current_user)).can_manage_api_keys
        ]
        return ApiKeyListResponse(
            items=[
                api_key_response(item)
                for item in await list_api_keys(settings, current_user.org_id, knowledge_base_ids=manageable_kb_ids)
            ]
        )

    @app.post("/api-keys", response_model=ApiKeyCreateResponse)
    async def create_api_key_endpoint(
        http_request: Request,
        request: ApiKeyCreateRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> ApiKeyCreateResponse:
        state = get_state(app)
        await require_kb_api_key_admin(state.settings, request.knowledge_base_id, current_user)
        created = await create_api_key(
            state.settings,
            org_id=current_user.org_id,
            user_id=current_user.id,
            knowledge_base_id=request.knowledge_base_id,
            name=request.name,
        )
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="api_key.create",
            target_type="api_key",
            target_id=created.record.id,
            metadata={"name": created.record.name, "knowledge_base_id": request.knowledge_base_id},
            **audit_context(http_request),
        )
        return ApiKeyCreateResponse(**api_key_response(created.record).model_dump(), secret=created.secret)

    @app.delete("/api-keys/{api_key_id}", response_model=ApiKeyResponse)
    async def delete_api_key_endpoint(
        http_request: Request,
        api_key_id: str,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> ApiKeyResponse:
        state = get_state(app)
        existing = await get_api_key(state.settings, current_user.org_id, api_key_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="API key not found")
        await require_kb_api_key_admin(state.settings, existing.knowledge_base_id, current_user)
        revoked = await revoke_api_key(state.settings, current_user.org_id, api_key_id)
        if revoked is None:
            raise HTTPException(status_code=404, detail="API key not found")
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="api_key.revoke",
            target_type="api_key",
            target_id=revoked.id,
            metadata={"name": revoked.name, "knowledge_base_id": revoked.knowledge_base_id},
            **audit_context(http_request),
        )
        return api_key_response(revoked)

    @app.post("/documents", response_model=DocumentUploadResponse)
    async def upload_document(
        user_id: str = Form(..., min_length=1),
        file: UploadFile = File(...),
    ) -> DocumentUploadResponse:
        raise HTTPException(status_code=410, detail="Use /knowledge-bases/{knowledge_base_id}/documents with bearer authentication")
        state = require_ready(app)
        filename = file.filename or "uploaded"
        if not any(filename.lower().endswith(extension) for extension in SUPPORTED_EXTENSIONS):
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise HTTPException(status_code=400, detail=f"Unsupported document type. Supported: {supported}")
        data = await file.read()
        content_type = file.content_type or "application/octet-stream"
        file_sha256 = hash_bytes(data)
        file_id = str(uuid4())
        try:
            text = extract_text(filename, data)
            content_sha256 = hash_text(text)
            existing_file = await find_file_by_content_hash(state.settings, user_id, content_sha256)
            if existing_file is not None:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "Document content already exists",
                        "file": file_record_response(existing_file).model_dump(),
                    },
                )
            await create_processing_file(
                state.settings,
                file_id=file_id,
                user_id=user_id,
                filename=filename,
                content_type=content_type,
                file_size=len(data),
                file_sha256=file_sha256,
                content_sha256=content_sha256,
            )
            document_id, documents = build_chunks_from_text(
                filename=filename,
                content_type=content_type,
                text=text,
                settings=state.settings,
                document_id=file_id,
                content_sha256=content_sha256,
                user_id=user_id,
            )
            ids, indexed_documents = add_documents(state.vector_store, documents)
            await save_file_chunks(
                state.settings,
                user_id=user_id,
                file_id=file_id,
                chunks=[
                    (
                        ids[index],
                        int(document.metadata["chunk_index"]),
                        document.page_content,
                        hash_text(document.page_content),
                    )
                    for index, document in enumerate(indexed_documents)
                ],
            )
            await mark_file_completed(
                state.settings,
                file_id,
                chunk_count=len(documents),
                chunk_ids=ids,
                vector_ids=ids,
            )
        except ValueError as exc:
            await mark_file_failed(state.settings, file_id, str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except IntegrityError as exc:
            raise HTTPException(status_code=409, detail="Document content already exists") from exc
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            await mark_file_failed(state.settings, file_id, str(exc))
            raise HTTPException(status_code=500, detail=f"Document ingestion failed: {exc}") from exc
        return DocumentUploadResponse(
            file_id=file_id,
            document_id=document_id,
            user_id=user_id,
            filename=filename,
            content_sha256=content_sha256,
            chunks=len(documents),
            chunk_ids=ids,
            ids=ids,
        )

    @app.post("/knowledge-bases/{knowledge_base_id}/documents", response_model=IngestJobResponse)
    async def upload_knowledge_base_document(
        http_request: Request,
        knowledge_base_id: str,
        file: UploadFile = File(...),
        current_user: CurrentUser = Depends(require_current_user),
    ) -> IngestJobResponse:
        state = get_state(app)
        await require_knowledge_base_access(state.settings, knowledge_base_id, current_user, write=True)
        filename = file.filename or "uploaded"
        if not any(filename.lower().endswith(extension) for extension in SUPPORTED_EXTENSIONS):
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise HTTPException(status_code=400, detail=f"Unsupported document type. Supported: {supported}")
        data = await file.read()
        storage_dir = Path(state.settings.upload_storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        job_file_id = str(uuid4())
        safe_name = filename.replace("/", "_").replace("\\", "_")
        storage_path = storage_dir / f"{job_file_id}_{safe_name}"
        storage_path.write_bytes(data)
        job = await create_ingest_job(
            state.settings,
            org_id=current_user.org_id,
            knowledge_base_id=knowledge_base_id,
            created_by_user_id=current_user.id,
            source_type="file",
            source_uri=str(storage_path),
            filename=filename,
            payload={
                "path": str(storage_path),
                "content_type": file.content_type or "application/octet-stream",
                "file_size": len(data),
            },
        )
        enqueue_ingest_job(state.settings, job.id)
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="document.ingest_requested",
            target_type="ingest_job",
            target_id=job.id,
            metadata={"knowledge_base_id": knowledge_base_id, "filename": filename},
            **audit_context(http_request),
        )
        return ingest_job_response(job)

    @app.post("/knowledge-bases/{knowledge_base_id}/urls", response_model=IngestJobResponse)
    async def ingest_knowledge_base_url(
        http_request: Request,
        knowledge_base_id: str,
        request: UrlIngestRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> IngestJobResponse:
        state = get_state(app)
        await require_knowledge_base_access(state.settings, knowledge_base_id, current_user, write=True)
        job = await create_ingest_job(
            state.settings,
            org_id=current_user.org_id,
            knowledge_base_id=knowledge_base_id,
            created_by_user_id=current_user.id,
            source_type="url",
            source_uri=request.url,
            filename=request.filename,
            payload={"url": request.url},
        )
        enqueue_ingest_job(state.settings, job.id)
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="url.ingest_requested",
            target_type="ingest_job",
            target_id=job.id,
            metadata={"knowledge_base_id": knowledge_base_id, "url": request.url},
            **audit_context(http_request),
        )
        return ingest_job_response(job)

    @app.post("/knowledge-bases/{knowledge_base_id}/urls/plan", response_model=UrlImportPlanResponse)
    async def plan_knowledge_base_urls(
        http_request: Request,
        knowledge_base_id: str,
        request: UrlBatchPlanRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> UrlImportPlanResponse:
        state = get_state(app)
        await require_knowledge_base_access(state.settings, knowledge_base_id, current_user, write=True)
        plan = await build_url_import_plan(
            state.settings,
            org_id=current_user.org_id,
            knowledge_base_id=knowledge_base_id,
            user_id=current_user.id,
            urls=request.urls,
            skip_duplicates=request.skip_duplicates,
        )
        save_import_plan(state.settings, plan.plan_id, plan_cache_payload(plan, current_user))
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="batch_import.dry_run",
            target_type="knowledge_base",
            target_id=knowledge_base_id,
            metadata={"plan_id": plan.plan_id, "total": plan.total, "ready": plan.ready_count, "blocked": plan.blocked_count},
            **audit_context(http_request),
        )
        return plan

    @app.post("/knowledge-bases/{knowledge_base_id}/urls/batch", response_model=BatchOperationResponse)
    async def batch_ingest_knowledge_base_urls(
        http_request: Request,
        knowledge_base_id: str,
        request: UrlBatchCommitRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> BatchOperationResponse:
        state = get_state(app)
        await require_knowledge_base_access(state.settings, knowledge_base_id, current_user, write=True)
        lock_token = str(uuid4())
        if not acquire_import_plan_lock(state.settings, request.plan_id, lock_token):
            raise HTTPException(status_code=409, detail="Import plan is being confirmed")
        try:
            payload = load_import_plan(state.settings, request.plan_id)
            if payload is None:
                raise HTTPException(status_code=404, detail="Import plan not found or expired")
            if (
                payload.get("knowledge_base_id") != knowledge_base_id
                or payload.get("user_id") != current_user.id
                or payload.get("org_id") != current_user.org_id
            ):
                raise HTTPException(status_code=403, detail="Import plan does not belong to this user and knowledge base")
            expires_at = parse_datetime(str(payload.get("expires_at") or ""))
            if expires_at and expires_at < datetime.now(timezone.utc):
                raise HTTPException(status_code=410, detail="Import plan has expired")
            selected = set(request.client_item_ids)
            items = [UrlImportPlanItem(**item) for item in payload.get("items", [])]
            item_by_id = {item.client_item_id: item for item in items}
            results: list[BatchItemResult] = []
            for client_item_id in request.client_item_ids:
                item = item_by_id.get(client_item_id)
                if item is None:
                    results.append(BatchItemResult(id=client_item_id, status="failed", message="Import plan item not found"))
                    continue
                if item.job_id:
                    results.append(BatchItemResult(id=client_item_id, status="succeeded", job_id=item.job_id, message="Already enqueued"))
                    continue
                if not item.can_enqueue:
                    results.append(BatchItemResult(id=client_item_id, status="skipped", message=item.reason or item.status))
                    continue
                try:
                    job = await create_ingest_job(
                        state.settings,
                        org_id=current_user.org_id,
                        knowledge_base_id=knowledge_base_id,
                        created_by_user_id=current_user.id,
                        source_type="url",
                        source_uri=item.url,
                        filename=item.filename,
                        payload={
                            "url": item.url,
                            "planned_content_sha256": item.content_sha256,
                            "estimated_chunks": item.estimated_chunks,
                        },
                    )
                    try:
                        enqueue_ingest_job(state.settings, job.id)
                    except Exception as exc:  # noqa: BLE001
                        await mark_job_failed(state.settings, job.id, f"Queue enqueue failed: {exc}")
                        raise
                    item.job_id = job.id
                    item.confirmed_at = datetime.now(timezone.utc).isoformat()
                    item_by_id[client_item_id] = item
                    results.append(BatchItemResult(id=client_item_id, status="succeeded", job_id=job.id))
                except Exception as exc:  # noqa: BLE001
                    results.append(BatchItemResult(id=client_item_id, status="failed", message=str(exc)))
            payload["items"] = [item_by_id.get(item.client_item_id, item).model_dump() for item in items]
            refresh_import_plan(state.settings, request.plan_id, payload)
        finally:
            release_import_plan_lock(state.settings, request.plan_id, lock_token)
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="batch_import.confirmed",
            target_type="ingest_job",
            metadata={
                "knowledge_base_id": knowledge_base_id,
                "plan_id": request.plan_id,
                "selected": len(selected),
                "succeeded": sum(1 for item in results if item.status == "succeeded"),
                "failed": sum(1 for item in results if item.status == "failed"),
                "skipped": sum(1 for item in results if item.status == "skipped"),
            },
            **audit_context(http_request),
        )
        return batch_response(results)

    @app.get("/knowledge-bases/{knowledge_base_id}/ingest-jobs", response_model=IngestJobListResponse)
    async def knowledge_base_ingest_jobs(
        knowledge_base_id: str,
        limit: int = Query(default=100, ge=1, le=500),
        status: str = Query(default="all", pattern="^(active|history|all|pending|running|succeeded|failed|cancelled)$"),
        current_user: CurrentUser = Depends(require_current_user),
    ) -> IngestJobListResponse:
        state = get_state(app)
        await require_knowledge_base_access(state.settings, knowledge_base_id, current_user)
        return IngestJobListResponse(
            items=[
                ingest_job_response(item)
                for item in await list_ingest_jobs(state.settings, knowledge_base_id, limit=limit, status=status)
            ]
        )

    @app.get("/knowledge-bases/{knowledge_base_id}/queue-health", response_model=QueueHealthResponse)
    async def knowledge_base_queue_health(
        knowledge_base_id: str,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> QueueHealthResponse:
        state = get_state(app)
        await require_knowledge_base_access(state.settings, knowledge_base_id, current_user)
        redis_queue_length = -1
        worker_last_seen_at = None
        try:
            redis_queue_length = get_ingest_queue_length(state.settings)
            worker_last_seen_at = get_worker_heartbeat(state.settings)
        except Exception:  # noqa: BLE001
            pass
        return queue_health_response(
            await get_ingest_queue_health(state.settings, knowledge_base_id),
            redis_queue_length=redis_queue_length,
            worker_last_seen_at=worker_last_seen_at,
        )

    @app.get("/ingest-jobs/{job_id}", response_model=IngestJobResponse)
    async def ingest_job_detail(
        job_id: str,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> IngestJobResponse:
        state = get_state(app)
        job = await get_ingest_job(state.settings, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Ingest job not found")
        await require_knowledge_base_access(state.settings, job.knowledge_base_id, current_user)
        return ingest_job_response(job)

    @app.post("/ingest-jobs/{job_id}/retry", response_model=IngestJobResponse)
    async def retry_ingest_job_endpoint(
        http_request: Request,
        job_id: str,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> IngestJobResponse:
        state = get_state(app)
        job = await get_ingest_job(state.settings, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Ingest job not found")
        await require_knowledge_base_access(state.settings, job.knowledge_base_id, current_user, write=True)
        try:
            retried = await retry_failed_ingest_job(state.settings, job_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        enqueue_ingest_job(state.settings, job_id)
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="ingest.retry_requested",
            target_type="ingest_job",
            target_id=job_id,
            metadata={"knowledge_base_id": job.knowledge_base_id, "retry_count": retried.retry_count},
            **audit_context(http_request),
        )
        return ingest_job_response(retried)

    @app.post("/ingest-jobs/{job_id}/cancel", response_model=IngestJobResponse)
    async def cancel_ingest_job_endpoint(
        http_request: Request,
        job_id: str,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> IngestJobResponse:
        state = get_state(app)
        job = await get_ingest_job(state.settings, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Ingest job not found")
        await require_knowledge_base_access(state.settings, job.knowledge_base_id, current_user, write=True)
        try:
            cancelled = await cancel_ingest_job(state.settings, job_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="ingest.cancel_requested",
            target_type="ingest_job",
            target_id=job_id,
            metadata={"knowledge_base_id": job.knowledge_base_id, "status": job.status},
            **audit_context(http_request),
        )
        return ingest_job_response(cancelled)

    @app.post("/ingest-jobs/actions/batch-retry", response_model=BatchOperationResponse)
    async def batch_retry_ingest_jobs(
        http_request: Request,
        request: IngestJobBatchRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> BatchOperationResponse:
        state = get_state(app)
        results: list[BatchItemResult] = []
        for job_id in request.job_ids:
            try:
                job = await get_ingest_job(state.settings, job_id)
                if job is None:
                    raise ValueError("Ingest job not found")
                await require_knowledge_base_access(state.settings, job.knowledge_base_id, current_user, write=True)
                retried = await retry_failed_ingest_job(state.settings, job_id)
                enqueue_ingest_job(state.settings, job_id)
                results.append(BatchItemResult(id=job_id, status="succeeded", job_id=retried.id))
            except Exception as exc:  # noqa: BLE001
                results.append(BatchItemResult(id=job_id, status="failed", message=str(exc)))
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="ingest.batch_retry_requested",
            target_type="ingest_job",
            metadata={"job_count": len(request.job_ids), "failed": sum(1 for item in results if item.status == "failed")},
            **audit_context(http_request),
        )
        return batch_response(results)

    @app.post("/ingest-jobs/actions/batch-cancel", response_model=BatchOperationResponse)
    async def batch_cancel_ingest_jobs(
        http_request: Request,
        request: IngestJobBatchRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> BatchOperationResponse:
        state = get_state(app)
        results: list[BatchItemResult] = []
        for job_id in request.job_ids:
            try:
                job = await get_ingest_job(state.settings, job_id)
                if job is None:
                    raise ValueError("Ingest job not found")
                await require_knowledge_base_access(state.settings, job.knowledge_base_id, current_user, write=True)
                cancelled = await cancel_ingest_job(state.settings, job_id)
                results.append(BatchItemResult(id=job_id, status="succeeded", job_id=cancelled.id))
            except Exception as exc:  # noqa: BLE001
                results.append(BatchItemResult(id=job_id, status="failed", message=str(exc)))
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="ingest.batch_cancel_requested",
            target_type="ingest_job",
            metadata={"job_count": len(request.job_ids), "failed": sum(1 for item in results if item.status == "failed")},
            **audit_context(http_request),
        )
        return batch_response(results)

    @app.get("/knowledge-bases/{knowledge_base_id}/documents", response_model=FileListResponse)
    async def knowledge_base_documents(
        knowledge_base_id: str,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        status: str | None = Query(default=None, pattern="^(processing|completed|failed|deleted)$"),
        include_deleted: bool = Query(default=False),
        current_user: CurrentUser = Depends(require_current_user),
    ) -> FileListResponse:
        state = get_state(app)
        await require_knowledge_base_access(state.settings, knowledge_base_id, current_user)
        return FileListResponse(
            items=[
                file_record_response(item)
                for item in await list_knowledge_base_files(
                    state.settings,
                    knowledge_base_id,
                    limit=limit,
                    offset=offset,
                    status=status,
                    include_deleted=include_deleted,
                )
            ]
        )

    @app.get("/knowledge-bases/{knowledge_base_id}/documents/{file_id}/preview", response_model=DocumentPreviewResponse)
    async def knowledge_base_document_preview(
        knowledge_base_id: str,
        file_id: str,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> DocumentPreviewResponse:
        state = get_state(app)
        await require_knowledge_base_access(state.settings, knowledge_base_id, current_user)
        item = await get_knowledge_base_file(state.settings, knowledge_base_id, file_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DocumentPreviewResponse(
            file=file_record_response(item),
            pages=await list_document_pages(state.settings, file_id),
            chunks=await list_document_chunks(state.settings, file_id),
        )

    @app.get("/knowledge-bases/{knowledge_base_id}/documents/{file_id}/raw")
    async def knowledge_base_document_raw(
        knowledge_base_id: str,
        file_id: str,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> FileResponse:
        state = get_state(app)
        await require_knowledge_base_access(state.settings, knowledge_base_id, current_user)
        item = await get_knowledge_base_file(state.settings, knowledge_base_id, file_id)
        if item is None or not item.source_uri:
            raise HTTPException(status_code=404, detail="Document not found")
        path = safe_upload_path(state.settings.upload_storage_dir, item.source_uri)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail="Stored document file not found")
        return FileResponse(path, media_type=item.content_type, filename=item.filename)

    @app.post("/knowledge-bases/{knowledge_base_id}/documents/{file_id}/reindex", response_model=IngestJobResponse)
    async def reindex_knowledge_base_document(
        http_request: Request,
        knowledge_base_id: str,
        file_id: str,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> IngestJobResponse:
        state = get_state(app)
        await require_knowledge_base_access(state.settings, knowledge_base_id, current_user, write=True)
        item = await get_knowledge_base_file(state.settings, knowledge_base_id, file_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Document not found")
        try:
            payload, source_uri = build_reindex_payload(state.settings, item)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        job = await create_ingest_job(
            state.settings,
            org_id=current_user.org_id,
            knowledge_base_id=knowledge_base_id,
            created_by_user_id=current_user.id,
            source_type=item.source_type,
            source_uri=source_uri,
            filename=item.filename,
            payload=payload,
        )
        enqueue_ingest_job(state.settings, job.id)
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="document.reindex_requested",
            target_type="document",
            target_id=file_id,
            metadata={"knowledge_base_id": knowledge_base_id, "job_id": job.id, "filename": item.filename},
            **audit_context(http_request),
        )
        return ingest_job_response(job)

    @app.delete("/knowledge-bases/{knowledge_base_id}/documents/{file_id}", response_model=DeleteDocumentResponse)
    async def delete_knowledge_base_document(
        http_request: Request,
        knowledge_base_id: str,
        file_id: str,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> DeleteDocumentResponse:
        state = require_ready(app)
        await require_knowledge_base_access(state.settings, knowledge_base_id, current_user, write=True)
        item = await get_knowledge_base_file(state.settings, knowledge_base_id, file_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Document not found")
        try:
            delete_documents(state.vector_store, item.vector_ids)
            await delete_file_chunks_by_file_id(state.settings, file_id)
            await delete_document_artifacts(state.settings, file_id)
            await mark_file_deleted(state.settings, file_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Document deletion failed: {exc}") from exc
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="document.delete",
            target_type="document",
            target_id=file_id,
            metadata={"knowledge_base_id": knowledge_base_id, "filename": item.filename, "chunks": item.chunk_count},
            **audit_context(http_request),
        )
        return DeleteDocumentResponse(file_id=file_id, deleted_vectors=len(item.vector_ids), deleted_chunks=len(item.chunk_ids))

    @app.post("/knowledge-bases/{knowledge_base_id}/documents/batch-delete", response_model=BatchOperationResponse)
    async def batch_delete_knowledge_base_documents(
        http_request: Request,
        knowledge_base_id: str,
        request: DocumentBatchRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> BatchOperationResponse:
        state = require_ready(app)
        await require_knowledge_base_access(state.settings, knowledge_base_id, current_user, write=True)
        results: list[BatchItemResult] = []
        for file_id in request.file_ids:
            try:
                item = await get_knowledge_base_file(state.settings, knowledge_base_id, file_id)
                if item is None:
                    raise ValueError("Document not found")
                delete_documents(state.vector_store, item.vector_ids)
                await delete_file_chunks_by_file_id(state.settings, file_id)
                await delete_document_artifacts(state.settings, file_id)
                await mark_file_deleted(state.settings, file_id)
                results.append(BatchItemResult(id=file_id, status="succeeded"))
            except Exception as exc:  # noqa: BLE001
                results.append(BatchItemResult(id=file_id, status="failed", message=str(exc)))
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="document.batch_delete",
            target_type="document",
            metadata={"knowledge_base_id": knowledge_base_id, "file_count": len(request.file_ids), "failed": sum(1 for item in results if item.status == "failed")},
            **audit_context(http_request),
        )
        return batch_response(results)

    @app.post("/knowledge-bases/{knowledge_base_id}/documents/batch-reindex", response_model=BatchOperationResponse)
    async def batch_reindex_knowledge_base_documents(
        http_request: Request,
        knowledge_base_id: str,
        request: DocumentBatchRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> BatchOperationResponse:
        state = get_state(app)
        await require_knowledge_base_access(state.settings, knowledge_base_id, current_user, write=True)
        results: list[BatchItemResult] = []
        for file_id in request.file_ids:
            try:
                item = await get_knowledge_base_file(state.settings, knowledge_base_id, file_id)
                if item is None:
                    raise ValueError("Document not found")
                payload, source_uri = build_reindex_payload(state.settings, item)
                job = await create_ingest_job(
                    state.settings,
                    org_id=current_user.org_id,
                    knowledge_base_id=knowledge_base_id,
                    created_by_user_id=current_user.id,
                    source_type=item.source_type,
                    source_uri=source_uri,
                    filename=item.filename,
                    payload=payload,
                )
                enqueue_ingest_job(state.settings, job.id)
                results.append(BatchItemResult(id=file_id, status="succeeded", job_id=job.id))
            except Exception as exc:  # noqa: BLE001
                results.append(BatchItemResult(id=file_id, status="failed", message=str(exc)))
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="document.batch_reindex_requested",
            target_type="document",
            metadata={"knowledge_base_id": knowledge_base_id, "file_count": len(request.file_ids), "failed": sum(1 for item in results if item.status == "failed")},
            **audit_context(http_request),
        )
        return batch_response(results)

    @app.get("/documents", response_model=FileListResponse)
    async def documents(
        user_id: str = Query(..., min_length=1),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> FileListResponse:
        raise HTTPException(status_code=410, detail="Use /knowledge-bases/{knowledge_base_id}/documents with bearer authentication")
        state = require_ready(app)
        return FileListResponse(
            items=[
                file_record_response(item)
                for item in await list_files(state.settings, user_id=user_id, limit=limit, offset=offset)
            ]
        )

    @app.get("/documents/{file_id}", response_model=FileRecordResponse)
    async def document_detail(file_id: str, user_id: str = Query(..., min_length=1)) -> FileRecordResponse:
        raise HTTPException(status_code=410, detail="Use /knowledge-bases/{knowledge_base_id}/documents/{file_id}/preview with bearer authentication")
        state = require_ready(app)
        item = await get_file(state.settings, user_id, file_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return file_record_response(item)

    @app.delete("/documents/{file_id}", response_model=DeleteDocumentResponse)
    async def delete_document(file_id: str, user_id: str = Query(..., min_length=1)) -> DeleteDocumentResponse:
        raise HTTPException(status_code=410, detail="Use /knowledge-bases/{knowledge_base_id}/documents/{file_id} with bearer authentication")
        state = require_ready(app)
        item = await get_file(state.settings, user_id, file_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Document not found")
        try:
            delete_documents(state.vector_store, item.vector_ids)
            await delete_file_chunks(state.settings, user_id, file_id)
            deleted = await delete_file_record(state.settings, user_id, file_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Document deletion failed: {exc}") from exc
        if deleted is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return DeleteDocumentResponse(
            file_id=file_id,
            deleted_vectors=len(item.vector_ids),
            deleted_chunks=len(item.chunk_ids),
        )

    @app.get("/conversations", response_model=ConversationListResponse)
    async def conversations(
        knowledge_base_id: str = Query(..., min_length=1),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        current_user: CurrentUser = Depends(require_current_user),
    ) -> ConversationListResponse:
        state = require_ready(app)
        await require_knowledge_base_access(state.settings, knowledge_base_id, current_user)
        return ConversationListResponse(
            items=[
                conversation_response(item)
                for item in await list_conversations(
                    state.settings,
                    user_id=current_user.id,
                    knowledge_base_id=knowledge_base_id,
                    limit=limit,
                    offset=offset,
                )
            ]
        )

    @app.get("/conversations/{conversation_id}/messages", response_model=ChatMessageListResponse)
    async def conversation_messages(
        conversation_id: UUID,
        knowledge_base_id: UUID = Query(...),
        limit: int = Query(default=200, ge=1, le=500),
        current_user: CurrentUser = Depends(require_current_user),
    ) -> ChatMessageListResponse:
        state = require_ready(app)
        conversation_id_text = str(conversation_id)
        knowledge_base_id_text = str(knowledge_base_id)
        await require_knowledge_base_access(state.settings, knowledge_base_id_text, current_user)
        conversation = await get_conversation(
            state.settings,
            conversation_id=conversation_id_text,
            user_id=current_user.id,
            knowledge_base_id=knowledge_base_id_text,
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return ChatMessageListResponse(
            items=[
                chat_message_response(item)
                for item in await list_conversation_messages(
                    state.settings,
                    conversation_id=conversation_id_text,
                    user_id=current_user.id,
                    knowledge_base_id=knowledge_base_id_text,
                    limit=limit,
                )
            ]
        )

    @app.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
    async def update_conversation_endpoint(
        http_request: Request,
        conversation_id: UUID,
        request: ConversationUpdateRequest,
        knowledge_base_id: UUID = Query(...),
        current_user: CurrentUser = Depends(require_current_user),
    ) -> ConversationResponse:
        state = require_ready(app)
        conversation_id_text = str(conversation_id)
        knowledge_base_id_text = str(knowledge_base_id)
        await require_knowledge_base_access(state.settings, knowledge_base_id_text, current_user)
        title = request.title.strip()
        if not title:
            raise HTTPException(status_code=422, detail="Conversation title is required")
        conversation = await update_conversation_title(
            state.settings,
            conversation_id=conversation_id_text,
            user_id=current_user.id,
            knowledge_base_id=knowledge_base_id_text,
            title=title,
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="conversation.update",
            target_type="conversation",
            target_id=conversation_id_text,
            metadata={"knowledge_base_id": knowledge_base_id_text, "title": conversation.title},
            **audit_context(http_request),
        )
        return conversation_response(conversation)

    @app.delete("/conversations/{conversation_id}", response_model=DeleteConversationResponse)
    async def delete_conversation_endpoint(
        http_request: Request,
        conversation_id: UUID,
        knowledge_base_id: UUID = Query(...),
        current_user: CurrentUser = Depends(require_current_user),
    ) -> DeleteConversationResponse:
        state = require_ready(app)
        conversation_id_text = str(conversation_id)
        knowledge_base_id_text = str(knowledge_base_id)
        await require_knowledge_base_access(state.settings, knowledge_base_id_text, current_user)
        deleted = await delete_conversation(
            state.settings,
            conversation_id=conversation_id_text,
            user_id=current_user.id,
            knowledge_base_id=knowledge_base_id_text,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="conversation.delete",
            target_type="conversation",
            target_id=conversation_id_text,
            metadata={"knowledge_base_id": knowledge_base_id_text},
            **audit_context(http_request),
        )
        return DeleteConversationResponse(conversation_id=conversation_id_text, deleted=True)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        http_request: Request,
        request: ChatRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> ChatResponse:
        state = require_ready(app)
        await require_knowledge_base_access(state.settings, request.knowledge_base_id, current_user)
        if await count_completed_knowledge_base_files(state.settings, request.knowledge_base_id) <= 0:
            raise HTTPException(
                status_code=409,
                detail={"message": "该知识库还没有完成入库的文档，请先上传并等待处理完成。", "stage": "documents"},
            )
        top_k = request.top_k or state.settings.rag_top_k
        rerank_top_n = request.rerank_top_n if request.rerank_top_n is not None else state.settings.rerank_top_n
        try:
            conversation_id = str(UUID(request.conversation_id)) if request.conversation_id else str(uuid4())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Invalid conversation_id") from exc
        started_at = time.perf_counter()
        try:
            try:
                await ensure_conversation(
                    state.settings,
                    conversation_id=conversation_id,
                    org_id=current_user.org_id,
                    knowledge_base_id=request.knowledge_base_id,
                    user_id=current_user.id,
                    title=request.message[:80],
                )
            except ValueError as exc:
                raise HTTPException(status_code=403, detail="Conversation is not available in this knowledge base") from exc
            history = await get_recent_messages(
                state.settings,
                conversation_id=conversation_id,
                user_id=current_user.id,
                limit=state.settings.chat_history_limit,
                knowledge_base_id=request.knowledge_base_id,
            )
            retrieval_query = await rewrite_query_async(
                state.chat_model,
                request.message,
                history,
                state.settings.openai_timeout_seconds,
            )
            user_message = await add_message(
                state.settings,
                message_id=str(uuid4()),
                conversation_id=conversation_id,
                org_id=current_user.org_id,
                knowledge_base_id=request.knowledge_base_id,
                user_id=current_user.id,
                role="user",
                content=request.message,
            )
            retrieval = await asyncio.wait_for(
                hybrid_search(
                    state.settings,
                    state.vector_store,
                    retrieval_query,
                    user_id=current_user.id,
                    knowledge_base_id=request.knowledge_base_id,
                    top_k=top_k,
                ),
                timeout=state.settings.retrieval_timeout_seconds,
            )
            if not retrieval.documents:
                ranked_documents = []
            else:
                ranked_documents = await rerank_or_original_async(
                    state.reranker,
                    retrieval_query,
                    retrieval.documents,
                    rerank_top_n,
                    state.settings.rerank_timeout_seconds,
                )
            selected = ranked_documents[: rerank_top_n or len(ranked_documents)]
            if selected:
                context = format_context([item.document for item in selected])
                prompt = build_prompt().invoke(
                    {
                        "question": request.message,
                        "context": context,
                        "history": format_history(history),
                    }
                )
                answer = (await invoke_chat_model(
                    state.chat_model,
                    prompt,
                    state.settings.openai_timeout_seconds,
                )).content
            else:
                answer = "没有在当前知识库中检索到相关内容。请确认文档已经完成入库，或换一种问法。"
            sources = [
                source_from_document(item.document, rerank_score=item.score)
                for item in selected
            ]
            confidence, confidence_score = answer_confidence(sources)
            assistant_message = await add_message(
                state.settings,
                message_id=str(uuid4()),
                conversation_id=conversation_id,
                org_id=current_user.org_id,
                knowledge_base_id=request.knowledge_base_id,
                user_id=current_user.id,
                role="assistant",
                content=str(answer),
                metadata={
                    "sources": [source.model_dump() for source in sources],
                    "confidence": confidence,
                    "confidence_score": confidence_score,
                },
            )
            await add_chat_log(
                state.settings,
                org_id=current_user.org_id,
                knowledge_base_id=request.knowledge_base_id,
                user_id=current_user.id,
                conversation_id=conversation_id,
                question=request.message,
                answer=str(answer),
                sources=[source.model_dump() for source in sources],
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
            await add_audit_log(
                state.settings,
                org_id=current_user.org_id,
                actor_user_id=current_user.id,
                action="chat.ask",
                target_type="knowledge_base",
                target_id=request.knowledge_base_id,
                metadata={
                    "conversation_id": conversation_id,
                    "source_count": len(sources),
                    "query": request.message,
                    "retrieval_query": retrieval_query,
                },
                actor_department_id=current_user.department_id,
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                **audit_context(http_request),
            )
        except HTTPException:
            raise
        except asyncio.TimeoutError as exc:
            await add_audit_log(
                state.settings,
                org_id=current_user.org_id,
                actor_user_id=current_user.id,
                actor_department_id=current_user.department_id,
                action="chat.ask",
                target_type="knowledge_base",
                target_id=request.knowledge_base_id,
                result="failed",
                error_message="Chat model request timed out",
                metadata={"conversation_id": conversation_id, "query": request.message},
                **audit_context(http_request),
            )
            raise HTTPException(
                status_code=504,
                detail={"message": "模型响应超时，请稍后重试或检查模型服务。", "stage": "chat"},
            ) from exc
        except Exception as exc:  # noqa: BLE001
            await add_audit_log(
                state.settings,
                org_id=current_user.org_id,
                actor_user_id=current_user.id,
                actor_department_id=current_user.department_id,
                action="chat.ask",
                target_type="knowledge_base",
                target_id=request.knowledge_base_id,
                result="failed",
                error_message=str(exc)[:1000],
                metadata={"conversation_id": conversation_id, "query": request.message},
                **audit_context(http_request),
            )
            raise HTTPException(status_code=500, detail={"message": f"问答失败：{exc}", "stage": "chat"}) from exc

        return ChatResponse(
            conversation_id=conversation_id,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            answer=str(answer),
            sources=sources,
            confidence=confidence,
            confidence_score=confidence_score,
        )

    @app.post("/feedback", response_model=FeedbackResponse)
    async def create_feedback_endpoint(
        http_request: Request,
        request: FeedbackRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> FeedbackResponse:
        state = require_ready(app)
        await require_knowledge_base_access(state.settings, request.knowledge_base_id, current_user)
        try:
            UUID(request.conversation_id)
            UUID(request.assistant_message_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Invalid conversation_id or assistant_message_id") from exc
        conversation = await get_conversation(
            state.settings,
            conversation_id=request.conversation_id,
            user_id=current_user.id,
            knowledge_base_id=request.knowledge_base_id,
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        messages = await list_conversation_messages(
            state.settings,
            conversation_id=request.conversation_id,
            user_id=current_user.id,
            knowledge_base_id=request.knowledge_base_id,
            limit=500,
        )
        assistant_index = next(
            (
                index
                for index, message in enumerate(messages)
                if message.id == request.assistant_message_id and message.role == "assistant"
            ),
            None,
        )
        if assistant_index is None:
            raise HTTPException(status_code=404, detail="Assistant message not found")
        assistant_message = messages[assistant_index]
        previous_user_message = next(
            (message for message in reversed(messages[:assistant_index]) if message.role == "user"),
            None,
        )
        feedback = await create_feedback(
            state.settings,
            org_id=current_user.org_id,
            knowledge_base_id=request.knowledge_base_id,
            user_id=current_user.id,
            conversation_id=request.conversation_id,
            assistant_message_id=request.assistant_message_id,
            rating=request.rating,
            reason=request.reason,
            comment=request.comment,
            question=previous_user_message.content if previous_user_message else request.question,
            answer=assistant_message.content,
            sources_snapshot=assistant_message.metadata.get("sources", request.sources_snapshot),
        )
        await add_audit_log(
            state.settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="feedback.create",
            target_type="assistant_message",
            target_id=request.assistant_message_id,
            metadata={"knowledge_base_id": request.knowledge_base_id, "rating": request.rating, "reason": request.reason},
            **audit_context(http_request),
        )
        return feedback_response(feedback)

    @app.get("/feedback", response_model=FeedbackListResponse)
    async def feedback_items(
        knowledge_base_id: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        current_user: CurrentUser = Depends(require_current_user),
    ) -> FeedbackListResponse:
        state = require_ready(app)
        require_manager(current_user)
        if knowledge_base_id:
            await require_knowledge_base_access(state.settings, knowledge_base_id, current_user)
        elif not current_user.is_admin:
            raise HTTPException(status_code=403, detail="Managers must filter feedback by knowledge_base_id")
        return FeedbackListResponse(
            items=[
                feedback_response(item)
                for item in await list_feedback(
                    state.settings,
                    org_id=current_user.org_id,
                    knowledge_base_id=knowledge_base_id,
                    limit=limit,
                    offset=offset,
                )
            ]
        )

    @app.post("/v1/knowledge/{knowledge_base_id}/retrieval", response_model=RetrievalResponse)
    async def open_retrieval(
        http_request: Request,
        knowledge_base_id: str,
        request: RetrievalRequest,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> RetrievalResponse:
        state = require_ready(app)
        started_at = time.perf_counter()
        api_key = await require_valid_api_key(state.settings, authorization, x_api_key)
        if api_key.knowledge_base_id != knowledge_base_id:
            await add_audit_log(
                state.settings,
                org_id=api_key.org_id,
                actor_user_id=api_key.user_id,
                action="openapi.retrieval",
                target_type="knowledge_base",
                target_id=knowledge_base_id,
                metadata={"api_key_id": api_key.id, "key_prefix": api_key.key_prefix, "bound_knowledge_base_id": api_key.knowledge_base_id},
                result="failed",
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                error_message="API key is not bound to this knowledge base",
                **audit_context(http_request),
            )
            raise HTTPException(status_code=403, detail="API key is not bound to this knowledge base")
        try:
            retrieval = await hybrid_search(
                state.settings,
                state.vector_store,
                request.query,
                top_k=request.top_k,
                user_id=api_key.user_id,
                knowledge_base_id=knowledge_base_id,
            )
            ranked = rerank_or_original(
                state.reranker,
                request.query,
                retrieval.documents,
                min(request.top_k, state.settings.rerank_top_n or request.top_k),
            )
            sources = [source_from_document(item.document, item.score) for item in ranked]
        except Exception as exc:  # noqa: BLE001
            await add_audit_log(
                state.settings,
                org_id=api_key.org_id,
                actor_user_id=api_key.user_id,
                action="openapi.retrieval",
                target_type="knowledge_base",
                target_id=knowledge_base_id,
                metadata={"api_key_id": api_key.id, "key_prefix": api_key.key_prefix, "top_k": request.top_k},
                result="failed",
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                error_message=str(exc)[:1000],
                **audit_context(http_request),
            )
            raise
        await add_audit_log(
            state.settings,
            org_id=api_key.org_id,
            actor_user_id=api_key.user_id,
            action="openapi.retrieval",
            target_type="knowledge_base",
            target_id=knowledge_base_id,
            metadata={"api_key_id": api_key.id, "key_prefix": api_key.key_prefix, "top_k": request.top_k, "source_count": len(sources)},
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            **audit_context(http_request),
        )
        return RetrievalResponse(items=sources)

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(
        http_request: Request,
        request: OpenAIChatCompletionRequest,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is reserved but not enabled yet")
        state = require_ready(app)
        api_key = await require_valid_api_key(state.settings, authorization, x_api_key)
        started_at = time.perf_counter()
        question = next((message.content for message in reversed(request.messages) if message.role == "user"), "")
        if not question:
            raise HTTPException(status_code=400, detail="At least one user message is required")
        top_k = request.top_k or state.settings.rag_top_k
        retrieval_query = await rewrite_query_async(
            state.chat_model,
            question,
            request.messages[:-1],
            state.settings.openai_timeout_seconds,
        )
        retrieval = await asyncio.wait_for(
            hybrid_search(
                state.settings,
                state.vector_store,
                retrieval_query,
                top_k=top_k,
                user_id=api_key.user_id,
                knowledge_base_id=api_key.knowledge_base_id,
            ),
            timeout=state.settings.retrieval_timeout_seconds,
        )
        ranked_documents = await rerank_or_original_async(
            state.reranker,
            retrieval_query,
            retrieval.documents,
            state.settings.rerank_top_n,
            state.settings.rerank_timeout_seconds,
        )
        selected = ranked_documents[: state.settings.rerank_top_n or len(ranked_documents)]
        if selected:
            prompt = build_prompt().invoke(
                {
                    "question": question,
                    "context": format_context([item.document for item in selected]),
                    "history": "\n".join(f"{item.role}: {item.content}" for item in request.messages[:-1]) or "No prior messages.",
                }
            )
            answer = str((await invoke_chat_model(
                state.chat_model,
                prompt,
                state.settings.openai_timeout_seconds,
            )).content)
        else:
            answer = "没有在当前知识库中检索到相关内容。请确认文档已经完成入库，或换一种问法。"
        sources = [source_from_document(item.document, item.score).model_dump() for item in selected]
        confidence, confidence_score = answer_confidence([Source(**source) for source in sources])
        conversation_id = request.conversation_id or str(uuid4())
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        await add_chat_log(
            state.settings,
            org_id=api_key.org_id,
            knowledge_base_id=api_key.knowledge_base_id,
            user_id=api_key.user_id,
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
        )
        await add_audit_log(
            state.settings,
            org_id=api_key.org_id,
            actor_user_id=api_key.user_id,
            action="openapi.chat_completion",
            target_type="knowledge_base",
            target_id=api_key.knowledge_base_id,
            metadata={"api_key_id": api_key.id, "conversation_id": conversation_id, "source_count": len(sources)},
            latency_ms=latency_ms,
            **audit_context(http_request),
        )
        prompt_tokens = sum(len(message.content) for message in request.messages) // 4
        completion_tokens = len(answer) // 4
        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model or state.settings.openai_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "sources": sources,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "conversation_id": conversation_id,
        }

    return app


def initialize_state() -> RagState:
    settings = AppSettings.load()
    errors = settings.validation_errors()
    state = RagState(settings=settings, init_errors=errors)
    if settings.postgres_dsn:
        try:
            initialize_database(settings)
            import asyncio

            asyncio.run(bootstrap_default_org(settings))
        except Exception as exc:  # noqa: BLE001
            state.init_errors = [*(state.init_errors or []), f"enterprise bootstrap failed: {exc}"]
    if errors:
        return state
    try:
        state.chat_model = build_chat_model(settings)
        state.embeddings = build_embeddings(settings)
        if hasattr(state.embeddings, "probe_dimension"):
            state.embeddings.probe_dimension()
        state.vector_store = build_vector_store(settings, state.embeddings)
        state.reranker = build_reranker(settings)
    except Exception as exc:  # noqa: BLE001
        state.init_errors = [str(exc)]
    return state


async def initialize_state_async() -> RagState:
    settings = AppSettings.load()
    errors = settings.validation_errors()
    state = RagState(settings=settings, init_errors=errors)
    if settings.postgres_dsn:
        try:
            await initialize_database_async(settings)
            await bootstrap_default_org(settings)
        except Exception as exc:  # noqa: BLE001
            state.init_errors = [*(state.init_errors or []), f"enterprise bootstrap failed: {exc}"]
    if errors:
        return state
    try:
        state.chat_model = build_chat_model(settings)
        state.embeddings = build_embeddings(settings)
        if hasattr(state.embeddings, "probe_dimension"):
            state.embeddings.probe_dimension()
        state.vector_store = build_vector_store(settings, state.embeddings, initialize_first=False)
        state.reranker = build_reranker(settings)
    except Exception as exc:  # noqa: BLE001
        state.init_errors = [str(exc)]
    return state


def get_state(app: FastAPI) -> RagState:
    state = getattr(app.state, "rag", None)
    if state is None:
        state = initialize_state()
        app.state.rag = state
    return state


def require_ready(app: FastAPI) -> RagState:
    state = get_state(app)
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "RAG service is not ready",
                "errors": state.init_errors or ["unknown initialization error"],
            },
        )
    return state


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a RAG assistant. Answer using only the provided context. "
                "If the context does not contain the answer, say you do not know. "
                "Use the same language as the user's question. "
                "Conversation history is for continuity only; retrieved context is the source of truth.",
            ),
            (
                "human",
                "Conversation history:\n{history}\n\nQuestion:\n{question}\n\nContext:\n{context}",
            ),
        ]
    )


def format_context(documents: list[Document]) -> str:
    if not documents:
        return "No relevant context was retrieved."
    parts = []
    for index, document in enumerate(documents, start=1):
        filename = document.metadata.get("filename", "unknown")
        chunk_index = document.metadata.get("chunk_index", "unknown")
        parts.append(
            f"[{index}] source={filename} chunk={chunk_index}\n{document.page_content}"
        )
    return "\n\n".join(parts)


def source_from_document(document: Document, rerank_score: float | None) -> Source:
    metadata = document.metadata
    snippet = document.page_content.strip().replace("\n", " ")
    if len(snippet) > 300:
        snippet = f"{snippet[:297]}..."
    vector_score = metadata.get("vector_score")
    bm25_score = metadata.get("bm25_score")
    hybrid_score = metadata.get("hybrid_score")
    bbox = metadata.get("bbox")
    return Source(
        file_id=metadata.get("file_id"),
        chunk_id=metadata.get("chunk_id"),
        document_id=metadata.get("document_id"),
        filename=metadata.get("filename"),
        chunk_index=metadata.get("chunk_index"),
        content_type=metadata.get("content_type"),
        upload_time=metadata.get("upload_time"),
        vector_score=float(vector_score) if vector_score is not None else None,
        bm25_score=float(bm25_score) if bm25_score is not None else None,
        hybrid_score=float(hybrid_score) if hybrid_score is not None else None,
        rerank_score=rerank_score,
        page_number=metadata.get("page_number"),
        bbox={key: float(value) for key, value in bbox.items()} if isinstance(bbox, dict) else None,
        snippet=snippet,
    )


def answer_confidence(sources: list[Source]) -> tuple[str, float | None]:
    scores = [
        score
        for source in sources
        for score in (source.rerank_score, source.hybrid_score, source.vector_score, source.bm25_score)
        if score is not None
    ]
    if not sources:
        return "low", None
    if not scores:
        return "medium", None
    best_score = max(scores)
    if best_score >= 0.7 and len(sources) >= 2:
        return "high", best_score
    if best_score >= 0.35:
        return "medium", best_score
    return "low", best_score


def feedback_response(feedback: Feedback) -> FeedbackResponse:
    return FeedbackResponse(
        id=feedback.id,
        knowledge_base_id=feedback.knowledge_base_id,
        user_id=feedback.user_id,
        conversation_id=feedback.conversation_id,
        assistant_message_id=feedback.assistant_message_id,
        rating=feedback.rating,
        reason=feedback.reason,
        comment=feedback.comment,
        question=feedback.question,
        answer=feedback.answer,
        sources_snapshot=feedback.sources_snapshot,
        created_at=feedback.created_at.isoformat(),
    )


def file_record_response(file_record: RagFile) -> FileRecordResponse:
    return FileRecordResponse(
        id=file_record.id,
        org_id=file_record.org_id,
        knowledge_base_id=file_record.knowledge_base_id,
        owner_user_id=file_record.owner_user_id,
        user_id=file_record.user_id,
        filename=file_record.filename,
        content_type=file_record.content_type,
        source_type=file_record.source_type,
        source_uri=file_record.source_uri,
        file_size=file_record.file_size,
        file_sha256=file_record.file_sha256,
        content_sha256=file_record.content_sha256,
        chunk_count=file_record.chunk_count,
        chunk_ids=file_record.chunk_ids,
        vector_ids=file_record.vector_ids,
        status=file_record.status,
        error_message=file_record.error_message,
        created_at=file_record.created_at.isoformat(),
        updated_at=file_record.updated_at.isoformat(),
    )


def user_response(user: Any) -> UserResponse:
    return UserResponse(
        id=user.id,
        org_id=user.org_id,
        department_id=user.department_id,
        email=user.email,
        display_name=user.display_name,
        role=user.role,
        is_active=user.is_active,
        last_login_at=user.last_login_at.isoformat() if getattr(user, "last_login_at", None) else None,
        created_at=user.created_at.isoformat() if user.created_at else "",
    )


def api_key_response(api_key: ApiKey) -> ApiKeyResponse:
    return ApiKeyResponse(
        id=api_key.id,
        org_id=api_key.org_id,
        user_id=api_key.user_id,
        knowledge_base_id=api_key.knowledge_base_id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        is_active=api_key.is_active,
        last_used_at=api_key.last_used_at.isoformat() if api_key.last_used_at else None,
        created_at=api_key.created_at.isoformat(),
    )


def department_response(department: Any) -> DepartmentResponse:
    return DepartmentResponse(
        id=department.id,
        org_id=department.org_id,
        name=department.name,
        parent_id=department.parent_id,
        created_at=department.created_at.isoformat(),
    )


def kb_response(kb: Any, stats: Any | None = None, capabilities: Any | None = None) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse(
        id=kb.id,
        org_id=kb.org_id,
        owner_user_id=kb.owner_user_id,
        name=kb.name,
        description=kb.description,
        visibility=kb.visibility,
        department_ids=kb.department_ids,
        status=getattr(kb, "status", "active"),
        file_count=int(getattr(stats, "file_count", 0) if stats else 0),
        completed_file_count=int(getattr(stats, "completed_file_count", 0) if stats else 0),
        failed_job_count=int(getattr(stats, "failed_job_count", 0) if stats else 0),
        current_user_role=getattr(capabilities, "current_user_role", "none"),
        can_read=bool(getattr(capabilities, "can_read", False)),
        can_write=bool(getattr(capabilities, "can_write", False)),
        can_manage_members=bool(getattr(capabilities, "can_manage_members", False)),
        can_manage_settings=bool(getattr(capabilities, "can_manage_settings", False)),
        can_manage_api_keys=bool(getattr(capabilities, "can_manage_api_keys", False)),
        created_at=kb.created_at.isoformat(),
        updated_at=kb.updated_at.isoformat(),
    )


def kb_member_response(member: Any) -> KnowledgeBaseMemberResponse:
    return KnowledgeBaseMemberResponse(
        id=member.id,
        knowledge_base_id=member.knowledge_base_id,
        user_id=member.user_id,
        role=member.role,
        email=member.email,
        display_name=member.display_name,
        department_id=member.department_id,
        created_at=member.created_at.isoformat(),
    )


def ingest_job_response(job: IngestJob) -> IngestJobResponse:
    return IngestJobResponse(
        id=job.id,
        org_id=job.org_id,
        knowledge_base_id=job.knowledge_base_id,
        created_by_user_id=job.created_by_user_id,
        source_type=job.source_type,
        source_uri=job.source_uri,
        filename=job.filename,
        status=job.status,
        progress=job.progress,
        error_message=job.error_message,
        retry_count=job.retry_count,
        payload=job.payload,
        file_id=job.file_id,
        duration_ms=job.duration_ms,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
    )


def batch_response(items: list[BatchItemResult]) -> BatchOperationResponse:
    failed = sum(1 for item in items if item.status == "failed")
    skipped = sum(1 for item in items if item.status == "skipped")
    succeeded = sum(1 for item in items if item.status == "succeeded")
    return BatchOperationResponse(succeeded=succeeded, failed=failed, skipped=skipped, items=items)


def queue_health_response(
    health: IngestQueueHealth,
    *,
    redis_queue_length: int,
    worker_last_seen_at: str | None,
) -> QueueHealthResponse:
    now = datetime.now(timezone.utc)
    oldest_pending_wait_seconds = seconds_since(health.oldest_pending_at, now)
    oldest_running_seconds = seconds_since(health.oldest_running_at, now)
    worker_seen_seconds = seconds_since(parse_datetime(worker_last_seen_at), now) if worker_last_seen_at else None
    return QueueHealthResponse(
        pending_count=health.pending_count,
        running_count=health.running_count,
        succeeded_count=health.succeeded_count,
        failed_count=health.failed_count,
        cancelled_count=health.cancelled_count,
        redis_queue_length=redis_queue_length,
        oldest_pending_at=health.oldest_pending_at.isoformat() if health.oldest_pending_at else None,
        oldest_pending_wait_seconds=oldest_pending_wait_seconds,
        oldest_running_at=health.oldest_running_at.isoformat() if health.oldest_running_at else None,
        oldest_running_seconds=oldest_running_seconds,
        worker_last_seen_at=worker_last_seen_at,
        worker_stale=worker_seen_seconds is None or worker_seen_seconds > 90,
    )


def seconds_since(value: datetime | None, now: datetime) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return max(0, int((now - value).total_seconds()))


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def require_manager(user: CurrentUser) -> None:
    if not user.is_manager:
        raise HTTPException(status_code=403, detail="Manager or admin role required")


def audit_context(request: Request) -> dict[str, Any]:
    return {
        "ip_address": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
        "request_id": request.headers.get("x-request-id") or str(uuid4()),
    }


async def require_valid_api_key(
    settings: AppSettings,
    authorization: str | None,
    x_api_key: str | None,
) -> ApiKey:
    secret = x_api_key
    if not secret and authorization:
        if authorization.lower().startswith("bearer "):
            secret = authorization.split(" ", 1)[1].strip()
        elif authorization.lower().startswith("api-key "):
            secret = authorization.split(" ", 1)[1].strip()
    if not secret:
        raise HTTPException(status_code=401, detail="API key required")
    api_key = await verify_api_key(settings, secret)
    if api_key is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


async def require_kb_member_admin(settings: AppSettings, kb_id: str, user: CurrentUser) -> None:
    capabilities = await get_knowledge_base_capabilities(settings, kb_id, user)
    if capabilities.can_manage_members:
        return
    raise HTTPException(status_code=403, detail="Knowledge base owner or admin role required")


async def require_kb_settings_admin(settings: AppSettings, kb_id: str, user: CurrentUser) -> None:
    capabilities = await get_knowledge_base_capabilities(settings, kb_id, user)
    if capabilities.can_manage_settings:
        return
    raise HTTPException(status_code=403, detail="Knowledge base owner or admin role required")


async def require_kb_api_key_admin(settings: AppSettings, kb_id: str, user: CurrentUser) -> None:
    capabilities = await get_knowledge_base_capabilities(settings, kb_id, user)
    if capabilities.can_manage_api_keys:
        return
    raise HTTPException(status_code=403, detail="Knowledge base API key management denied")


def build_reindex_payload(settings: AppSettings, item: RagFile) -> tuple[dict[str, Any], str | None]:
    if item.source_type == "file":
        path = safe_upload_path(settings.upload_storage_dir, item.source_uri or "")
        if path is None or not path.exists():
            raise ValueError("Stored source file is missing; upload it again")
        return {"path": str(path), "content_type": item.content_type, "file_size": item.file_size}, str(path)
    if item.source_type == "url":
        return {"url": item.source_uri}, item.source_uri
    raise ValueError(f"Unsupported source type: {item.source_type}")


def plan_cache_payload(plan: UrlImportPlanResponse, user: CurrentUser) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "org_id": user.org_id,
        "user_id": user.id,
        "knowledge_base_id": plan.knowledge_base_id,
        "created_at": plan.created_at,
        "expires_at": plan.expires_at,
        "items": [item.model_dump() for item in plan.items],
    }


async def build_url_import_plan(
    settings: AppSettings,
    *,
    org_id: str,
    knowledge_base_id: str,
    user_id: str,
    urls: list[str],
    skip_duplicates: bool,
) -> UrlImportPlanResponse:
    now = datetime.now(timezone.utc)
    plan_id = str(uuid4())
    expires_at = now + timedelta(seconds=IMPORT_PLAN_TTL_SECONDS)
    items: list[UrlImportPlanItem] = []
    seen_urls: dict[str, UrlImportPlanItem] = {}
    seen_hashes: dict[str, str] = {}
    for index, raw_url in enumerate(urls):
        url = raw_url.strip()
        client_item_id = f"url-{index}"
        if not url:
            items.append(
                UrlImportPlanItem(
                    index=index,
                    client_item_id=client_item_id,
                    url=raw_url,
                    status="invalid_url",
                    severity="blocked",
                    can_enqueue=False,
                    reason_code="empty_url",
                    reason="URL is empty",
                )
            )
            continue
        normalized_url = normalize_url(url)
        if normalized_url in seen_urls:
            first_item = seen_urls[normalized_url]
            can_enqueue = (not skip_duplicates) and first_item.can_enqueue
            items.append(
                UrlImportPlanItem(
                    index=index,
                    client_item_id=client_item_id,
                    url=url,
                    filename=first_item.filename,
                    content_type=first_item.content_type,
                    file_size=first_item.file_size,
                    status="duplicate_in_batch",
                    severity="warning" if can_enqueue else "blocked",
                    can_enqueue=can_enqueue,
                    reason_code="duplicate_url_in_batch",
                    reason="Duplicate URL in this batch" if can_enqueue else "Duplicate URL points to a blocked or failed item",
                    content_sha256=first_item.content_sha256,
                    content_length=first_item.content_length,
                    estimated_chunks=first_item.estimated_chunks,
                    duplicate_file_id=first_item.duplicate_file_id,
                    duplicate_of=first_item.client_item_id,
                )
            )
            continue
        item = await plan_single_url_import(
            settings,
            index=index,
            client_item_id=client_item_id,
            url=url,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            skip_duplicates=skip_duplicates,
        )
        seen_urls[normalized_url] = item
        if item.content_sha256 and item.content_sha256 in seen_hashes:
            item = item.model_copy(
                update={
                    "status": "duplicate_in_batch",
                    "severity": "blocked" if skip_duplicates else "warning",
                    "can_enqueue": not skip_duplicates,
                    "reason_code": "duplicate_content_in_batch",
                    "reason": "Same parsed content appears earlier in this batch",
                    "duplicate_of": seen_hashes[item.content_sha256],
                }
            )
        elif item.content_sha256:
            seen_hashes[item.content_sha256] = item.client_item_id
        items.append(item)
    blocked_count = sum(1 for item in items if item.severity == "blocked")
    return UrlImportPlanResponse(
        plan_id=plan_id,
        knowledge_base_id=knowledge_base_id,
        created_at=now.isoformat(),
        expires_at=expires_at.isoformat(),
        total=len(items),
        ready_count=sum(1 for item in items if item.can_enqueue),
        warning_count=sum(1 for item in items if item.severity == "warning"),
        blocked_count=blocked_count,
        duplicate_count=sum(1 for item in items if item.status in {"duplicate_existing", "duplicate_in_batch"}),
        invalid_count=sum(1 for item in items if item.status == "invalid_url"),
        error_count=sum(1 for item in items if item.status in {"url_fetch_failed", "empty_content", "parse_failed", "system_error"}),
        items=items,
    )


async def plan_single_url_import(
    settings: AppSettings,
    *,
    index: int,
    client_item_id: str,
    url: str,
    user_id: str,
    knowledge_base_id: str,
    skip_duplicates: bool,
) -> UrlImportPlanItem:
    parsed_url = urlsplit(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return UrlImportPlanItem(
            index=index,
            client_item_id=client_item_id,
            url=url,
            status="invalid_url",
            severity="blocked",
            can_enqueue=False,
            reason_code="invalid_url",
            reason="Only http(s) URLs are supported",
        )
    if not is_public_http_url(parsed_url.hostname):
        return UrlImportPlanItem(
            index=index,
            client_item_id=client_item_id,
            url=url,
            status="invalid_url",
            severity="blocked",
            can_enqueue=False,
            reason_code="private_or_local_host",
            reason="Private, local, or reserved hosts are not allowed",
        )
    try:
        html = await asyncio.to_thread(_fetch_url, url, 10)
        parsed = parse_html(_slugify(_extract_title(html) or url) + ".html", html)
        content = parsed.text.strip()
        if not content:
            return UrlImportPlanItem(
                index=index,
                client_item_id=client_item_id,
                url=url,
                status="empty_content",
                severity="blocked",
                can_enqueue=False,
                reason_code="empty_content",
                reason="No extractable text found",
            )
        content_sha256 = hash_text(content)
        duplicate = await find_file_by_content_hash(
            settings,
            user_id=user_id,
            content_sha256=content_sha256,
            knowledge_base_id=knowledge_base_id,
        )
        title = _extract_title(html)
        filename = f"{_slugify(title or url)}.html"
        _, chunks = build_chunks_from_text(
            filename=filename,
            content_type=parsed.content_type,
            text=content,
            settings=settings,
            document_id=str(uuid4()),
            content_sha256=content_sha256,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            source_type="url",
            source_uri=url,
        )
        if duplicate:
            return UrlImportPlanItem(
                index=index,
                client_item_id=client_item_id,
                url=url,
                filename=filename,
                content_type=parsed.content_type,
                file_size=len(html.encode("utf-8")),
                status="duplicate_existing",
                severity="blocked" if skip_duplicates else "warning",
                can_enqueue=not skip_duplicates,
                reason_code="duplicate_existing",
                reason="Same content already exists in this knowledge base",
                content_sha256=content_sha256,
                content_length=len(content),
                estimated_chunks=len(chunks),
                duplicate_file_id=duplicate.id,
            )
        return UrlImportPlanItem(
            index=index,
            client_item_id=client_item_id,
            url=url,
            filename=filename,
            content_type=parsed.content_type,
            file_size=len(html.encode("utf-8")),
            status="ready",
            severity="pass",
            can_enqueue=True,
            content_sha256=content_sha256,
            content_length=len(content),
            estimated_chunks=len(chunks),
        )
    except Exception as exc:  # noqa: BLE001
        return UrlImportPlanItem(
            index=index,
            client_item_id=client_item_id,
            url=url,
            status="url_fetch_failed",
            severity="blocked",
            can_enqueue=False,
            reason_code="url_fetch_failed",
            reason=str(exc)[:1000],
        )


def normalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    return parsed._replace(fragment="").geturl().rstrip("/")


def is_public_http_url(hostname: str | None) -> bool:
    if not hostname:
        return False
    host = hostname.strip().lower().rstrip(".")
    if host in {"localhost"} or host.endswith(".localhost") or host.endswith(".local"):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def safe_upload_path(upload_storage_dir: str, source_uri: str) -> Path | None:
    if not source_uri:
        return None
    base = Path(upload_storage_dir).resolve()
    path = Path(source_uri).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        return None
    return path


def rerank_or_original(
    reranker: Reranker,
    query: str,
    documents: list[Document],
    top_n: int,
) -> list[RerankedDocument]:
    if not documents:
        return []
    if top_n <= 0:
        return [RerankedDocument(document=document, score=None) for document in documents]
    try:
        return reranker.rerank(query, documents, top_n)
    except Exception:  # noqa: BLE001
        return [RerankedDocument(document=document, score=None) for document in documents[:top_n]]


async def rerank_or_original_async(
    reranker: Reranker,
    query: str,
    documents: list[Document],
    top_n: int,
    timeout_seconds: int,
) -> list[RerankedDocument]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(rerank_or_original, reranker, query, documents, top_n),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return [RerankedDocument(document=document, score=None) for document in documents[:top_n]]


def rewrite_query(chat_model: Any, question: str, history: list[Any]) -> str:
    if not history:
        return question
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Rewrite the latest user question into a standalone retrieval query. "
                    "Keep names, entities, dates, and the user's language. Return only the query.",
                ),
                ("human", "History:\n{history}\n\nLatest question:\n{question}"),
            ]
        ).invoke({"history": format_history(history), "question": question})
        rewritten = str(chat_model.invoke(prompt).content).strip()
        return rewritten or question
    except Exception:  # noqa: BLE001
        return question


async def rewrite_query_async(chat_model: Any, question: str, history: list[Any], timeout_seconds: int) -> str:
    if not history:
        return question
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Rewrite the latest user question into a standalone retrieval query. "
                    "Keep names, entities, dates, and the user's language. Return only the query.",
                ),
                ("human", "History:\n{history}\n\nLatest question:\n{question}"),
            ]
        ).invoke({"history": format_history(history), "question": question})
        rewritten = str((await invoke_chat_model(chat_model, prompt, timeout_seconds)).content).strip()
        return rewritten or question
    except Exception:  # noqa: BLE001
        return question


async def invoke_chat_model(chat_model: Any, prompt: Any, timeout_seconds: int) -> Any:
    return await asyncio.wait_for(chat_model.ainvoke(prompt), timeout=timeout_seconds)


def conversation_response(conversation: Conversation) -> ConversationResponse:
    return ConversationResponse(
        id=conversation.id,
        user_id=conversation.user_id,
        knowledge_base_id=conversation.knowledge_base_id,
        title=conversation.title,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
    )


def chat_message_response(message: Any) -> ChatMessageResponse:
    return ChatMessageResponse(
        id=message.id,
        conversation_id=message.conversation_id,
        knowledge_base_id=message.knowledge_base_id,
        role=message.role,
        content=message.content,
        metadata=message.metadata,
        created_at=message.created_at.isoformat(),
    )


def format_history(messages: list[Any]) -> str:
    if not messages:
        return "No prior messages."
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


app = create_app()
