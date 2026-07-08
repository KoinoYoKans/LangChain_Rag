from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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
from core.api_key_store import ApiKey, create_api_key, list_api_keys, verify_api_key
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
    get_knowledge_base_stats,
    is_knowledge_base_owner,
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
from core.ingest_store import IngestJob, create_ingest_job, get_ingest_job, list_ingest_jobs
from core.ingest_store import retry_failed_ingest_job
from core.job_queue import enqueue_ingest_job
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
        return KnowledgeBaseListResponse(
            items=[
                kb_response(item, await get_knowledge_base_stats(settings, item.id))
                for item in await list_knowledge_bases(settings, current_user)
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
        return kb_response(kb, await get_knowledge_base_stats(settings, kb.id))

    @app.patch("/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
    async def update_knowledge_base_endpoint(
        http_request: Request,
        knowledge_base_id: str,
        request: KnowledgeBaseUpdateRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> KnowledgeBaseResponse:
        require_manager(current_user)
        settings = get_state(app).settings
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
        return kb_response(kb, await get_knowledge_base_stats(settings, kb.id))

    @app.delete("/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
    async def delete_knowledge_base_endpoint(
        http_request: Request,
        knowledge_base_id: str,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> KnowledgeBaseResponse:
        require_manager(current_user)
        settings = get_state(app).settings
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
        await add_audit_log(
            settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="knowledge_base.member_upsert",
            target_type="knowledge_base",
            target_id=knowledge_base_id,
            metadata={"user_id": request.user_id, "role": request.role},
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
        await remove_knowledge_base_member(settings, kb_id=knowledge_base_id, user_id=member_user_id)
        await add_audit_log(
            settings,
            org_id=current_user.org_id,
            actor_user_id=current_user.id,
            actor_department_id=current_user.department_id,
            action="knowledge_base.member_remove",
            target_type="knowledge_base",
            target_id=knowledge_base_id,
            metadata={"user_id": member_user_id},
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
        return AuditLogListResponse(items=await list_audit_logs(settings, current_user.org_id, limit=limit))

    @app.get("/api-keys", response_model=ApiKeyListResponse)
    async def api_keys(current_user: CurrentUser = Depends(require_current_user)) -> ApiKeyListResponse:
        require_manager(current_user)
        settings = get_state(app).settings
        return ApiKeyListResponse(items=[api_key_response(item) for item in await list_api_keys(settings, current_user.org_id)])

    @app.post("/api-keys", response_model=ApiKeyCreateResponse)
    async def create_api_key_endpoint(
        http_request: Request,
        request: ApiKeyCreateRequest,
        current_user: CurrentUser = Depends(require_current_user),
    ) -> ApiKeyCreateResponse:
        require_manager(current_user)
        state = get_state(app)
        await require_knowledge_base_access(state.settings, request.knowledge_base_id, current_user, write=True)
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

    @app.post("/documents", response_model=DocumentUploadResponse)
    async def upload_document(
        user_id: str = Form(..., min_length=1),
        file: UploadFile = File(...),
    ) -> DocumentUploadResponse:
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
        if item.source_type == "file":
            path = safe_upload_path(state.settings.upload_storage_dir, item.source_uri or "")
            if path is None or not path.exists():
                raise HTTPException(status_code=400, detail="Stored source file is missing; upload it again")
            payload = {"path": str(path), "content_type": item.content_type, "file_size": item.file_size}
            source_uri = str(path)
        elif item.source_type == "url":
            payload = {"url": item.source_uri}
            source_uri = item.source_uri
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source type: {item.source_type}")
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

    @app.get("/documents", response_model=FileListResponse)
    async def documents(
        user_id: str = Query(..., min_length=1),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> FileListResponse:
        state = require_ready(app)
        return FileListResponse(
            items=[
                file_record_response(item)
                for item in await list_files(state.settings, user_id=user_id, limit=limit, offset=offset)
            ]
        )

    @app.get("/documents/{file_id}", response_model=FileRecordResponse)
    async def document_detail(file_id: str, user_id: str = Query(..., min_length=1)) -> FileRecordResponse:
        state = require_ready(app)
        item = await get_file(state.settings, user_id, file_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return file_record_response(item)

    @app.delete("/documents/{file_id}", response_model=DeleteDocumentResponse)
    async def delete_document(file_id: str, user_id: str = Query(..., min_length=1)) -> DeleteDocumentResponse:
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
            assistant_message = await add_message(
                state.settings,
                message_id=str(uuid4()),
                conversation_id=conversation_id,
                org_id=current_user.org_id,
                knowledge_base_id=request.knowledge_base_id,
                user_id=current_user.id,
                role="assistant",
                content=str(answer),
                metadata={"sources": [source.model_dump() for source in sources]},
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
        )

    @app.post("/v1/knowledge/{knowledge_base_id}/retrieval", response_model=RetrievalResponse)
    async def open_retrieval(
        knowledge_base_id: str,
        request: RetrievalRequest,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> RetrievalResponse:
        state = require_ready(app)
        api_key = await require_valid_api_key(state.settings, authorization, x_api_key)
        if api_key.knowledge_base_id != knowledge_base_id:
            raise HTTPException(status_code=403, detail="API key is not bound to this knowledge base")
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
        return RetrievalResponse(items=[source_from_document(item.document, item.score) for item in ranked])

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


def kb_response(kb: Any, stats: Any | None = None) -> KnowledgeBaseResponse:
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
    if user.is_manager or await is_knowledge_base_owner(settings, kb_id=kb_id, user_id=user.id):
        return
    raise HTTPException(status_code=403, detail="Knowledge base owner or manager role required")


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
