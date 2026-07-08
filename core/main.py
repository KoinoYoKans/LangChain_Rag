from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError

from config.model import build_chat_model
from config.settings import AppSettings
from core.conversation_store import (
    Conversation,
    add_message,
    ensure_conversation,
    get_recent_messages,
    list_conversations,
)
from core.document_loader import (
    SUPPORTED_EXTENSIONS,
    build_chunks_from_text,
    extract_text,
    hash_bytes,
    hash_text,
)
from core.embeddings import build_embeddings
from core.file_store import (
    RagFile,
    create_processing_file,
    delete_file_chunks,
    delete_file_record,
    find_file_by_content_hash,
    get_file,
    list_files,
    mark_file_completed,
    mark_file_failed,
    save_file_chunks,
)
from core.rerank import Reranker, build_reranker
from core.vector_store import (
    add_documents,
    build_vector_store,
    check_database,
    delete_documents,
    initialize_database_async,
    similarity_search,
)


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1)
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
    rerank_score: float | None = None
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
    title: str | None
    created_at: str
    updated_at: str


class ConversationListResponse(BaseModel):
    items: list[ConversationResponse]


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
            },
        }

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
        user_id: str = Query(..., min_length=1),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> ConversationListResponse:
        state = require_ready(app)
        return ConversationListResponse(
            items=[
                conversation_response(item)
                for item in await list_conversations(state.settings, user_id=user_id, limit=limit, offset=offset)
            ]
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        state = require_ready(app)
        top_k = request.top_k or state.settings.rag_top_k
        rerank_top_n = request.rerank_top_n if request.rerank_top_n is not None else state.settings.rerank_top_n
        conversation_id = request.conversation_id or str(uuid4())
        try:
            await ensure_conversation(
                state.settings,
                conversation_id=conversation_id,
                user_id=request.user_id,
                title=request.message[:80],
            )
            history = await get_recent_messages(
                state.settings,
                conversation_id=conversation_id,
                user_id=request.user_id,
                limit=state.settings.chat_history_limit,
            )
            user_message = await add_message(
                state.settings,
                message_id=str(uuid4()),
                conversation_id=conversation_id,
                user_id=request.user_id,
                role="user",
                content=request.message,
            )
            retrieved = similarity_search(
                state.vector_store,
                request.message,
                top_k=top_k,
                user_id=request.user_id,
            )
            ranked_documents = state.reranker.rerank(request.message, retrieved, rerank_top_n)
            selected = ranked_documents[: rerank_top_n or len(ranked_documents)]
            context = format_context([item.document for item in selected])
            prompt = build_prompt().invoke(
                {
                    "question": request.message,
                    "context": context,
                    "history": format_history(history),
                }
            )
            answer = state.chat_model.invoke(prompt).content
            sources = [
                source_from_document(item.document, rerank_score=item.score)
                for item in selected
            ]
            assistant_message = await add_message(
                state.settings,
                message_id=str(uuid4()),
                conversation_id=conversation_id,
                user_id=request.user_id,
                role="assistant",
                content=str(answer),
                metadata={"sources": [source.model_dump() for source in sources]},
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

        return ChatResponse(
            conversation_id=conversation_id,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            answer=str(answer),
            sources=sources,
        )

    return app


def initialize_state() -> RagState:
    settings = AppSettings.load()
    errors = settings.validation_errors()
    state = RagState(settings=settings, init_errors=errors)
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
    if errors:
        return state
    try:
        state.chat_model = build_chat_model(settings)
        state.embeddings = build_embeddings(settings)
        if hasattr(state.embeddings, "probe_dimension"):
            state.embeddings.probe_dimension()
        await initialize_database_async(settings)
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
    return Source(
        file_id=metadata.get("file_id"),
        chunk_id=metadata.get("chunk_id"),
        document_id=metadata.get("document_id"),
        filename=metadata.get("filename"),
        chunk_index=metadata.get("chunk_index"),
        content_type=metadata.get("content_type"),
        upload_time=metadata.get("upload_time"),
        vector_score=float(vector_score) if vector_score is not None else None,
        rerank_score=rerank_score,
        snippet=snippet,
    )


def file_record_response(file_record: RagFile) -> FileRecordResponse:
    return FileRecordResponse(
        id=file_record.id,
        user_id=file_record.user_id,
        filename=file_record.filename,
        content_type=file_record.content_type,
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


def conversation_response(conversation: Conversation) -> ConversationResponse:
    return ConversationResponse(
        id=conversation.id,
        user_id=conversation.user_id,
        title=conversation.title,
        created_at=conversation.created_at.isoformat(),
        updated_at=conversation.updated_at.isoformat(),
    )


def format_history(messages: list[Any]) -> str:
    if not messages:
        return "No prior messages."
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


app = create_app()
