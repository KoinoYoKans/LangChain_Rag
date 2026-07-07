from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.model import build_chat_model
from config.settings import AppSettings
from core.document_loader import SUPPORTED_EXTENSIONS, build_chunks
from core.embeddings import build_embeddings
from core.rerank import Reranker, build_reranker
from core.vector_store import add_documents, build_vector_store, check_database, similarity_search


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank_top_n: int | None = Field(default=None, ge=0, le=50)


class Source(BaseModel):
    document_id: str | None
    filename: str | None
    chunk_index: int | None
    content_type: str | None
    upload_time: str | None
    vector_score: float | None = None
    rerank_score: float | None = None
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    ids: list[str]


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
        description="FastAPI RAG service with OpenAI-compatible chat, Qwen embeddings/rerank, and pgvector.",
    )

    @app.on_event("startup")
    def startup() -> None:
        app.state.rag = initialize_state()

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
                "embedding_model": state.settings.local_embedding_model_path
                if state.settings.embedding_provider == "local"
                else state.settings.qwen_embedding_model,
                "rerank_provider": state.settings.rerank_provider,
                "rerank_model": state.settings.local_rerank_model_path
                if state.settings.rerank_provider == "local"
                else state.settings.qwen_rerank_model,
                "pgvector_table": state.settings.pgvector_table,
                "rag_top_k": state.settings.rag_top_k,
                "rerank_top_n": state.settings.rerank_top_n,
            },
        }

    @app.post("/documents", response_model=DocumentUploadResponse)
    async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
        state = require_ready(app)
        filename = file.filename or "uploaded"
        if not any(filename.lower().endswith(extension) for extension in SUPPORTED_EXTENSIONS):
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise HTTPException(status_code=400, detail=f"Unsupported document type. Supported: {supported}")
        data = await file.read()
        try:
            document_id, documents = build_chunks(
                filename=filename,
                content_type=file.content_type,
                data=data,
                settings=state.settings,
            )
            ids = add_documents(state.vector_store, documents)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Document ingestion failed: {exc}") from exc
        return DocumentUploadResponse(
            document_id=document_id,
            filename=filename,
            chunks=len(documents),
            ids=ids,
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        state = require_ready(app)
        top_k = request.top_k or state.settings.rag_top_k
        rerank_top_n = request.rerank_top_n if request.rerank_top_n is not None else state.settings.rerank_top_n
        try:
            retrieved = similarity_search(state.vector_store, request.message, top_k=top_k)
            ranked_documents = state.reranker.rerank(request.message, retrieved, rerank_top_n)
            selected = ranked_documents[: rerank_top_n or len(ranked_documents)]
            context = format_context([item.document for item in selected])
            prompt = build_prompt().invoke(
                {
                    "question": request.message,
                    "context": context,
                }
            )
            answer = state.chat_model.invoke(prompt).content
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

        sources = [
            source_from_document(item.document, rerank_score=item.score)
            for item in selected
        ]
        return ChatResponse(answer=str(answer), sources=sources)

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
                "Use the same language as the user's question.",
            ),
            (
                "human",
                "Question:\n{question}\n\nContext:\n{context}",
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
        document_id=metadata.get("document_id"),
        filename=metadata.get("filename"),
        chunk_index=metadata.get("chunk_index"),
        content_type=metadata.get("content_type"),
        upload_time=metadata.get("upload_time"),
        vector_score=float(vector_score) if vector_score is not None else None,
        rerank_score=rerank_score,
        snippet=snippet,
    )


app = create_app()
