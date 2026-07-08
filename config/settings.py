from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, quote_plus, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv


def _as_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_postgres_dsn() -> str | None:
    explicit_dsn = os.getenv("POSTGRES_DSN")
    if explicit_dsn:
        return explicit_dsn

    host = os.getenv("PG_HOST")
    user = os.getenv("PG_USER")
    password = os.getenv("PG_PWD")
    database = os.getenv("PG_DATABASE", "postgres")
    sslmode = os.getenv("PG_SSLMODE")
    if not host or not user or password is None:
        return None

    dsn = f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}@{host}/{database}"
    if sslmode:
        dsn = f"{dsn}?sslmode={quote_plus(sslmode)}"
    return dsn


@dataclass(frozen=True)
class AppSettings:
    app_env: str
    app_host: str
    app_port: int
    log_level: str
    openai_api_key: str | None
    openai_base_url: str | None
    openai_model: str
    default_model_agent_name: str
    dashscope_api_key: str | None
    embedding_provider: str
    local_embedding_model_path: str | None
    local_embedding_device: str | None
    local_embedding_trust_remote_code: bool
    qwen_embedding_model: str
    rerank_provider: str
    local_rerank_model_path: str | None
    local_rerank_device: str | None
    local_rerank_trust_remote_code: bool
    local_rerank_max_length: int
    local_rerank_batch_size: int
    local_rerank_instruction: str
    qwen_rerank_model: str
    embedding_dimension: int
    embedding_batch_size: int
    postgres_dsn: str | None
    pgvector_table: str
    rag_file_table: str
    rag_chunk_table: str
    rag_conversation_table: str
    rag_message_table: str
    chat_history_limit: int
    rag_top_k: int
    rerank_top_n: int
    chunk_size: int
    chunk_overlap: int

    @classmethod
    def load(cls, env_file: str | Path = ".env") -> "AppSettings":
        load_dotenv(env_file)
        return cls(
            app_env=os.getenv("APP_ENV", "development"),
            app_host=os.getenv("APP_HOST", "127.0.0.1"),
            app_port=_as_int("APP_PORT", 8000),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
            default_model_agent_name=os.getenv("DEFAULT_MODEL_AGENT_NAME", "default"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local").lower(),
            local_embedding_model_path=os.getenv("LOCAL_EMBEDDING_MODEL_PATH"),
            local_embedding_device=os.getenv("LOCAL_EMBEDDING_DEVICE"),
            local_embedding_trust_remote_code=_as_bool(
                "LOCAL_EMBEDDING_TRUST_REMOTE_CODE",
                True,
            ),
            qwen_embedding_model=os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v4"),
            rerank_provider=os.getenv("RERANK_PROVIDER", "local").lower(),
            local_rerank_model_path=os.getenv("LOCAL_RERANK_MODEL_PATH"),
            local_rerank_device=os.getenv("LOCAL_RERANK_DEVICE"),
            local_rerank_trust_remote_code=_as_bool("LOCAL_RERANK_TRUST_REMOTE_CODE", True),
            local_rerank_max_length=_as_int("LOCAL_RERANK_MAX_LENGTH", 8192),
            local_rerank_batch_size=_as_int("LOCAL_RERANK_BATCH_SIZE", 4),
            local_rerank_instruction=os.getenv(
                "LOCAL_RERANK_INSTRUCTION",
                "Given a web search query, retrieve relevant passages that answer the query",
            ),
            qwen_rerank_model=os.getenv("QWEN_RERANK_MODEL", "gte-rerank-v2"),
            embedding_dimension=_as_int("EMBEDDING_DIMENSION", 1024),
            embedding_batch_size=_as_int("EMBEDDING_BATCH_SIZE", 16),
            postgres_dsn=_build_postgres_dsn(),
            pgvector_table=os.getenv("PGVECTOR_TABLE", "rag_documents"),
            rag_file_table=os.getenv("RAG_FILE_TABLE", "rag_files"),
            rag_chunk_table=os.getenv("RAG_CHUNK_TABLE", "rag_file_chunks"),
            rag_conversation_table=os.getenv("RAG_CONVERSATION_TABLE", "rag_conversations"),
            rag_message_table=os.getenv("RAG_MESSAGE_TABLE", "rag_messages"),
            chat_history_limit=_as_int("CHAT_HISTORY_LIMIT", 10),
            rag_top_k=_as_int("RAG_TOP_K", 8),
            rerank_top_n=_as_int("RERANK_TOP_N", 4),
            chunk_size=_as_int("CHUNK_SIZE", 1000),
            chunk_overlap=_as_int("CHUNK_OVERLAP", 150),
        )

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required for chat completions")
        if not self.openai_base_url:
            errors.append("OPENAI_BASE_URL is required for the OpenAI-compatible chat API")
        if not self.openai_model:
            errors.append("OPENAI_MODEL is required")
        if not self.postgres_dsn:
            errors.append("POSTGRES_DSN is required for pgvector persistence")
        if self.embedding_dimension <= 0:
            errors.append("EMBEDDING_DIMENSION must be greater than 0")
        if self.rag_top_k <= 0:
            errors.append("RAG_TOP_K must be greater than 0")
        if self.chat_history_limit < 0:
            errors.append("CHAT_HISTORY_LIMIT must be greater than or equal to 0")
        if self.rerank_top_n < 0:
            errors.append("RERANK_TOP_N must be greater than or equal to 0")
        if self.chunk_size <= self.chunk_overlap:
            errors.append("CHUNK_SIZE must be greater than CHUNK_OVERLAP")
        if self.embedding_provider != "local":
            errors.append("EMBEDDING_PROVIDER must be local")
        if not self.local_embedding_model_path:
            errors.append("LOCAL_EMBEDDING_MODEL_PATH is required when EMBEDDING_PROVIDER=local")
        elif not Path(self.local_embedding_model_path).exists():
            errors.append(f"LOCAL_EMBEDDING_MODEL_PATH does not exist: {self.local_embedding_model_path}")
        if self.rerank_provider != "local":
            errors.append("RERANK_PROVIDER must be local")
        if self.rerank_top_n > 0 and self.rerank_provider == "local":
            if not self.local_rerank_model_path:
                errors.append("LOCAL_RERANK_MODEL_PATH is required when RERANK_PROVIDER=local")
            elif not Path(self.local_rerank_model_path).exists():
                errors.append(f"LOCAL_RERANK_MODEL_PATH does not exist: {self.local_rerank_model_path}")
            if self.local_rerank_max_length <= 0:
                errors.append("LOCAL_RERANK_MAX_LENGTH must be greater than 0")
            if self.local_rerank_batch_size <= 0:
                errors.append("LOCAL_RERANK_BATCH_SIZE must be greater than 0")
        return errors

    @property
    def sqlalchemy_postgres_dsn(self) -> str:
        if not self.postgres_dsn:
            raise ValueError("POSTGRES_DSN is required")
        if self.postgres_dsn.startswith("postgresql+"):
            return self.postgres_dsn
        if self.postgres_dsn.startswith("postgresql://"):
            return self.postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1)
        return self.postgres_dsn

    @property
    def psycopg_postgres_dsn(self) -> str:
        if not self.postgres_dsn:
            raise ValueError("POSTGRES_DSN is required")
        return self.postgres_dsn.replace("postgresql+psycopg://", "postgresql://", 1)

    @property
    def asyncpg_postgres_dsn(self) -> str:
        if not self.postgres_dsn:
            raise ValueError("POSTGRES_DSN is required")
        dsn = self.postgres_dsn
        if dsn.startswith("postgresql+psycopg://"):
            dsn = dsn.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
        elif dsn.startswith("postgresql://"):
            dsn = dsn.replace("postgresql://", "postgresql+asyncpg://", 1)

        parsed = urlsplit(dsn)
        query_items = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            if key == "sslmode" and value in {"disable", "allow", "prefer"}:
                continue
            if key == "sslmode" and value in {"require", "verify-ca", "verify-full"}:
                query_items.append(("ssl", "true"))
                continue
            query_items.append((key, value))
        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode(query_items),
                parsed.fragment,
            )
        )
