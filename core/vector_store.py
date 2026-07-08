from __future__ import annotations

import asyncio
import sys
import uuid

import psycopg
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore
from sqlalchemy.exc import ProgrammingError

from config.database import initialize_orm_tables
from config.settings import AppSettings


def check_database(settings: AppSettings) -> None:
    with psycopg.connect(settings.psycopg_postgres_dsn, connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            cursor.execute("select 1")
            cursor.fetchone()


def initialize_vector_extension(settings: AppSettings) -> None:
    with psycopg.connect(settings.psycopg_postgres_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("create extension if not exists vector")


def initialize_database(settings: AppSettings) -> None:
    initialize_vector_extension(settings)
    asyncio.run(initialize_orm_tables(settings))


async def initialize_database_async(settings: AppSettings) -> None:
    initialize_vector_extension(settings)
    await initialize_orm_tables(settings)


def build_vector_store(
    settings: AppSettings,
    embeddings: Embeddings,
    initialize_first: bool = True,
) -> PGVectorStore:
    configure_windows_event_loop()
    if initialize_first:
        initialize_database(settings)
    engine = PGEngine.from_connection_string(url=settings.sqlalchemy_postgres_dsn)
    try:
        engine.init_vectorstore_table(
            table_name=settings.pgvector_table,
            vector_size=settings.embedding_dimension,
        )
    except ProgrammingError as exc:
        message = str(exc).lower()
        if "already exists" not in message and "duplicate" not in message:
            raise
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=settings.pgvector_table,
        embedding_service=embeddings,
    )


def configure_windows_event_loop() -> None:
    if sys.platform != "win32":
        return
    selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if selector_policy is not None:
        asyncio.set_event_loop_policy(selector_policy())


configure_windows_event_loop()


def add_documents(vector_store: PGVectorStore, documents: list[Document]) -> tuple[list[str], list[Document]]:
    ids = [str(uuid.uuid4()) for _ in documents]
    enriched_documents = [
        Document(
            page_content=document.page_content,
            metadata={
                **document.metadata,
                "chunk_id": document_id,
                "vector_id": document_id,
            },
        )
        for document, document_id in zip(documents, ids, strict=True)
    ]
    vector_store.add_documents(enriched_documents, ids=ids)
    return ids, enriched_documents


def delete_documents(vector_store: PGVectorStore, ids: list[str]) -> None:
    if not ids:
        return
    delete = getattr(vector_store, "delete", None)
    if delete is None:
        raise RuntimeError("Current vector store does not support deleting documents by id")
    delete(ids=ids)


def similarity_search(
    vector_store: PGVectorStore,
    query: str,
    top_k: int,
    user_id: str,
) -> list[Document]:
    metadata_filter = {"user_id": user_id}
    if hasattr(vector_store, "similarity_search_with_score"):
        docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k, filter=metadata_filter)
        documents: list[Document] = []
        for document, score in docs_with_scores:
            metadata = dict(document.metadata)
            metadata["vector_score"] = float(score)
            documents.append(Document(page_content=document.page_content, metadata=metadata))
        return documents
    return vector_store.similarity_search(query, k=top_k, filter=metadata_filter)
