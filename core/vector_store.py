from __future__ import annotations

import uuid

import psycopg
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore
from sqlalchemy.exc import ProgrammingError

from config.settings import AppSettings


def check_database(settings: AppSettings) -> None:
    with psycopg.connect(settings.psycopg_postgres_dsn, connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            cursor.execute("select 1")
            cursor.fetchone()


def build_vector_store(settings: AppSettings, embeddings: Embeddings) -> PGVectorStore:
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


def add_documents(vector_store: PGVectorStore, documents: list[Document]) -> list[str]:
    ids = [
        f"{document.metadata['document_id']}:{document.metadata['chunk_index']}:{uuid.uuid4().hex[:8]}"
        for document in documents
    ]
    vector_store.add_documents(documents, ids=ids)
    return ids


def similarity_search(vector_store: PGVectorStore, query: str, top_k: int) -> list[Document]:
    if hasattr(vector_store, "similarity_search_with_score"):
        docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
        documents: list[Document] = []
        for document, score in docs_with_scores:
            metadata = dict(document.metadata)
            metadata["vector_score"] = float(score)
            documents.append(Document(page_content=document.page_content, metadata=metadata))
        return documents
    return vector_store.similarity_search(query, k=top_k)
