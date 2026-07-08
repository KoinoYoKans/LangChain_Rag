from __future__ import annotations

from dataclasses import dataclass
from math import log
from uuid import UUID

from langchain_core.documents import Document
from sqlalchemy import select

from config.database import ChunkLocationModel, RagFileChunkModel, RagFileModel, build_async_session_maker
from config.settings import AppSettings
from core.vector_store import similarity_search


@dataclass(frozen=True)
class RetrievalResult:
    documents: list[Document]
    rewritten_query: str


async def hybrid_search(
    settings: AppSettings,
    vector_store: object,
    query: str,
    *,
    top_k: int,
    user_id: str,
    knowledge_base_id: str,
) -> RetrievalResult:
    vector_docs = similarity_search(
        vector_store,
        query,
        top_k=max(top_k * 2, top_k),
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
    )
    bm25_docs = await _bm25_search(settings, query, knowledge_base_id=knowledge_base_id, limit=max(top_k * 2, top_k))
    return RetrievalResult(documents=_rrf_merge(vector_docs, bm25_docs, top_k=top_k), rewritten_query=query)


async def _bm25_search(settings: AppSettings, query: str, *, knowledge_base_id: str, limit: int) -> list[Document]:
    session_maker = build_async_session_maker(settings)
    async with session_maker() as session:
        rows = await session.execute(
            select(
                RagFileChunkModel,
                RagFileModel.filename,
                RagFileModel.content_type,
                RagFileModel.source_type,
                RagFileModel.source_uri,
                ChunkLocationModel.page_number,
                ChunkLocationModel.bbox,
            )
            .join(RagFileModel, RagFileChunkModel.file_id == RagFileModel.id)
            .outerjoin(ChunkLocationModel, RagFileChunkModel.id == ChunkLocationModel.chunk_id)
            .where(
                RagFileChunkModel.knowledge_base_id == UUID(knowledge_base_id),
                RagFileModel.status == "completed",
            )
            .order_by(RagFileChunkModel.created_at.desc())
            .limit(2000)
        )
        chunks = list(rows)
    if not chunks:
        return []

    tokenized = [_tokenize(chunk.content) for chunk, *_ in chunks]
    query_tokens = _tokenize(query)
    scores = _score_bm25(tokenized, query_tokens)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:limit]
    documents: list[Document] = []
    for index, score in ranked:
        if score <= 0:
            continue
        chunk, filename, content_type, source_type, source_uri, page_number, bbox = chunks[index]
        documents.append(
            Document(
                page_content=chunk.content,
                metadata={
                    "document_id": str(chunk.file_id),
                    "file_id": str(chunk.file_id),
                    "chunk_id": str(chunk.id),
                    "vector_id": str(chunk.vector_id),
                    "org_id": str(chunk.org_id) if chunk.org_id else None,
                    "knowledge_base_id": str(chunk.knowledge_base_id) if chunk.knowledge_base_id else None,
                    "user_id": chunk.user_id,
                    "filename": filename,
                    "content_type": content_type,
                    "source_type": source_type,
                    "source_uri": source_uri,
                    "chunk_index": chunk.chunk_index,
                    "keywords": list(chunk.keywords or []),
                    "page_number": page_number,
                    "bbox": dict(bbox or {}),
                    "bm25_score": float(score),
                },
            )
        )
    return documents


def _tokenize(text: str) -> list[str]:
    try:
        import jieba

        raw_tokens = jieba.lcut(text)
    except Exception:  # noqa: BLE001
        raw_tokens = text.split()
    return [token.strip().lower() for token in raw_tokens if len(token.strip()) > 1]


def _score_bm25(corpus: list[list[str]], query_tokens: list[str]) -> list[float]:
    if not corpus or not query_tokens:
        return [0.0 for _ in corpus]
    try:
        from rank_bm25 import BM25Okapi

        return [float(score) for score in BM25Okapi(corpus).get_scores(query_tokens)]
    except Exception:  # noqa: BLE001
        doc_count = len(corpus)
        document_frequencies: dict[str, int] = {}
        for tokens in corpus:
            for token in set(tokens):
                document_frequencies[token] = document_frequencies.get(token, 0) + 1
        scores: list[float] = []
        for tokens in corpus:
            score = 0.0
            for token in query_tokens:
                if token in tokens:
                    idf = log((doc_count + 1) / (document_frequencies.get(token, 0) + 0.5))
                    score += idf * tokens.count(token)
            scores.append(score)
        return scores


def _rrf_merge(vector_docs: list[Document], bm25_docs: list[Document], *, top_k: int) -> list[Document]:
    ranked: dict[str, tuple[Document, float]] = {}
    for source_name, documents in (("vector", vector_docs), ("bm25", bm25_docs)):
        for rank, document in enumerate(documents, start=1):
            key = str(document.metadata.get("chunk_id") or document.metadata.get("vector_id") or hash(document.page_content))
            existing_document, existing_score = ranked.get(key, (document, 0.0))
            score = existing_score + 1.0 / (60 + rank)
            metadata = dict(existing_document.metadata)
            metadata[f"{source_name}_rank"] = rank
            if source_name == "bm25" and document.metadata.get("bm25_score") is not None:
                metadata["bm25_score"] = document.metadata["bm25_score"]
                metadata["page_number"] = document.metadata.get("page_number")
                metadata["bbox"] = document.metadata.get("bbox")
            ranked[key] = (Document(page_content=existing_document.page_content, metadata=metadata), score)
    merged = []
    for document, score in ranked.values():
        metadata = dict(document.metadata)
        metadata["hybrid_score"] = score
        merged.append(Document(page_content=document.page_content, metadata=metadata))
    return sorted(merged, key=lambda document: float(document.metadata.get("hybrid_score") or 0.0), reverse=True)[:top_k]
