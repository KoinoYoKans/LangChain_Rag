from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4

from config.settings import AppSettings
from core.document_loader import build_chunks_from_text, hash_bytes, hash_text
from core.document_parser import locate_chunk, parse_document, parse_html
from core.document_store import delete_document_artifacts, replace_chunk_locations, replace_document_pages
from core.embeddings import build_embeddings
from core.enterprise_store import add_audit_log
from core.file_store import (
    create_processing_file,
    find_file_by_content_hash,
    delete_file_chunks_by_file_id,
    mark_file_completed,
    mark_file_deleted,
    mark_file_failed,
    save_file_chunks,
)
from core.ingest_store import (
    IngestJobCancelledError,
    get_ingest_job,
    mark_job_failed,
    mark_job_file_id,
    mark_job_progress,
    mark_job_running,
    mark_job_succeeded,
)
from core.vector_store import add_documents, build_vector_store, delete_documents, initialize_vector_extension


async def process_ingest_job(settings: AppSettings, job_id: str) -> None:
    job = await get_ingest_job(settings, job_id)
    if job is None:
        raise ValueError(f"Ingest job not found: {job_id}")
    if job.status == "cancelled":
        raise IngestJobCancelledError(f"Ingest job has been cancelled: {job_id}")
    await mark_job_running(settings, job_id)
    file_id = str(uuid4())
    vector_store = None
    ids: list[str] = []
    try:
        source = _load_job_source(job.payload, job.source_type, job.source_uri, job.filename)
        parsed = source["parsed"]
        content_sha256 = hash_text(parsed.text)
        planned_content_sha256 = str(job.payload.get("planned_content_sha256") or "")
        if planned_content_sha256 and planned_content_sha256 != content_sha256:
            raise ValueError("URL content changed since import plan; rerun dry-run before confirming")
        await mark_job_progress(settings, job_id, 25)
        await _raise_if_cancelled(settings, job_id)
        existing = await find_file_by_content_hash(
            settings,
            user_id=job.created_by_user_id,
            content_sha256=content_sha256,
            knowledge_base_id=job.knowledge_base_id,
        )
        if existing is not None:
            await mark_job_succeeded(settings, job_id, file_id=existing.id)
            return

        await create_processing_file(
            settings,
            file_id=file_id,
            org_id=job.org_id,
            knowledge_base_id=job.knowledge_base_id,
            owner_user_id=job.created_by_user_id,
            user_id=job.created_by_user_id,
            filename=source["filename"],
            content_type=source["content_type"],
            source_type=job.source_type,
            source_uri=source["source_uri"],
            file_size=len(source["raw_bytes"]),
            file_sha256=hash_bytes(source["raw_bytes"]),
            content_sha256=content_sha256,
        )
        await mark_job_file_id(settings, job_id, file_id)
        await _raise_if_cancelled(settings, job_id)
        _, documents = build_chunks_from_text(
            filename=source["filename"],
            content_type=source["content_type"],
            text=parsed.text,
            settings=settings,
            document_id=file_id,
            content_sha256=content_sha256,
            user_id=job.created_by_user_id,
            org_id=job.org_id,
            knowledge_base_id=job.knowledge_base_id,
            source_type=job.source_type,
            source_uri=source["source_uri"],
        )
        await mark_job_progress(settings, job_id, 45)
        await _raise_if_cancelled(settings, job_id)
        embeddings = build_embeddings(settings)
        initialize_vector_extension(settings)
        vector_store = build_vector_store(settings, embeddings, initialize_first=False)
        ids, indexed_documents = add_documents(vector_store, documents)
        await mark_job_progress(settings, job_id, 70)
        await _raise_if_cancelled(settings, job_id)
        await replace_document_pages(
            settings,
            org_id=job.org_id,
            knowledge_base_id=job.knowledge_base_id,
            file_id=file_id,
            pages=parsed.pages,
        )
        locations = []
        for index, document in enumerate(indexed_documents):
            location = locate_chunk(document.page_content, parsed.pages)
            locations.append(
                {
                    "chunk_id": ids[index],
                    "page_number": location["page_number"],
                    "bbox": location["bbox"],
                }
            )
        await replace_chunk_locations(
            settings,
            org_id=job.org_id,
            knowledge_base_id=job.knowledge_base_id,
            file_id=file_id,
            locations=locations,
        )
        await mark_job_progress(settings, job_id, 85)
        await _raise_if_cancelled(settings, job_id)
        await save_file_chunks(
            settings,
            user_id=job.created_by_user_id,
            org_id=job.org_id,
            knowledge_base_id=job.knowledge_base_id,
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
        await _raise_if_cancelled(settings, job_id)
        await mark_file_completed(settings, file_id, chunk_count=len(documents), chunk_ids=ids, vector_ids=ids)
        await _raise_if_cancelled(settings, job_id)
        await mark_job_succeeded(settings, job_id, file_id=file_id)
        await add_audit_log(
            settings,
            org_id=job.org_id,
            actor_user_id=job.created_by_user_id,
            action="ingest.succeeded",
            target_type="ingest_job",
            target_id=job_id,
            metadata={"file_id": file_id, "source_type": job.source_type},
        )
    except IngestJobCancelledError:
        if vector_store is not None and ids:
            try:
                delete_documents(vector_store, ids)
            except Exception:  # noqa: BLE001
                pass
        try:
            await delete_file_chunks_by_file_id(settings, file_id)
            await delete_document_artifacts(settings, file_id)
            await mark_file_deleted(settings, file_id)
        except Exception:  # noqa: BLE001
            pass
        await add_audit_log(
            settings,
            org_id=job.org_id,
            actor_user_id=job.created_by_user_id,
            action="ingest.cancelled",
            target_type="ingest_job",
            target_id=job_id,
            metadata={"file_id": file_id, "source_type": job.source_type},
        )
    except Exception as exc:  # noqa: BLE001
        if vector_store is not None and ids:
            try:
                delete_documents(vector_store, ids)
            except Exception:  # noqa: BLE001
                pass
        try:
            await delete_file_chunks_by_file_id(settings, file_id)
            await delete_document_artifacts(settings, file_id)
        except Exception:  # noqa: BLE001
            pass
        await mark_file_failed(settings, file_id, str(exc))
        await mark_job_failed(settings, job_id, str(exc))
        await add_audit_log(
            settings,
            org_id=job.org_id,
            actor_user_id=job.created_by_user_id,
            action="ingest.failed",
            target_type="ingest_job",
            target_id=job_id,
            metadata={"error": str(exc)[:1000]},
        )
        raise


def process_ingest_job_sync(settings: AppSettings, job_id: str) -> None:
    asyncio.run(process_ingest_job(settings, job_id))


async def _raise_if_cancelled(settings: AppSettings, job_id: str) -> None:
    job = await get_ingest_job(settings, job_id)
    if job and job.status == "cancelled":
        raise IngestJobCancelledError(f"Ingest job has been cancelled: {job_id}")


def _load_job_source(payload: dict[str, Any], source_type: str, source_uri: str | None, filename: str | None) -> dict[str, Any]:
    if source_type == "file":
        path = Path(str(payload["path"]))
        data = path.read_bytes()
        resolved_filename = filename or path.name
        parsed = parse_document(resolved_filename, data, str(payload.get("content_type") or "application/octet-stream"))
        return {
            "filename": resolved_filename,
            "content_type": parsed.content_type,
            "raw_bytes": data,
            "parsed": parsed,
            "source_uri": str(path),
        }
    if source_type == "url":
        if not source_uri:
            raise ValueError("URL ingest job requires source_uri")
        html = _fetch_url(source_uri)
        title = _extract_title(html)
        resolved_filename = filename or f"{_slugify(title or source_uri)}.html"
        parsed = parse_html(resolved_filename, html)
        raw = parsed.text.encode("utf-8")
        return {
            "filename": resolved_filename,
            "content_type": parsed.content_type,
            "raw_bytes": raw,
            "parsed": parsed,
            "source_uri": source_uri,
        }
    raise ValueError(f"Unsupported ingest source_type: {source_type}")


def _fetch_url(url: str, timeout_seconds: int = 20) -> str:
    request = Request(url, headers={"User-Agent": "LangChain-RagBot/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="ignore")


def _extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return (slug or "web-page")[:80]
