from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile, ZipFile, is_zipfile

from fastapi import HTTPException, UploadFile


READ_CHUNK_SIZE = 1024 * 1024


async def read_upload_bytes(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an upload with a hard byte limit, including chunked requests."""
    parts: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(READ_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds the {max_bytes} byte limit",
            )
        parts.append(chunk)
    return b"".join(parts)


def sanitize_filename(filename: str, *, fallback: str = "uploaded", max_length: int = 180) -> str:
    value = filename.replace("\\", "/").split("/")[-1].strip().replace("\x00", "")
    value = value or fallback
    suffix = Path(value).suffix[:16]
    stem = Path(value).stem[: max_length - len(suffix)]
    return f"{stem}{suffix}" or fallback


def validate_document_bytes(filename: str, data: bytes, max_expanded_bytes: int) -> None:
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="File content is not a valid PDF")
        return
    if extension == ".docx":
        _validate_docx_archive(data, max_expanded_bytes)
        return
    if extension in {".txt", ".md", ".html"} and b"\x00" in data[:8192]:
        raise HTTPException(status_code=400, detail="Text document contains binary content")


def _validate_docx_archive(data: bytes, max_expanded_bytes: int) -> None:
    if not is_zipfile(data):
        raise HTTPException(status_code=400, detail="File content is not a valid DOCX archive")
    try:
        with ZipFile(io_bytes(data)) as archive:
            entries = archive.infolist()
            total_expanded = sum(max(0, item.file_size) for item in entries)
            if len(entries) > 10_000 or total_expanded > max_expanded_bytes:
                raise HTTPException(status_code=400, detail="DOCX archive exceeds safe extraction limits")
            for item in entries:
                if item.is_dir() or item.file_size == 0:
                    continue
                if item.compress_size == 0 or item.file_size / item.compress_size > 150:
                    raise HTTPException(status_code=400, detail="DOCX archive has an unsafe compression ratio")
            if "[Content_Types].xml" not in archive.namelist() or "word/document.xml" not in archive.namelist():
                raise HTTPException(status_code=400, detail="DOCX archive is missing required document parts")
    except BadZipFile as exc:
        raise HTTPException(status_code=400, detail="File content is not a valid DOCX archive") from exc


def io_bytes(data: bytes):
    from io import BytesIO

    return BytesIO(data)
