from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import AppSettings

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _read_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(page.strip() for page in pages if page.strip())


def _read_docx(data: bytes) -> str:
    from docx import Document as DocxDocument

    doc = DocxDocument(BytesIO(data))
    paragraphs = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text(filename: str, data: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported document type: {extension or '<none>'}. Supported: {supported}")
    if extension in {".txt", ".md"}:
        return _decode_text(data)
    if extension == ".pdf":
        return _read_pdf(data)
    if extension == ".docx":
        return _read_docx(data)
    raise ValueError(f"Unsupported document type: {extension}")


def build_chunks(
    filename: str,
    content_type: str | None,
    data: bytes,
    settings: AppSettings,
) -> tuple[str, list[Document]]:
    text = extract_text(filename, data).strip()
    if not text:
        raise ValueError("Document contains no extractable text")

    document_id = str(uuid4())
    upload_time = datetime.now(timezone.utc).isoformat()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "document_id": document_id,
                "filename": filename,
                "content_type": content_type or "application/octet-stream",
                "chunk_index": index,
                "upload_time": upload_time,
            },
        )
        for index, chunk in enumerate(chunks)
    ]
    return document_id, documents
