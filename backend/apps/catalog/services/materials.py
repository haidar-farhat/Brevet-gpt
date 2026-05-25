"""Manage-Materials orchestration: validate an upload, detect duplicates,
parse + embed it (reusing intake + ingest), and freeze/unfreeze/delete books.

One ingest runs at a time (a process-wide non-blocking lock) to protect the CPU
and the shared embedder / Chroma singletons. The heavy ``run_upload`` is sync and
is meant to be called via ``asyncio.to_thread`` from the SSE view, which passes a
thread-safe ``on_stage`` callback for progress.
"""
from __future__ import annotations

import re
import threading
import unicodedata
import uuid
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from apps.catalog.enums import BookStatus
from apps.catalog.models import Book, Subject
from apps.catalog.services import intake
from apps.catalog.services import ocr as ocr_svc
from apps.catalog.services.ingest import document_hash, ingest_book

_ALLOWED_EXT = {".pdf", ".docx"}
_PDF_MAGIC = b"%PDF"
_ZIP_MAGIC = b"PK\x03\x04"  # .docx is a zip container

_INGEST_LOCK = threading.Lock()


class UploadError(ValueError):
    """A user-facing upload problem (bad file, busy, nothing extractable, …)."""


class NeedsDecision(Exception):
    """Raised mid-upload when a duplicate is found and the client hasn't chosen a
    resolution. Carries the matched-book info for the UI's choice modal."""

    def __init__(self, match: dict) -> None:
        super().__init__("duplicate — needs a resolution")
        self.match = match


def ingest_in_progress() -> bool:
    return _INGEST_LOCK.locked()


# --- shared singletons (never load bge-m3 twice — Windows segfault risk) ----
def _embedder_and_collection():
    from apps.rag.services.pipeline import get_retriever

    retriever = get_retriever()
    return retriever.embedder, retriever.collection


# --- validation / storage ---------------------------------------------------
def sanitize_filename(name: str) -> str:
    base = Path(name or "").name  # drop any directory components (path traversal)
    base = unicodedata.normalize("NFKD", base)
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "upload"
    return base[:128]


def validate_upload(filename: str, size: int, head: bytes) -> str:
    """Return the normalized extension, or raise UploadError."""
    ext = Path(filename or "").suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise UploadError(f"Unsupported file type '{ext or '?'}'. Allowed: PDF or Word (.docx).")
    if size <= 0:
        raise UploadError("The file is empty.")
    max_mb = getattr(settings, "MAX_UPLOAD_MB", 100)
    if size > max_mb * 1024 * 1024:
        raise UploadError(f"File too large ({size // (1024 * 1024)} MB); the limit is {max_mb} MB.")
    if ext == ".pdf" and not head.startswith(_PDF_MAGIC):
        raise UploadError("This file is not a valid PDF.")
    if ext == ".docx" and not head.startswith(_ZIP_MAGIC):
        raise UploadError("This file is not a valid Word (.docx) document.")
    return ext


def _lang_folder(language: str) -> str:
    """Uploads/corpus subfolder for a language code, from the Language registry."""
    _tess, _subdir, folder = ocr_svc.lang_map().get(language, ("", language, language))
    return folder or language or "other"


def save_upload(language: str, filename: str, chunks) -> Path:
    """Stream the uploaded file to UPLOADS_DIR/{lang}/ and return its path."""
    folder = Path(settings.UPLOADS_DIR) / _lang_folder(language)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / sanitize_filename(filename)
    with open(dest, "wb") as fh:
        for chunk in chunks:
            fh.write(chunk)
    return dest


# --- dedup ------------------------------------------------------------------
def _match_dict(book: Book, reason: str) -> dict:
    return {
        "book_id": book.id,
        "title": book.title,
        "status": book.status,
        "subject": book.subject.code,
        "language": book.language,
        "chunks": book.chunks.count(),
        "match_reason": reason,
    }


def precheck(*, title: str, language: str, subject_code: str, content_hash: str | None = None) -> dict | None:
    """Find a likely-duplicate book: same (subject+title+language), or (if given)
    an identical document content hash. Returns match info or None."""
    qs = Book.objects.select_related("subject")
    meta = qs.filter(
        language=language, subject__code=subject_code, title__iexact=(title or "").strip()
    ).first()
    if meta:
        return _match_dict(meta, "metadata")
    if content_hash:
        same = qs.filter(content_hash=content_hash).first()
        if same:
            return _match_dict(same, "content")
    return None


# --- browse -----------------------------------------------------------------
def list_books(*, status=None, subject=None, language=None, q=None) -> list[dict]:
    qs = Book.objects.select_related("subject", "replaces").all()
    if status:
        qs = qs.filter(status=status)
    if subject:
        qs = qs.filter(subject__code=subject)
    if language:
        qs = qs.filter(language=language)
    if q:
        qs = qs.filter(title__icontains=q)
    return [
        {
            "id": b.id,
            "title": b.title,
            "language": b.language,
            "subject": b.subject.code,
            "level": b.level,
            "status": b.status,
            "total_pages": b.total_pages,
            "chunks": b.chunks.count(),
            "content_hash": b.content_hash[:12],
            "replaces": b.replaces_id,
            "processed_at": b.processed_at.isoformat() if b.processed_at else None,
        }
        for b in qs
    ]


def book_detail(book_id: int, *, offset: int = 0, limit: int = 50) -> dict:
    book = Book.objects.select_related("subject").get(pk=book_id)
    total = book.chunks.count()
    chunks = (
        book.chunks.order_by("chunk_index")
        .values("chunk_index", "page_start", "page_end", "heading_path", "token_count", "content")[
            offset : offset + limit
        ]
    )
    items = [{**c, "content": c["content"][:600]} for c in chunks]
    return {
        "id": book.id, "title": book.title, "language": book.language,
        "subject": book.subject.code, "level": book.level, "status": book.status,
        "total_pages": book.total_pages, "chunk_count": total,
        "offset": offset, "limit": limit, "chunks": items,
    }


# --- freeze / delete --------------------------------------------------------
def set_status(book_id: int, status: str) -> dict:
    if status not in BookStatus.values:
        raise UploadError(f"Invalid status '{status}'.")
    book = Book.objects.get(pk=book_id)
    book.status = status
    book.save(update_fields=["status", "updated_at"])
    return {"id": book.id, "status": book.status}


def delete_book(book_id: int) -> dict:
    """Hard delete: remove vectors, DB rows (cascades chunks/sections), and files."""
    _, collection = _embedder_and_collection()
    book = Book.objects.get(pk=book_id)
    try:
        collection.delete(where={"book_id": book.id})
    except Exception:
        pass  # vectors may already be gone; the DB delete is the source of truth
    for p in {book.pdf_path, _upload_path(book)}:
        if p:
            Path(p).unlink(missing_ok=True)
    book.delete()
    return {"id": book_id, "deleted": True}


def _upload_path(book: Book) -> str | None:
    try:
        return str(Path(settings.UPLOADS_DIR) / book.source_file)
    except Exception:
        return None


# --- the upload driver (sync; run via asyncio.to_thread) --------------------
def _noop(*_a, **_k) -> None:
    pass


def run_upload(*, src_path, language: str, subject_code: str, title: str, level: str = "brevet",
               resolution: str | None = None, target_id: int | None = None, on_stage=_noop) -> dict:
    """Parse + embed an already-saved upload. Raises NeedsDecision (duplicate) or
    UploadError. Returns a summary dict on success. One ingest at a time."""
    if not _INGEST_LOCK.acquire(blocking=False):
        raise UploadError("Another upload is in progress — please wait for it to finish.")
    try:
        on_stage("validate", "preparing")
        subject = Subject.objects.get(code=subject_code)
        title = (title or "").strip() or Path(src_path).stem

        # 1) cheap metadata dedup — BEFORE any OCR
        if not resolution:
            match = precheck(title=title, language=language, subject_code=subject_code)
            if match:
                raise NeedsDecision(match)

        # 2) parse / OCR / chunk
        records, meta = intake.build_records_for_upload(
            src_path=src_path, language=language, on_stage=on_stage
        )
        if not records:
            raise UploadError("No extractable text was found in the document.")
        content_hash = document_hash(records)

        # 3) content dedup — now that we know the text (cheap for docx/native)
        if not resolution:
            match = precheck(
                title=title, language=language, subject_code=subject_code, content_hash=content_hash
            )
            if match:
                raise NeedsDecision(match)

        on_stage("embed", f"embedding {len(records)} chunks")
        embedder, collection = _embedder_and_collection()
        book = _resolve_target_book(
            resolution=resolution, target_id=target_id, language=language,
            filename=Path(src_path).name, title=title, subject=subject, level=level,
            meta=meta, content_hash=content_hash,
        )
        result = ingest_book(book=book, records=records, embedder=embedder, collection=collection)
        on_stage("store", "saved")
        return {
            "book_id": book.id, "title": book.title, "status": book.status,
            "chunks": result.chunks, "tokens": result.tokens, "kind": meta.kind,
        }
    finally:
        _INGEST_LOCK.release()


def _resolve_target_book(*, resolution, target_id, language, filename, title, subject, level,
                         meta, content_hash) -> Book:
    source_file = f"{_lang_folder(language)}/{sanitize_filename(filename)}"
    defaults = {
        "title": title, "subject": subject, "level": level,
        "pdf_path": meta.pdf_path, "total_pages": meta.total_pages,
        "content_hash": content_hash, "status": BookStatus.ACTIVE,
        "processed_at": timezone.now(),
    }
    if resolution == "update" and target_id:
        book = Book.objects.get(pk=target_id)
        for key, value in defaults.items():
            setattr(book, key, value)
        book.save()
        return book
    if resolution == "freeze_replace" and target_id:
        old = Book.objects.get(pk=target_id)
        Book.objects.filter(pk=old.pk).update(status=BookStatus.FROZEN)
        unique = f"{source_file}#{uuid.uuid4().hex[:8]}"  # avoid the (language, source_file) clash
        return Book.objects.create(language=language, source_file=unique, replaces=old, **defaults)
    # "new" (or no match): upsert by (language, source_file)
    book, _ = Book.objects.update_or_create(
        language=language, source_file=source_file, defaults=defaults
    )
    return book
