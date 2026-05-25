"""Smart intake: turn an uploaded PDF or .docx into ChunkRecords for ingestion.

Routing by file kind (decided here, not by the caller):
- ``.docx``                   -> extract text via python-docx (headings from styles)
- text-layer ("native") PDF   -> ``chunk_pdf`` directly (reads the embedded text)
- scanned (image-only) PDF    -> OCR via the shared ``ocr`` service, then ``chunk_pdf``

Emits coarse stage events through ``on_stage(stage, message)`` so the SSE upload
view can show progress. Reuses the existing chunking/OCR code — the only new
parsing is the docx path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from django.conf import settings

from apps.catalog.services import ocr as ocr_svc
from apps.catalog.services.chunking import (
    ChunkRecord,
    chunk_pdf,
    records_from_units,
    text_to_units,
)

_WS = re.compile(r"\s+")
# Word heading styles, EN + FR (e.g. "Heading 2", "Titre 2").
_HEADING_STYLE = re.compile(r"^(?:Heading|Titre)\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class IntakeMeta:
    kind: str  # "docx" | "native_pdf" | "scanned_pdf"
    total_pages: int | None
    pdf_path: str = ""  # citation PDF: clean OCR'd PDF (scanned) / the upload (native); "" for docx
    ocr_confidence: float | None = None


def _noop(*_a, **_k) -> None:
    pass


def build_records_for_upload(*, src_path, language: str, on_stage=None) -> tuple[list[ChunkRecord], IntakeMeta]:
    """Parse an uploaded file into ChunkRecords + metadata, choosing OCR vs direct
    text extraction automatically. ``src_path`` is a validated local file."""
    emit = on_stage or _noop
    path = Path(src_path)
    ext = path.suffix.lower()
    target, overlap = settings.EMBED_CHUNK_TOKENS, settings.EMBED_CHUNK_OVERLAP

    if ext == ".docx":
        emit("parse", "extracting text from the Word document")
        return _records_from_docx(path, target, overlap), IntakeMeta(kind="docx", total_pages=None)

    if ext == ".pdf":
        if _is_scanned_pdf(path):
            emit("ocr", "scanned PDF — running OCR (this can take a few minutes)")
            clean = ocr_svc.ocr_to_clean_pdf(path, language)
            emit("chunk", "chunking the OCR'd text")
            return chunk_pdf(clean, target, overlap), IntakeMeta(
                kind="scanned_pdf", total_pages=ocr_svc.page_count(clean), pdf_path=str(clean)
            )
        emit("parse", "extracting the PDF text layer")
        return chunk_pdf(path, target, overlap), IntakeMeta(
            kind="native_pdf", total_pages=_page_count(path), pdf_path=str(path)
        )

    raise ValueError(f"Unsupported file type: {ext or '(none)'}")


def _page_count(pdf: Path) -> int | None:
    try:
        with fitz.open(pdf) as doc:
            return doc.page_count
    except Exception:  # pragma: no cover
        return None


def _is_scanned_pdf(path: Path) -> bool:
    """Heuristic: an image-only (scanned) PDF has little/no extractable text. Sample
    the first pages and compare mean chars/page to a configurable threshold."""
    threshold = getattr(settings, "UPLOAD_SCANNED_TEXT_THRESHOLD", 100)
    try:
        with fitz.open(path) as doc:
            n = doc.page_count
            if n == 0:
                return False
            sample = min(n, 8)
            total = sum(len(doc.load_page(i).get_text("text").strip()) for i in range(sample))
            return (total / sample) < threshold
    except Exception:  # pragma: no cover
        return False  # undecidable -> treat as native (cheap path)


def _heading_level(para) -> int | None:
    name = (getattr(getattr(para, "style", None), "name", "") or "").strip()
    m = _HEADING_STYLE.match(name)
    if m:
        return int(m.group(1))
    if name.lower() in ("title", "titre"):
        return 1
    return None


def _records_from_docx(path: Path, target: int, overlap: int) -> list[ChunkRecord]:
    """Extract docx text directly (no OCR, no PDF conversion). Headings come from
    paragraph styles (Heading N / Titre N); body paragraphs become text units under
    the current heading breadcrumb. docx has no pages, so page is a constant 1."""
    from docx import Document  # python-docx (lazy import)

    document = Document(str(path))
    stack: list[tuple[int, str]] = []  # (level, title) heading stack -> breadcrumb
    units = []
    for para in document.paragraphs:
        text = _WS.sub(" ", (para.text or "")).strip()
        if not text:
            continue
        level = _heading_level(para)
        if level:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            continue  # the heading line itself is not a content unit
        crumb = " > ".join(t for _, t in stack)
        units.extend(text_to_units(1, crumb, text))
    return records_from_units(units, target, overlap)
