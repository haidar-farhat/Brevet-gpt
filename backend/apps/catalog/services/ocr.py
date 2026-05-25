"""Shared OCR helpers: turn a SCANNED PDF into a clean, structured (TOC'd) PDF via
``gpu_ocr_books``, plus small PDF metadata readers. Used by the ``ocr_embed``
command and the upload intake service so both share one code path.

Note: ``gpu_ocr_books.process_pdf`` writes to its own ``RESULTS_FOLDER`` (NOT
``settings.RESULTS_DIR``) and returns ``None`` — we resolve the deterministic
output path ourselves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz  # PyMuPDF
from django.conf import settings

# language code -> (tesseract lang, gpu_ocr_books results subdir, assets folder)
LANG_MAP: dict[str, tuple[str, str, str]] = {
    "en": ("eng", "eng", "english"),
    "fr": ("fra", "fr", "french"),
}


class OCRError(RuntimeError):
    """OCR is unavailable or failed (missing module, language pack, or output)."""


def load_ocr_module():
    """Import the repo-root ``gpu_ocr_books`` (lazy: heavy deps + Tesseract probe)."""
    root = str(settings.PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import gpu_ocr_books  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise OCRError(f"Could not import gpu_ocr_books from {root}: {exc}") from exc
    return gpu_ocr_books


def clean_pdf_path(input_path: Path, language: str):
    """Deterministic output path for a scanned PDF's clean version."""
    ocr = load_ocr_module()
    _tess, out_subdir, _iso = LANG_MAP[language]
    return Path(ocr.RESULTS_FOLDER) / out_subdir / f"{Path(input_path).stem}.pdf"


def ocr_to_clean_pdf(input_path: Path, language: str) -> Path:
    """OCR a scanned PDF into a clean, TOC'd PDF and return its path. If the
    expected output already exists (e.g. a dedup re-submit of the same file), reuse
    it instead of re-running the expensive OCR."""
    if language not in LANG_MAP:
        raise OCRError(f"Unsupported OCR language: {language}")
    ocr = load_ocr_module()
    tess_lang, out_subdir, iso = LANG_MAP[language]
    out = Path(ocr.RESULTS_FOLDER) / out_subdir / f"{Path(input_path).stem}.pdf"
    if out.is_file():
        return out  # already OCR'd — skip the work
    tessdata_arg, available = ocr.ensure_languages({tess_lang})
    if tess_lang not in available or "osd" not in available:
        raise OCRError(f"Tesseract language '{tess_lang}' is unavailable; cannot OCR.")
    ocr.process_pdf(str(input_path), tess_lang, tessdata_arg, out_subdir, iso)
    if not out.is_file():
        raise OCRError(f"Expected OCR output not found: {out}")
    return out


def pdf_title(pdf) -> str | None:
    try:
        with fitz.open(pdf) as doc:
            return doc.metadata.get("title")
    except Exception:  # pragma: no cover
        return None


def page_count(pdf) -> int | None:
    try:
        with fitz.open(pdf) as doc:
            return doc.page_count
    except Exception:  # pragma: no cover
        return None
