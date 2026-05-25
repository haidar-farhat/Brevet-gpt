"""Shared OCR helpers: turn a SCANNED PDF into a clean, structured (TOC'd) PDF via
``gpu_ocr_books``, plus small PDF metadata readers. Used by the ``ocr_embed``
command and the upload intake service so both share one code path.

Note: ``gpu_ocr_books.process_pdf`` writes to its own ``RESULTS_FOLDER`` (NOT
``settings.RESULTS_DIR``) and returns ``None`` — we resolve the deterministic
output path ourselves.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
from django.conf import settings

# Built-in fallback, used only if the Language registry is empty/unavailable.
# code -> (tesseract lang, gpu_ocr_books results subdir, assets folder)
_FALLBACK_LANG_MAP: dict[str, tuple[str, str, str]] = {
    "en": ("eng", "eng", "english"),
    "fr": ("fra", "fr", "french"),
    "ar": ("ara", "ar", "arabic"),
}


def lang_map() -> dict[str, tuple[str, str, str]]:
    """code -> (tesseract, ocr_subdir, assets_folder), sourced from the Language
    registry so newly-added languages work; falls back to the built-ins when the
    table is empty or the DB is unavailable."""
    try:
        from apps.catalog.models import Language

        rows = {
            lng.code: (
                lng.tesseract or _FALLBACK_LANG_MAP.get(lng.code, ("", "", ""))[0],
                lng.ocr_subdir or lng.code,
                lng.assets_folder or lng.code,
            )
            for lng in Language.objects.filter(enabled=True)
        }
        return rows or dict(_FALLBACK_LANG_MAP)
    except Exception:
        return dict(_FALLBACK_LANG_MAP)


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
    # Redirect OCR output into the project's results dir. The standalone module hard-
    # codes its own external ``BOOKS_FOLDER/results`` (e.g. C:\Users\...\Documents\
    # books\results); point it at ``settings.RESULTS_DIR`` so OCR'd PDFs live inside
    # the project (alongside assets/), consistent with where uploads are now stored.
    gpu_ocr_books.RESULTS_FOLDER = str(settings.RESULTS_DIR)
    return gpu_ocr_books


def clean_pdf_path(input_path: Path, language: str):
    """Deterministic output path for a scanned PDF's clean version."""
    ocr = load_ocr_module()
    _tess, out_subdir, _iso = lang_map().get(language, _FALLBACK_LANG_MAP.get(language, ("", language, language)))
    return Path(ocr.RESULTS_FOLDER) / out_subdir / f"{Path(input_path).stem}.pdf"


def ocr_to_clean_pdf(input_path: Path, language: str) -> Path:
    """OCR a scanned PDF into a clean, TOC'd PDF and return its path. If the
    expected output already exists (e.g. a dedup re-submit of the same file), reuse
    it instead of re-running the expensive OCR."""
    m = lang_map()
    if language not in m:
        raise OCRError(f"Unsupported OCR language: {language}")
    ocr = load_ocr_module()
    tess_lang, out_subdir, iso = m[language]
    out = Path(ocr.RESULTS_FOLDER) / out_subdir / f"{Path(input_path).stem}.pdf"
    if out.is_file():
        return out  # already OCR'd — skip the work
    tessdata_arg, available = ocr.ensure_languages({tess_lang})
    if tess_lang not in available or "osd" not in available:
        raise OCRError(f"Tesseract language '{tess_lang}' is unavailable; cannot OCR.")
    # process_pdf writes straight to this path but does NOT create the folder (only
    # gpu_ocr_books' batch path does). For a newly-added language subdir (e.g. 'ar')
    # it won't exist yet, so create it here or the canvas write FileNotFoundErrors.
    out.parent.mkdir(parents=True, exist_ok=True)
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


def read_ocr_sidecar(clean_pdf):
    """Return the OCR logical-text sidecar written next to the clean PDF by
    gpu_ocr_books — ``[{"number": int|None, "paras": [{"text": str, "level": int}]}]`` —
    or None when absent (older output / non-OCR PDF). The ingest pipeline embeds THIS
    clean logical text instead of re-extracting it from the rendered PDF, whose bidi-
    reordered text layer corrupts Arabic / mixed-RTL content."""
    p = Path(clean_pdf)
    sidecar = p.parent / (p.stem + ".ocr.json")
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover
        return None
    return data.get("pages") or None
