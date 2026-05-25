"""Pure helpers for deriving book metadata from PDF titles / filenames.

No Django imports → trivially unit-testable in isolation.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from apps.catalog.data.subjects import SUBJECTS
from apps.catalog.enums import SubjectCode

_PDF_SUFFIX = re.compile(r"(?i)\.pdf$")
_PAGES_SUFFIX = re.compile(r"(?i)\s*[-–]\s*pages?\s*\d+\s*[-–]\s*\d+\s*$")
_WHITESPACE = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def normalize(text: str) -> str:
    """Lowercase + accent-fold for robust, locale-agnostic substring matching."""
    return _strip_accents(text).casefold().strip()


def prettify_stem(filename: str) -> str:
    stem = Path(filename).stem.replace("_", " ").strip()
    return stem.title() if stem.islower() else stem


def clean_title(meta_title: str | None, filename: str) -> str:
    """Human-friendly title: strip the scanner's ' - Pages X-Y.pdf' suffix.

    Falls back to a prettified filename when the embedded title is empty or
    contains U+FFFD (a genuine decode failure, as opposed to a valid accent).
    """
    title = (meta_title or "").strip()
    title = _PDF_SUFFIX.sub("", title).strip()
    title = _PAGES_SUFFIX.sub("", title).strip(" -–")
    if not title or "�" in title:
        title = prettify_stem(filename)
    return _WHITESPACE.sub(" ", title).strip()


def infer_subject_code(title: str, filename: str) -> SubjectCode | None:
    """Map a book onto a SubjectCode via the taxonomy aliases (priority order)."""
    haystack = normalize(f"{title} {filename}")
    for subject in SUBJECTS:
        if any(normalize(alias) in haystack for alias in subject.aliases):
            return subject.code
    return None
