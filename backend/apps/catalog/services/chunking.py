"""Turn a clean, structured OCR'd PDF into retrieval chunks.

Chunks are section- and page-aware: the PDF's bookmark outline (reconstructed
TOC) yields a per-page heading breadcrumb, chunks never cross a section
boundary, and each chunk records the exact page span for citation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import fitz  # PyMuPDF

# Footer stamped by gpu_ocr_books.py, e.g. "p. 84" or "p. 84*" — not content.
_FOOTER = re.compile(r"^\s*p\.\s*\d+\*?\s*$")
# Sentence-ish boundary used to keep page provenance per segment.
_SEGMENT = re.compile(r"(?<=[.!?:])\s+")
_WHITESPACE = re.compile(r"\s+")

_encoder = None  # cached tiktoken encoder (lazy; optional dependency)


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_index: int
    page_start: int
    page_end: int
    heading_path: str
    text: str
    token_count: int


@dataclass(frozen=True, slots=True)
class _Unit:
    page: int
    crumb: str
    text: str
    tokens: int


def count_tokens(text: str) -> int:
    """Token count via tiktoken (cl100k_base, matches text-embedding-3); falls
    back to a word-based estimate when tiktoken is unavailable."""
    global _encoder
    if _encoder is None:
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover
            _encoder = False
    if _encoder:
        return len(_encoder.encode(text))
    return max(1, round(len(text.split()) * 1.3))


def _breadcrumbs_per_page(toc: list[list], n_pages: int) -> list[str]:
    """Map each 1-based page to its heading breadcrumb ('A > B > C')."""
    crumbs = [""] * (n_pages + 1)
    if not toc:
        return crumbs

    stack: list[tuple[int, str]] = []
    entries: list[tuple[int, str]] = []
    for level, title, page in toc:
        title = _WHITESPACE.sub(" ", title).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        entries.append((page, " > ".join(t for _, t in stack)))

    cursor, current = 0, ""
    for page in range(1, n_pages + 1):
        while cursor < len(entries) and entries[cursor][0] <= page:
            current = entries[cursor][1]
            cursor += 1
        crumbs[page] = current
    return crumbs


def _clean_page_text(raw: str) -> str:
    lines = [ln for ln in raw.splitlines() if not _FOOTER.match(ln.strip())]
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate across line breaks
    return _WHITESPACE.sub(" ", text).strip()


def _segments(text: str) -> list[str]:
    return [s.strip() for s in _SEGMENT.split(text) if s.strip()]


def _windows(units: list[_Unit], target: int, overlap: int) -> list[list[_Unit]]:
    """Sliding token-budget windows over units of one section, with overlap."""
    out: list[list[_Unit]] = []
    start, n = 0, len(units)
    while start < n:
        tokens, end = 0, start
        while end < n and tokens < target:
            tokens += units[end].tokens
            end += 1
        out.append(units[start:end])
        if end >= n:
            break
        back, cursor = 0, end
        while cursor > start + 1 and back < overlap:
            cursor -= 1
            back += units[cursor].tokens
        start = cursor  # cursor > start guarantees forward progress
    return out


def records_from_units(units: list[_Unit], target_tokens: int, overlap_tokens: int) -> list[ChunkRecord]:
    """Group units by section breadcrumb, window each section to the token budget,
    and emit ChunkRecords. Shared by ``chunk_pdf`` (scanned/native PDF) and the
    docx intake path so every input chunks identically."""
    records: list[ChunkRecord] = []
    index = 0
    for crumb, group in groupby(units, key=lambda u: u.crumb):
        section_units = list(group)
        for window in _windows(section_units, target_tokens, overlap_tokens):
            records.append(
                ChunkRecord(
                    chunk_index=index,
                    page_start=min(u.page for u in window),
                    page_end=max(u.page for u in window),
                    heading_path=crumb,
                    text=" ".join(u.text for u in window),
                    token_count=sum(u.tokens for u in window),
                )
            )
            index += 1
    return records


def text_to_units(page: int, crumb: str, text: str) -> list[_Unit]:
    """Sentence-segment a text block into units (for non-PDF intake, e.g. docx)."""
    return [_Unit(page, crumb, seg, count_tokens(seg)) for seg in _segments(text)]


def chunk_pdf(pdf_path: str | Path, target_tokens: int, overlap_tokens: int) -> list[ChunkRecord]:
    with fitz.open(pdf_path) as doc:
        n_pages = doc.page_count
        crumbs = _breadcrumbs_per_page(doc.get_toc(), n_pages)
        units: list[_Unit] = []
        for page in range(1, n_pages + 1):
            text = _clean_page_text(doc.load_page(page - 1).get_text("text"))
            if not text:
                continue
            crumb = crumbs[page]
            units.extend(text_to_units(page, crumb, text))

    return records_from_units(units, target_tokens, overlap_tokens)
