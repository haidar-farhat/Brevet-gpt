"""Orchestrates one book's ingestion: chunk -> embed -> Chroma + MySQL.

The MySQL ``Chunk`` row is the system of record; the Chroma entry holds the
vector plus a routing/citation payload. They are linked by a deterministic
``vector_id`` (uuid5 of book id + chunk index), which makes re-ingestion
idempotent and lets either store be rebuilt from the other.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass

from django.conf import settings
from django.db import transaction

from apps.catalog.models import Book, Chunk
from apps.catalog.services.chunking import ChunkRecord, chunk_pdf

# Fixed namespace so vector ids are stable across runs/machines.
_NAMESPACE = uuid.UUID("6b3d2f8c-1c2a-4e5b-9a7d-2f0c1e4a5b6c")


@dataclass(frozen=True, slots=True)
class IngestResult:
    chunks: int
    tokens: int
    sample: ChunkRecord | None


def _vector_id(book_id: int, chunk_index: int) -> uuid.UUID:
    return uuid.uuid5(_NAMESPACE, f"{book_id}:{chunk_index}")


def _contextual_prefix(book: Book, record: ChunkRecord) -> str:
    """Hierarchical context prepended to the embedded text (improves recall and
    disambiguates the same topic across subjects)."""
    span = f"p.{record.page_start}-{record.page_end}"
    head = f" • {record.heading_path}" if record.heading_path else ""
    return f"[{book.language}] {book.subject.name_en}{head} ({span})\n"


def build_records(pdf_path) -> list[ChunkRecord]:
    return chunk_pdf(pdf_path, settings.EMBED_CHUNK_TOKENS, settings.EMBED_CHUNK_OVERLAP)


def document_hash(records: list[ChunkRecord]) -> str:
    """Stable SHA-256 over a document's chunk texts — the book-level dedup key.
    Used at ingest and to backfill existing books so a re-upload hashes identically."""
    joined = "\n".join(r.text for r in records)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def ingest_book(*, book: Book, records: list[ChunkRecord], embedder, collection,
                dry_run: bool = False) -> IngestResult:
    """Embed pre-built ``records`` and replace the book's chunks/vectors. Callers
    build records via ``build_records(pdf)`` (clean OCR'd PDF) or the upload intake
    service (docx / native PDF)."""
    total_tokens = sum(r.token_count for r in records)
    sample = records[len(records) // 2] if records else None

    if dry_run or not records:
        return IngestResult(chunks=len(records), tokens=total_tokens, sample=sample)

    embed_inputs = [_contextual_prefix(book, r) + r.text for r in records]
    embeddings = embedder.embed(embed_inputs)

    ids, documents, metadatas, rows = [], [], [], []
    for record, vector in zip(records, embeddings):
        vector_id = _vector_id(book.id, record.chunk_index)
        ids.append(str(vector_id))
        documents.append(record.text)
        metadatas.append(
            {
                "book_id": book.id,
                "source_file": book.source_file,
                "title": book.title,
                "language": book.language,
                "subject": book.subject.code,
                "subject_name": book.subject.name_en,
                "level": book.level,
                "page_start": record.page_start,
                "page_end": record.page_end,
                "heading_path": record.heading_path or "",
                "chunk_index": record.chunk_index,
                "token_count": record.token_count,
            }
        )
        rows.append(
            Chunk(
                book=book,
                language=book.language,
                subject=book.subject,
                chunk_index=record.chunk_index,
                page_start=record.page_start,
                page_end=record.page_end,
                heading_path=record.heading_path or "",
                token_count=record.token_count,
                content=record.text,
                content_hash=hashlib.sha256(record.text.encode("utf-8")).hexdigest(),
                vector_id=vector_id,
            )
        )

    # Re-ingestion is a clean replace, keeping both stores in lockstep.
    with transaction.atomic():
        Chunk.objects.filter(book=book).delete()
        Chunk.objects.bulk_create(rows)
    collection.delete(where={"book_id": book.id})
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    return IngestResult(chunks=len(records), tokens=total_tokens, sample=sample)
