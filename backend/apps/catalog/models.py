"""Catalog models — the relational system of record for the corpus.

Design notes:
* Routing dimensions (``language``, ``subject``) live here and are mirrored into
  the vector store's payload at ingest time.
* Chunk text and citation data live here so the vector DB never becomes the
  source of truth (it only holds embeddings keyed by ``Chunk.vector_id``).
* ``Chunk.language``/``Chunk.subject`` are deliberately denormalised from
  ``Book`` to keep retrieval-time filtering cheap and to match the vector payload.
"""
from __future__ import annotations

from django.core.validators import MinValueValidator
from django.db import models

from apps.catalog.enums import Language, SubjectCode


class TimeStampedModel(models.Model):
    """Abstract base adding self-managing audit timestamps."""

    created_at = models.DateTimeField(auto_now_add=True, editable=False)
    updated_at = models.DateTimeField(auto_now=True, editable=False)

    class Meta:
        abstract = True


class Subject(TimeStampedModel):
    """A routable subject (Mathematics, Physics, …)."""

    code = models.CharField(max_length=32, choices=SubjectCode.choices, unique=True)
    name_en = models.CharField(max_length=128)
    name_fr = models.CharField(max_length=128)
    aliases = models.JSONField(
        default=list,
        blank=True,
        help_text="Accent-folded, lowercase keywords used to route queries to this subject.",
    )

    class Meta:
        db_table = "subjects"
        ordering = ("code",)
        verbose_name = "subject"
        verbose_name_plural = "subjects"

    def __str__(self) -> str:
        return self.get_code_display()


class Book(TimeStampedModel):
    """One scanned textbook (one source PDF)."""

    title = models.CharField(max_length=255)
    language = models.CharField(max_length=2, choices=Language.choices)
    subject = models.ForeignKey(Subject, on_delete=models.PROTECT, related_name="books")
    level = models.CharField(max_length=32, default="brevet")

    source_file = models.CharField(
        max_length=512,
        help_text="Path relative to ASSETS_DIR, e.g. 'french/bio.pdf'.",
    )
    pdf_path = models.CharField(
        max_length=512,
        blank=True,
        help_text="Absolute path to the processed, structured PDF (citation source).",
    )
    total_pages = models.PositiveIntegerField(null=True, blank=True)
    ocr_confidence = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Mean OCR confidence (0–100); a corpus quality signal.",
    )
    processed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "books"
        ordering = ("language", "title")
        constraints = [
            models.UniqueConstraint(
                fields=("language", "source_file"),
                name="uq_book_language_source",
            ),
        ]
        indexes = [
            models.Index(fields=("language", "subject"), name="ix_book_routing"),
        ]

    def __str__(self) -> str:
        return f"{self.title} ({self.language})"


class Section(TimeStampedModel):
    """A node in a book's reconstructed table of contents (hierarchical context)."""

    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name="sections")
    parent = models.ForeignKey(
        "self",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="children",
    )
    level = models.PositiveSmallIntegerField(default=1)
    title = models.CharField(max_length=512)
    path = models.CharField(
        max_length=1024,
        blank=True,
        help_text="Breadcrumb, e.g. 'Optics > The Eye > Reduced eye'.",
    )
    page_start = models.PositiveIntegerField(null=True, blank=True)
    page_end = models.PositiveIntegerField(null=True, blank=True)
    ordinal = models.PositiveIntegerField(default=0, help_text="Order within the book.")

    class Meta:
        db_table = "sections"
        ordering = ("book", "ordinal")
        indexes = [
            models.Index(fields=("book", "ordinal"), name="ix_section_order"),
        ]

    def __str__(self) -> str:
        return self.path or self.title


class Chunk(TimeStampedModel):
    """A retrieval unit. ``vector_id`` joins to the embedding in the vector store."""

    book = models.ForeignKey(Book, on_delete=models.CASCADE, related_name="chunks")
    section = models.ForeignKey(
        Section,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="chunks",
    )
    language = models.CharField(max_length=2, choices=Language.choices)
    subject = models.ForeignKey(Subject, on_delete=models.PROTECT, related_name="chunks")

    chunk_index = models.PositiveIntegerField(help_text="Order within the book.")
    page_start = models.PositiveIntegerField(validators=[MinValueValidator(1)])
    page_end = models.PositiveIntegerField(validators=[MinValueValidator(1)])
    heading_path = models.CharField(max_length=1024, blank=True)
    token_count = models.PositiveIntegerField(null=True, blank=True)
    ocr_confidence = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)

    content = models.TextField()
    content_hash = models.CharField(
        max_length=64,
        blank=True,
        db_index=True,
        help_text="SHA-256 of content for idempotent re-ingest / dedup.",
    )
    vector_id = models.UUIDField(
        null=True,
        blank=True,
        unique=True,
        help_text="Identifier of the embedding in the vector store.",
    )

    class Meta:
        db_table = "chunks"
        ordering = ("book", "chunk_index")
        constraints = [
            models.UniqueConstraint(fields=("book", "chunk_index"), name="uq_chunk_book_index"),
        ]
        indexes = [
            models.Index(fields=("language", "subject"), name="ix_chunk_routing"),
            models.Index(fields=("book",), name="ix_chunk_book"),
        ]

    def __str__(self) -> str:
        return f"{self.book.title} #{self.chunk_index} (p.{self.page_start}-{self.page_end})"
