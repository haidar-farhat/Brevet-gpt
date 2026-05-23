"""Embed the already-OCR'd clean PDFs (RESULTS_DIR/eng, RESULTS_DIR/fr).

    python manage.py embed_books --dry-run        # validate chunking, no API calls
    python manage.py embed_books                  # embed everything -> Chroma + MySQL
    python manage.py embed_books --language fr --limit 1

Routing metadata is derived from the folder (language) + the subject taxonomy.
Books are upserted into MySQL; chunks land in both MySQL and ChromaDB.
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.utils import timezone

from apps.catalog.metadata import clean_title, infer_subject_code
from apps.catalog.models import Book, Subject
from apps.catalog.services.embeddings import OpenAIEmbedder
from apps.catalog.services.ingest import ingest_book
from apps.catalog.services.vectorstore import get_collection

# results subdir -> (language code, assets folder used for the Book.source_file)
_RESULTS_MAP: dict[str, tuple[str, str]] = {
    "eng": ("en", "english"),
    "fr": ("fr", "french"),
}


class Command(BaseCommand):
    help = "Embed clean OCR'd PDFs into ChromaDB (and record chunks in MySQL)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--results-dir", type=Path, default=None,
                            help="Override RESULTS_DIR (contains eng/ and fr/).")
        parser.add_argument("--language", choices=("en", "fr"), default=None,
                            help="Restrict to one language.")
        parser.add_argument("--limit", type=int, default=None,
                            help="Process at most N books per language (debugging).")
        parser.add_argument("--dry-run", action="store_true",
                            help="Chunk and report only; no embeddings, no writes.")

    def handle(self, *args: object, **options: object) -> None:
        results_dir = Path(options["results_dir"] or settings.RESULTS_DIR)
        dry_run = options["dry_run"]
        wanted = options["language"]
        limit = options["limit"]

        embedder = collection = None
        if not dry_run:
            if not settings.OPENAI_API_KEY:
                raise CommandError(
                    "OPENAI_API_KEY is not set. Add it to backend/.env, or use --dry-run."
                )
            embedder = OpenAIEmbedder(settings.OPENAI_API_KEY, settings.OPENAI_EMBED_MODEL)
            collection = get_collection(settings.CHROMA_DIR, settings.CHROMA_COLLECTION)

        if dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN — no embeddings, no DB writes."))

        total_books = total_chunks = total_tokens = 0
        for subdir, (language, folder) in _RESULTS_MAP.items():
            if wanted and wanted != language:
                continue
            language_dir = results_dir / subdir
            if not language_dir.is_dir():
                self.stdout.write(self.style.WARNING(f"missing folder: {language_dir}"))
                continue

            pdfs = sorted(language_dir.glob("*.pdf"))
            if limit:
                pdfs = pdfs[:limit]

            for pdf in pdfs:
                book = self._resolve_book(pdf, language, folder, persist=not dry_run)
                if book is None:
                    self.stdout.write(self.style.WARNING(f"  no subject match, skipping: {pdf.name}"))
                    continue

                result = ingest_book(
                    book=book, pdf_path=pdf, embedder=embedder, collection=collection, dry_run=dry_run,
                )
                total_books += 1
                total_chunks += result.chunks
                total_tokens += result.tokens
                self.stdout.write(
                    f"  [{language}] {book.title}: {result.chunks} chunks, {result.tokens} tokens"
                )
                if dry_run and result.sample is not None:
                    s = result.sample
                    self.stdout.write(
                        self.style.HTTP_INFO(
                            f"      e.g. p.{s.page_start}-{s.page_end} | {s.heading_path or '(no heading)'}\n"
                            f"           {s.text[:160]}..."
                        )
                    )

        self.stdout.write(self.style.SUCCESS(
            f"Done: {total_books} books, {total_chunks} chunks, ~{total_tokens} tokens."
        ))

    def _resolve_book(self, pdf: Path, language: str, folder: str, *, persist: bool) -> Book | None:
        raw_title, total_pages = self._read_pdf_metadata(pdf)
        title = clean_title(raw_title, pdf.name)
        code = infer_subject_code(title, pdf.name)
        if code is None:
            return None

        source_file = f"{folder}/{pdf.name}"
        if not persist:
            # Unsaved instance — enough for the dry-run chunking path.
            subject = Subject.objects.filter(code=code).first()
            return Book(language=language, source_file=source_file, title=title,
                        subject=subject, total_pages=total_pages, pdf_path=str(pdf))

        book, _ = Book.objects.update_or_create(
            language=language,
            source_file=source_file,
            defaults={
                "title": title,
                "subject": Subject.objects.get(code=code),
                "total_pages": total_pages,
                "pdf_path": str(pdf),
                "processed_at": timezone.now(),
            },
        )
        return book

    @staticmethod
    def _read_pdf_metadata(pdf: Path) -> tuple[str | None, int | None]:
        try:
            with fitz.open(pdf) as doc:
                return doc.metadata.get("title"), doc.page_count
        except Exception:  # pragma: no cover
            return None, None
