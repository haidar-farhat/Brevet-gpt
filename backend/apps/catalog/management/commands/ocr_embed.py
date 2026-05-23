"""Interactive OCR + embed for a single new scanned book.

Prompts in the CLI for the routing variables (language, subject, title, level),
OCRs the scan into a clean structured PDF (reusing gpu_ocr_books.py), upserts
the Book into MySQL, then embeds its chunks into ChromaDB + MySQL.

    python manage.py ocr_embed                                   # fully interactive
    python manage.py ocr_embed --input scan.pdf --language fr \
        --subject physics --title "Physique" --non-interactive
    python manage.py ocr_embed --input scan.pdf ... --dry-run    # OCR, skip embedding
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz  # PyMuPDF
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.utils import timezone

from apps.catalog.enums import Language, SubjectCode
from apps.catalog.metadata import clean_title
from apps.catalog.models import Book, Subject
from apps.catalog.services.embeddings import OpenAIEmbedder
from apps.catalog.services.ingest import ingest_book
from apps.catalog.services.vectorstore import get_collection

# language code -> (tesseract lang, gpu_ocr_books results subdir, assets folder)
_LANG_MAP: dict[str, tuple[str, str, str]] = {
    "en": ("eng", "eng", "english"),
    "fr": ("fra", "fr", "french"),
}


class Command(BaseCommand):
    help = "Interactively OCR a scanned book and embed it (routing -> MySQL + Chroma)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--input", type=Path, default=None, help="Path to the scanned PDF.")
        parser.add_argument("--language", choices=tuple(_LANG_MAP), default=None)
        parser.add_argument("--subject", choices=tuple(SubjectCode.values), default=None)
        parser.add_argument("--title", default=None)
        parser.add_argument("--level", default="brevet")
        parser.add_argument("--non-interactive", action="store_true",
                            help="Fail instead of prompting for any missing value.")
        parser.add_argument("--dry-run", action="store_true",
                            help="OCR + register the Book, but skip embedding/Chroma.")

    def handle(self, *args: object, **options: object) -> None:
        interactive = not options["non_interactive"]

        # --- 1. Gather routing variables -------------------------------------
        input_path = self._resolve_input(options["input"], interactive)
        language = options["language"] or self._choose(
            "Language", {code: label for code, label in Language.choices if code in _LANG_MAP}, interactive
        )
        subject_code = options["subject"] or self._choose(
            "Subject", dict(SubjectCode.choices), interactive
        )
        default_title = clean_title(self._pdf_title(input_path), input_path.name)
        title = options["title"] or self._ask("Title", default_title, interactive)
        level = options["level"] or self._ask("Level", "brevet", interactive)
        dry_run = options["dry_run"]

        # Fail fast on a missing key *before* the expensive OCR step.
        if not dry_run and not settings.OPENAI_API_KEY:
            raise CommandError("OPENAI_API_KEY is not set. Add it to backend/.env, or use --dry-run.")

        subject = Subject.objects.get(code=subject_code)
        self.stdout.write(self.style.MIGRATE_HEADING(
            f"\nOCR + embed: {title}\n"
            f"  language={language}  subject={subject_code}  level={level}\n"
            f"  input={input_path}\n"
        ))

        # --- 2. OCR into a clean structured PDF ------------------------------
        clean_pdf = self._ocr(input_path, language)
        self.stdout.write(self.style.SUCCESS(f"OCR complete -> {clean_pdf}"))

        # --- 3. Upsert the Book (routing system of record) -------------------
        _, total_pages = self._pdf_title(clean_pdf), self._page_count(clean_pdf)
        book, _ = Book.objects.update_or_create(
            language=language,
            source_file=f"{_LANG_MAP[language][2]}/{input_path.name}",
            defaults={
                "title": title,
                "subject": subject,
                "level": level,
                "pdf_path": str(clean_pdf),
                "total_pages": total_pages,
                "processed_at": timezone.now(),
            },
        )

        # --- 4. Embed -------------------------------------------------------
        embedder = collection = None
        if not dry_run:
            embedder = OpenAIEmbedder(settings.OPENAI_API_KEY, settings.OPENAI_EMBED_MODEL)
            collection = get_collection(settings.CHROMA_DIR, settings.CHROMA_COLLECTION)
        else:
            self.stdout.write(self.style.WARNING("DRY RUN — Book saved; skipping embeddings/Chroma."))

        result = ingest_book(
            book=book, pdf_path=clean_pdf, embedder=embedder, collection=collection, dry_run=dry_run,
        )
        self.stdout.write(self.style.SUCCESS(
            f"Done: book #{book.id} '{book.title}' — {result.chunks} chunks, ~{result.tokens} tokens."
        ))

    # ------------------------------------------------------------------ OCR
    def _ocr(self, input_path: Path, language: str) -> Path:
        ocr = self._load_ocr_module()
        tess_lang, out_subdir, _iso = _LANG_MAP[language]
        _tessdata_arg, available = ocr.ensure_languages({tess_lang})
        if tess_lang not in available or "osd" not in available:
            raise CommandError(f"Tesseract language '{tess_lang}' is unavailable; cannot OCR.")

        ocr.process_pdf(str(input_path), tess_lang, _tessdata_arg, out_subdir, _iso)
        clean_pdf = Path(ocr.RESULTS_FOLDER) / out_subdir / f"{input_path.stem}.pdf"
        if not clean_pdf.is_file():
            raise CommandError(f"Expected OCR output not found: {clean_pdf}")
        return clean_pdf

    @staticmethod
    def _load_ocr_module():
        root = str(settings.PROJECT_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            import gpu_ocr_books  # noqa: PLC0415 (lazy: heavy OCR deps + Tesseract check on import)
        except Exception as exc:  # pragma: no cover
            raise CommandError(f"Could not import gpu_ocr_books from {root}: {exc}") from exc
        return gpu_ocr_books

    # -------------------------------------------------------------- prompts
    def _resolve_input(self, value: Path | None, interactive: bool) -> Path:
        if value is None:
            if not interactive:
                raise CommandError("--input is required in non-interactive mode.")
            value = Path(self._ask("Path to the scanned PDF", None, interactive=True))
        value = value.expanduser()
        if not value.is_file():
            raise CommandError(f"Input PDF not found: {value}")
        return value

    def _ask(self, label: str, default: str | None, interactive: bool) -> str:
        if not interactive:
            if default is None:
                raise CommandError(f"Missing required value: {label}")
            return default
        suffix = f" [{default}]" if default else ""
        answer = input(f"{label}{suffix}: ").strip()
        if not answer and default is not None:
            return default
        if not answer:
            raise CommandError(f"{label} is required.")
        return answer

    def _choose(self, label: str, choices: dict[str, str], interactive: bool) -> str:
        if not interactive:
            raise CommandError(f"Missing required value: {label} (one of {', '.join(choices)})")
        keys = list(choices)
        self.stdout.write(f"\n{label}:")
        for i, key in enumerate(keys, 1):
            self.stdout.write(f"  {i}) {key} — {choices[key]}")
        while True:
            answer = input(f"Select {label} [1-{len(keys)} or code]: ").strip()
            if answer in choices:
                return answer
            if answer.isdigit() and 1 <= int(answer) <= len(keys):
                return keys[int(answer) - 1]
            self.stdout.write(self.style.ERROR("  invalid choice, try again"))

    # ---------------------------------------------------------------- pdf
    @staticmethod
    def _pdf_title(pdf: Path) -> str | None:
        try:
            with fitz.open(pdf) as doc:
                return doc.metadata.get("title")
        except Exception:  # pragma: no cover
            return None

    @staticmethod
    def _page_count(pdf: Path) -> int | None:
        try:
            with fitz.open(pdf) as doc:
                return doc.page_count
        except Exception:  # pragma: no cover
            return None
