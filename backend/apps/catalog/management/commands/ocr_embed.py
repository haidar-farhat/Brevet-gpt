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

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.utils import timezone

from apps.catalog.data import taxonomy
from apps.catalog.enums import LanguageCode, SubjectCode
from apps.catalog.metadata import clean_title
from apps.catalog.models import Book, Grade, School, Subject
from apps.catalog.services import ocr as ocr_svc
from apps.catalog.services.embeddings import build_embedder
from apps.catalog.services.ingest import build_records, ingest_book
from apps.catalog.services.vectorstore import get_collection


class Command(BaseCommand):
    help = "Interactively OCR a scanned book and embed it (routing -> MySQL + Chroma)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--input", type=Path, default=None, help="Path to the scanned PDF.")
        parser.add_argument("--language", choices=tuple(LanguageCode.values), default=None)
        parser.add_argument("--subject", choices=tuple(SubjectCode.values), default=None)
        parser.add_argument("--title", default=None)
        parser.add_argument("--level", default="brevet")
        parser.add_argument("--non-interactive", action="store_true",
                            help="Fail instead of prompting for any missing value.")
        parser.add_argument("--dry-run", action="store_true",
                            help="OCR + register the Book, but skip embedding/Chroma.")

    def handle(self, *args: object, **options: object) -> None:
        interactive = not options["non_interactive"]
        lang_map = ocr_svc.lang_map()

        # --- 1. Gather routing variables -------------------------------------
        input_path = self._resolve_input(options["input"], interactive)
        language = options["language"] or self._choose(
            "Language", {code: label for code, label in LanguageCode.choices if code in lang_map}, interactive
        )
        subject_code = options["subject"] or self._choose(
            "Subject", dict(SubjectCode.choices), interactive
        )
        default_title = clean_title(ocr_svc.pdf_title(input_path), input_path.name)
        title = options["title"] or self._ask("Title", default_title, interactive)
        level = options["level"] or self._ask("Level", "brevet", interactive)
        dry_run = options["dry_run"]

        subject = Subject.objects.get(code=subject_code)
        default_school = School.objects.filter(code=taxonomy.DEFAULT_SCHOOL[1]).first()
        default_grade = Grade.objects.filter(code=taxonomy.DEFAULT_GRADE_CODE).first()

        # Build the embedder up front so a misconfig fails *before* costly OCR.
        embedder = collection = None
        if dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN — Book saved; skipping embeddings/Chroma."))
        else:
            try:
                embedder = build_embedder()
            except ValueError as exc:
                raise CommandError(str(exc)) from exc
            collection = get_collection(settings.CHROMA_DIR, settings.CHROMA_COLLECTION)

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"\nOCR + embed: {title}\n"
            f"  language={language}  subject={subject_code}  level={level}\n"
            f"  input={input_path}\n"
        ))

        # --- 2. OCR into a clean structured PDF ------------------------------
        try:
            clean_pdf = ocr_svc.ocr_to_clean_pdf(input_path, language)
        except ocr_svc.OCRError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(f"OCR complete -> {clean_pdf}"))

        # --- 3. Upsert the Book (routing system of record) -------------------
        total_pages = ocr_svc.page_count(clean_pdf)
        book, _ = Book.objects.update_or_create(
            language=language,
            source_file=f"{lang_map[language][2]}/{input_path.name}",
            defaults={
                "title": title,
                "subject": subject,
                "school": default_school,
                "grade": default_grade,
                "level": level,
                "pdf_path": str(clean_pdf),
                "total_pages": total_pages,
                "processed_at": timezone.now(),
            },
        )

        # --- 4. Embed (embedder/collection were prepared up front) ----------
        result = ingest_book(
            book=book, records=build_records(clean_pdf), embedder=embedder,
            collection=collection, dry_run=dry_run,
        )
        self.stdout.write(self.style.SUCCESS(
            f"Done: book #{book.id} '{book.title}' — {result.chunks} chunks, ~{result.tokens} tokens."
        ))

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
