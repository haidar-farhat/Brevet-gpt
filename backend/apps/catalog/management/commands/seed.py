"""Idempotent database seed: subject taxonomy + book catalog.

    python manage.py seed                    # subjects + books (scans ASSETS_DIR)
    python manage.py seed --subjects-only    # taxonomy only
    python manage.py seed --assets-dir PATH  # override the corpus root

Sections and chunks are populated later by the ingestion pipeline; this command
only covers the relational metadata that exists prior to OCR/embedding.
"""
from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError, CommandParser
from django.db import transaction

from apps.catalog.data.subjects import SUBJECTS
from apps.catalog.enums import Language
from apps.catalog.metadata import clean_title, infer_subject_code
from apps.catalog.models import Book, Subject

# Corpus subfolder -> medium of instruction.
_FOLDER_LANGUAGE: dict[str, Language] = {
    "english": Language.ENGLISH,
    "french": Language.FRENCH,
}


class Command(BaseCommand):
    help = "Seed the subject taxonomy and book catalog (idempotent)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--subjects-only",
            action="store_true",
            help="Seed only the subject taxonomy; skip scanning for books.",
        )
        parser.add_argument(
            "--assets-dir",
            type=Path,
            default=None,
            help="Override ASSETS_DIR (the root containing english/ and french/).",
        )

    @transaction.atomic
    def handle(self, *args: object, **options: object) -> None:
        seeded = self._seed_subjects()
        self.stdout.write(self.style.SUCCESS(f"Subjects: {seeded} upserted."))

        if options["subjects_only"]:
            return

        assets_dir = options["assets_dir"] or settings.ASSETS_DIR
        created, updated, skipped = self._seed_books(Path(assets_dir))
        self.stdout.write(
            self.style.SUCCESS(
                f"Books: {created} created, {updated} updated, {skipped} skipped."
            )
        )

    def _seed_subjects(self) -> int:
        for seed in SUBJECTS:
            Subject.objects.update_or_create(
                code=seed.code,
                defaults={
                    "name_en": seed.name_en,
                    "name_fr": seed.name_fr,
                    "aliases": list(seed.aliases),
                },
            )
        return len(SUBJECTS)

    def _seed_books(self, assets_dir: Path) -> tuple[int, int, int]:
        if not assets_dir.is_dir():
            raise CommandError(f"ASSETS_DIR does not exist: {assets_dir}")

        subjects_by_code = {subject.code: subject for subject in Subject.objects.all()}
        created = updated = skipped = 0

        for folder, language in _FOLDER_LANGUAGE.items():
            language_dir = assets_dir / folder
            if not language_dir.is_dir():
                self.stdout.write(self.style.WARNING(f"  missing folder: {language_dir}"))
                continue

            for pdf in sorted(language_dir.glob("*.pdf")):
                raw_title, total_pages = self._read_pdf_metadata(pdf)
                title = clean_title(raw_title, pdf.name)
                code = infer_subject_code(title, pdf.name)
                if code is None:
                    self.stdout.write(self.style.WARNING(f"  no subject match, skipping: {pdf.name}"))
                    skipped += 1
                    continue

                _, was_created = Book.objects.update_or_create(
                    language=language,
                    source_file=f"{folder}/{pdf.name}",
                    defaults={
                        "title": title,
                        "subject": subjects_by_code[code],
                        "total_pages": total_pages,
                    },
                )
                created += int(was_created)
                updated += int(not was_created)
                verb = "created" if was_created else "updated"
                self.stdout.write(f"  [{language}] {verb}: {title}  [{code}]  ({total_pages or '?'} pp)")

        return created, updated, skipped

    @staticmethod
    def _read_pdf_metadata(pdf: Path) -> tuple[str | None, int | None]:
        """Return (embedded title, page count); degrade gracefully without PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return None, None
        try:
            with fitz.open(pdf) as doc:
                return doc.metadata.get("title"), doc.page_count
        except Exception:  # pragma: no cover - corrupt/locked file
            return None, None
