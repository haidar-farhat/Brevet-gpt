"""Admin registrations for inspecting the seeded catalog."""
from __future__ import annotations

from django.contrib import admin

from apps.catalog.models import Book, Chunk, Section, Subject


@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ("code", "name_en", "name_fr", "book_count")
    search_fields = ("code", "name_en", "name_fr")

    @admin.display(description="books")
    def book_count(self, obj: Subject) -> int:
        return obj.books.count()


@admin.register(Book)
class BookAdmin(admin.ModelAdmin):
    list_display = ("title", "language", "subject", "total_pages", "ocr_confidence", "processed_at")
    list_filter = ("language", "subject", "level")
    search_fields = ("title", "source_file")
    autocomplete_fields = ("subject",)
    ordering = ("language", "title")


@admin.register(Section)
class SectionAdmin(admin.ModelAdmin):
    list_display = ("book", "level", "title", "page_start", "page_end", "ordinal")
    list_filter = ("book__language", "level")
    search_fields = ("title", "path")
    autocomplete_fields = ("book", "parent")


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ("book", "chunk_index", "language", "subject", "page_start", "page_end", "token_count")
    list_filter = ("language", "subject", "book")
    search_fields = ("content", "heading_path")
    autocomplete_fields = ("book", "section", "subject")
