"""Enumerations shared across the catalog domain."""
from __future__ import annotations

from django.db import models


class LanguageCode(models.TextChoices):
    """Built-in language codes used as seed defaults / code constants. The full,
    extensible list of supported languages lives in the ``Language`` model so more
    can be added at runtime."""

    ENGLISH = "en", "English"
    FRENCH = "fr", "French"
    ARABIC = "ar", "Arabic"


class BookStatus(models.TextChoices):
    """Lifecycle of a book in the corpus. Frozen books are kept (and the state is
    reversible) but excluded from retrieval."""

    ACTIVE = "active", "Active"
    FROZEN = "frozen", "Frozen"


class SubjectCode(models.TextChoices):
    """Stable machine codes for the subject taxonomy (routing targets)."""

    MATHEMATICS = "math", "Mathematics"
    PHYSICS = "physics", "Physics"
    CHEMISTRY = "chemistry", "Chemistry"
    BIOLOGY = "biology", "Biology & Earth Sciences"
    INFORMATICS = "informatics", "Informatics"
    GRAMMAR = "grammar", "Grammar"
    READING = "reading", "Reading & Activities"
    FRENCH = "french", "French Language"
    ENGLISH = "english", "English Language"
