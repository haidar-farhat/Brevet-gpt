"""Enumerations shared across the catalog domain."""
from __future__ import annotations

from django.db import models


class Language(models.TextChoices):
    """Medium of instruction. Doubles as a routing dimension."""

    ENGLISH = "en", "English"
    FRENCH = "fr", "French"


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
