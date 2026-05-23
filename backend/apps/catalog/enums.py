"""Enumerations shared across the catalog domain."""
from __future__ import annotations

from django.db import models


class Language(models.TextChoices):
    """Medium of instruction. Doubles as a routing dimension."""

    ENGLISH = "en", "English"
    FRENCH = "fr", "French"


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
