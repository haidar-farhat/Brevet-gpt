"""Canonical subject taxonomy — the single source of truth for routing.

Seeded into the ``subjects`` table and reused by ``infer_subject_code`` to map
a book's title/filename onto a subject. Aliases are matched case- and
accent-insensitively, so they are stored unaccented and lowercase.

Order matters: more specific/content subjects are listed before the generic
language subjects so, e.g., a French maths book routes to MATHEMATICS rather
than FRENCH.
"""
from __future__ import annotations

from dataclasses import dataclass

from apps.catalog.enums import SubjectCode


@dataclass(frozen=True, slots=True)
class SubjectSeed:
    code: SubjectCode
    name_en: str
    name_fr: str
    aliases: tuple[str, ...]


SUBJECTS: tuple[SubjectSeed, ...] = (
    SubjectSeed(
        SubjectCode.MATHEMATICS, "Mathematics", "Mathématiques",
        ("math", "mathematics", "mathematiques", "construire les math", "algebre", "geometrie"),
    ),
    SubjectSeed(
        SubjectCode.PHYSICS, "Physics", "Physique",
        ("physics", "physique"),
    ),
    SubjectSeed(
        SubjectCode.CHEMISTRY, "Chemistry", "Chimie",
        ("chemistry", "chimie"),
    ),
    SubjectSeed(
        SubjectCode.BIOLOGY, "Biology & Earth Sciences", "Sciences de la Vie et de la Terre",
        ("biology", "biologie", "svt", "sciences de la vie", "vie et de la terre", "life and earth", "bio"),
    ),
    SubjectSeed(
        SubjectCode.INFORMATICS, "Informatics", "Informatique",
        ("informatics", "informatique", "informatic", "computer"),
    ),
    SubjectSeed(
        SubjectCode.GRAMMAR, "Grammar", "Grammaire",
        ("grammar", "grammaire"),
    ),
    SubjectSeed(
        SubjectCode.READING, "Reading & Activities", "Lecture et Activités",
        ("reading", "lecture", "activites", "activity"),
    ),
    SubjectSeed(
        SubjectCode.FRENCH, "French Language", "Français",
        ("french", "francais"),
    ),
    SubjectSeed(
        SubjectCode.ENGLISH, "English Language", "Anglais",
        ("english", "anglais"),
    ),
)
