"""Default taxonomy seeds: built-in languages, the default school, and grade
levels. Shared by the initial data migration and the ``seed`` command so both
create identical defaults."""
from __future__ import annotations

# code, name, native_name, tesseract, ocr_subdir, assets_folder
LANGUAGES: list[tuple[str, str, str, str, str, str]] = [
    ("en", "English", "English", "eng", "eng", "english"),
    ("fr", "French", "Français", "fra", "fr", "french"),
    ("ar", "Arabic", "العربية", "ara", "ar", "arabic"),
]

# name, code
DEFAULT_SCHOOL: tuple[str, str] = ("Lebanese Brevet", "lebanese-brevet")

# name, code, ordinal
GRADES: list[tuple[str, str, int]] = [
    ("Grade 7", "g7", 7),
    ("Grade 8", "g8", 8),
    ("Grade 9", "g9", 9),
]

# The existing corpus (Brevet) is grade 9.
DEFAULT_GRADE_CODE = "g9"
