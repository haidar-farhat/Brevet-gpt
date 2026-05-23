"""Prompt guard: input sanitization, prompt-injection detection, output checks.

This is defense-in-depth alongside the strict grounding in the system prompt
and the relevance threshold enforced by the pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_QUESTION_CHARS = 2000

# Common jailbreak / prompt-injection phrasings (EN + FR).
_INJECTION = re.compile(
    r"""(
        ignore\s+(all\s+|the\s+|previous\s+|above\s+)*instructions |
        disregard\s+(all\s+|the\s+|previous\s+)*instructions |
        forget\s+(everything|all|previous) |
        reveal\s+(the\s+)?(system\s+)?prompt |
        (show|print|repeat)\s+(me\s+)?(your\s+)?(system\s+)?(prompt|instructions) |
        you\s+are\s+now |
        act\s+as\s+(an?\s+)? |
        developer\s+mode |
        jailbreak |
        ignore\s+(les\s+|toutes\s+les\s+)?instructions |
        oublie\s+(tout|les\s+instructions) |
        révèle\s+(le\s+)?prompt
    )""",
    re.IGNORECASE | re.VERBOSE,
)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass(frozen=True, slots=True)
class GuardResult:
    ok: bool
    text: str
    reason: str = ""


def check_question(raw: str) -> GuardResult:
    text = _CONTROL.sub(" ", raw or "").strip()
    if not text:
        return GuardResult(False, "", "empty question")
    if len(text) > MAX_QUESTION_CHARS:
        text = text[:MAX_QUESTION_CHARS]
    if _INJECTION.search(text):
        return GuardResult(False, text, "possible prompt injection")
    return GuardResult(True, text)


def sanitize_answer(text: str) -> str:
    """Light output guard: strip leaked role/system markers, trim."""
    text = re.sub(r"(?im)^\s*(system|assistant|user)\s*:\s*", "", text).strip()
    return text
