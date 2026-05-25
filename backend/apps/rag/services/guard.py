"""Prompt guard: input sanitization, prompt-injection detection, output checks.

This is defense-in-depth alongside the strict grounding in the system prompt
and the relevance threshold enforced by the pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Non-Django fallback; the live limit is settings.RAG_MAX_QUESTION_CHARS so long
# multi-part worksheets aren't silently cut (see _max_question_chars).
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
        révèle\s+(le\s+)?prompt |
        تجاهل\s+(جميع\s+|كل\s+|كلّ\s+)?(الأوامر|التعليمات) |
        (انس|انسى|انسَ)\s+(كل\s+|جميع\s+)?(ما\s+سبق|التعليمات) |
        (اكشف|أظهر|اعرض)\s+(عن\s+)?(ال)?(prompt|البرومبت|التعليمات|نظامك) |
        (تظاهر|تصرّف|تصرف)\s+(أنك|بأنك|كأنك) |
        وضع\s+المطور
    )""",
    re.IGNORECASE | re.VERBOSE,
)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass(frozen=True, slots=True)
class GuardResult:
    ok: bool
    text: str
    reason: str = ""
    truncated: bool = False


def _max_question_chars() -> int:
    """Live char limit from settings, with a safe fallback when Django isn't set up."""
    try:
        from django.conf import settings

        return int(getattr(settings, "RAG_MAX_QUESTION_CHARS", MAX_QUESTION_CHARS))
    except Exception:
        return MAX_QUESTION_CHARS


def check_question(raw: str) -> GuardResult:
    text = _CONTROL.sub(" ", raw or "").strip()
    if not text:
        return GuardResult(False, "", "empty question")
    truncated = False
    limit = _max_question_chars()
    if len(text) > limit:
        text = text[:limit]
        truncated = True
    if _INJECTION.search(text):
        return GuardResult(False, text, "possible prompt injection")
    return GuardResult(True, text, truncated=truncated)


def sanitize_answer(text: str) -> str:
    """Light output guard: strip leaked role/system markers; neutralise leaked
    LaTeX list environments (KaTeX renders only math, so a stray \\begin{itemize}
    would hang unrendered) by turning them into Markdown; trim. Math environments
    like \\begin{aligned}/\\begin{cases} are left untouched."""
    text = re.sub(r"(?is)<think>.*?</think>\s*", "", text)  # reasoning-model chain-of-thought
    text = re.sub(r"(?im)^\s*(system|assistant|user)\s*:\s*", "", text)
    text = re.sub(r"\\(?:begin|end)\s*\{(?:itemize|enumerate)\}", "", text)
    text = re.sub(r"\\item\s*", "- ", text)
    return text.strip()
