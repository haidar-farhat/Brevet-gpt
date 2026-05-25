"""LLM-driven query planning: detect language, route to a subject, and expand
the question into several search queries (decomposing multi-part questions).
``broaden`` produces alternative queries for the recursive re-retrieval step.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from apps.rag.services import prompts
from apps.rag.services.llm import LLMResult, LMStudioClient

_VALID_SUBJECTS = {
    "math", "physics", "chemistry", "biology", "informatics",
    "grammar", "reading", "french", "english",
}
_FR_HINT = re.compile(r"[éèêàùçôîïœ]|\b(le|la|les|un|une|des|du|est|quoi|comment|pourquoi|qu'est|quelle?)\b", re.I)

# Heuristic backstop for "is this an exercise to SOLVE?" — a small model classifies
# this unreliably, so we OR its flag with these deterministic signals (EN + FR).
_PROBLEM_VERB = re.compile(
    r"\b(solv|calcul|comput|evaluat|factor|expand|simplif|prov|determin|deduc|deriv|"
    r"construct|find\s+(the|x|y|all|its|value)|show\s+that|"
    r"résou|resou|factoris|développ|developp|démontr|demontr|déduir|deduir|détermin|"
    r"construir|vérifi|verifi)"
    # Arabic solve-verbs (anchored at word start to limit false positives):
    r"|(?:(?<=\s)|^)(?:احسب|أحسب|احسبي|اوجد|أوجد|أوجدي|جد|حلّ|حل|اثبت|أثبت|برهن|بسّط|بسط|"
    r"بسّطي|عيّن|عين|حدّد|حدد|اشتقّ|اشتق|استنتج|بيّن|بيّني|ارسم|انشر|فكّك|فكك|حلّل|حلل|"
    r"تحقّق|تحقق|علّل|قارن)", re.IGNORECASE)
# Numbered / lettered parts: "1)", "2.", "a)", "Part 3", "Partie 2", "ii)".
_PART_MARKER = re.compile(r"(^|\s)(\d{1,2}[).]|[a-f][).]|i{1,3}[).]|partie?\s*\d)", re.IGNORECASE)
_EXPRESSION = re.compile(r"=|\b[A-Za-z]\s*\(\s*[a-z]\s*\)")  # an equation or f(x)-style expression


@dataclass(slots=True)
class QueryPlan:
    language: str
    subject: str | None
    queries: list[str]


@dataclass(slots=True)
class QueryAnalysis:
    language: str
    subject: str | None
    in_scope: bool
    needs_clarification: bool
    clarification: str
    queries: list[str]  # specific + one broad step-back query
    is_problem: bool = False           # an exercise to SOLVE (vs. a factual question)
    sub_problems: list[str] = field(default_factory=list)  # self-contained parts to solve


# Arabic script (incl. supplement + presentation forms) — detect 'ar' from the text.
_AR_HINT = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")

# Languages the router/pipeline supports end to end (prompts, fallbacks, retrieval).
SUPPORTED_LANGUAGES = ("en", "fr", "ar")


def _guess_language(text: str) -> str:
    if _AR_HINT.search(text or ""):
        return "ar"
    return "fr" if _FR_HINT.search(text) else "en"


def looks_like_problem(text: str) -> bool:
    """Deterministic backstop for the LLM's is_problem flag. True when the text
    reads like an exercise to work out (an explicit solve-verb, or numbered parts
    alongside an equation/expression)."""
    t = text or ""
    if _PROBLEM_VERB.search(t):
        return True
    return bool(_PART_MARKER.search(t) and _EXPRESSION.search(t))


def _as_text(item) -> str:
    """A sub-problem may arrive as a plain string or as an object (some models
    emit {"part": "...", "problem": "..."}); normalise either to text."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        vals = [str(v).strip() for v in item.values()
                if isinstance(v, (str, int, float)) and str(v).strip()]
        return " ".join(vals).strip()
    return ""


_catalog: str | None = None


def get_materials_catalog() -> str:
    """Cached 'subject: book titles' listing of the actual corpus, so the router
    decides scope/subject from real materials rather than guessing."""
    global _catalog
    if _catalog is None:
        from collections import defaultdict

        from apps.catalog.models import Book

        by_subject: dict[str, list[str]] = defaultdict(list)
        for code, title in Book.objects.values_list("subject__code", "title"):
            by_subject[code].append(title)
        _catalog = "\n".join(
            f"- {code}: {', '.join(sorted(set(titles)))}" for code, titles in sorted(by_subject.items())
        ) or "(catalog unavailable)"
    return _catalog


async def analyze_query(llm: LMStudioClient, question: str, language: str | None,
                        subject: str | None, catalog: str | None = None) -> tuple[QueryAnalysis, LLMResult]:
    """One up-front LLM call that routes, decides scope/clarification, and plans
    retrieval queries (including a broad step-back query). When a materials
    catalog is provided, routing is grounded in the books that actually exist."""
    system = prompts.ANALYZE_SYSTEM
    if catalog:
        system += "\n\nAVAILABLE COURSE MATERIALS (subject: books) — route using these:\n" + catalog
    data, result = await llm.chat_json(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=360,  # small JSON => reliable routing (decomposition is a separate call)
    )
    lang = language or data.get("language")
    if lang not in SUPPORTED_LANGUAGES:
        lang = _guess_language(question)

    subj = subject or data.get("subject")
    if subj not in _VALID_SUBJECTS:
        subj = None

    # Default to in-scope / no-clarify if the model omits/garbles the flags
    # (fail open — never silently drop a real question).
    in_scope = bool(data.get("in_scope", True))
    needs_clarification = bool(data.get("needs_clarification", False))
    clarification = str(data.get("clarification") or "").strip()
    if needs_clarification and not clarification:
        clarification = prompts.CLARIFY_FALLBACK[lang]

    queries = [q.strip() for q in (data.get("search_queries") or []) if isinstance(q, str) and q.strip()]
    if question not in queries:
        queries.append(question)
    if not queries:
        queries = [question]

    # is_problem gates the solve path. Decomposition into sub_problems is a
    # separate, focused call (decompose_problem) so this routing JSON stays small
    # and reliable on a small model; still accept sub_problems here if present.
    is_problem = bool(data.get("is_problem", False))
    sub_problems = [t for t in (_as_text(x) for x in (data.get("sub_problems") or [])) if t]

    return QueryAnalysis(
        language=lang, subject=subj, in_scope=in_scope,
        needs_clarification=needs_clarification, clarification=clarification,
        queries=queries[:5], is_problem=is_problem, sub_problems=sub_problems,
    ), result


async def plan_query(llm: LMStudioClient, question: str, language: str | None,
                     subject: str | None) -> tuple[QueryPlan, LLMResult]:
    data, result = await llm.chat_json(
        [
            {"role": "system", "content": prompts.REFORMULATE_SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    lang = language or data.get("language")
    if lang not in SUPPORTED_LANGUAGES:
        lang = _guess_language(question)

    subj = subject or data.get("subject")
    if subj not in _VALID_SUBJECTS:
        subj = None

    queries = [q.strip() for q in (data.get("search_queries") or []) if isinstance(q, str) and q.strip()]
    if question not in queries:
        queries.append(question)  # always keep the verbatim question for recall
    if not queries:
        queries = [question]
    return QueryPlan(language=lang, subject=subj, queries=queries[:5]), result


async def broaden(llm: LMStudioClient, question: str) -> tuple[list[str], LLMResult]:
    data, result = await llm.chat_json(
        [
            {"role": "system", "content": prompts.BROADEN_SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
        max_tokens=200,
    )
    queries = [q.strip() for q in (data.get("search_queries") or []) if isinstance(q, str) and q.strip()]
    return queries[:4], result


async def refine_queries(llm: LMStudioClient, question: str, *, missing: str,
                         prior_queries: list[str], language: str) -> tuple[list[str], LLMResult]:
    """Failure-aware refinement: target the grader's MISSING gap, avoid repeats."""
    user = (
        f"QUESTION: {question}\n"
        f"ALREADY_TRIED: {prior_queries}\n"
        f"MISSING: {missing or 'relevant supporting passages'}"
    )
    data, result = await llm.chat_json(
        [
            {"role": "system", "content": prompts.REFINE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        max_tokens=200,
    )
    tried = {q.lower() for q in prior_queries}
    queries = [
        q.strip() for q in (data.get("search_queries") or [])
        if isinstance(q, str) and q.strip() and q.strip().lower() not in tried
    ]
    return queries[:4], result


async def decompose_problem(llm: LMStudioClient, question: str, *,
                            language: str) -> tuple[list[str], LLMResult]:
    """Split a multi-part exercise into self-contained sub-problems (each carrying
    the shared data). A focused call with a generous token budget + tolerant
    parsing, so a small model returns complete, usable parts even for long
    worksheets. Returns ([] if it can't decompose, then the caller falls back)."""
    data, result = await llm.chat_json(
        [
            {"role": "system", "content": prompts.DECOMPOSE_SYSTEM},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    raw = data.get("parts") or data.get("sub_problems") or []
    parts = [t for t in (_as_text(x) for x in raw) if t] if isinstance(raw, list) else []
    return parts, result
