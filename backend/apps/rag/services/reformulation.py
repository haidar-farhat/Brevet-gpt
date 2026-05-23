"""LLM-driven query planning: detect language, route to a subject, and expand
the question into several search queries (decomposing multi-part questions).
``broaden`` produces alternative queries for the recursive re-retrieval step.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from apps.rag.services import prompts
from apps.rag.services.llm import LLMResult, LMStudioClient

_VALID_SUBJECTS = {
    "math", "physics", "chemistry", "biology", "informatics",
    "grammar", "reading", "french", "english",
}
_FR_HINT = re.compile(r"[éèêàùçôîïœ]|\b(le|la|les|un|une|des|du|est|quoi|comment|pourquoi|qu'est|quelle?)\b", re.I)


@dataclass(slots=True)
class QueryPlan:
    language: str
    subject: str | None
    queries: list[str]


def _guess_language(text: str) -> str:
    return "fr" if _FR_HINT.search(text) else "en"


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
    if lang not in ("en", "fr"):
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
