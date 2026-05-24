"""Throwaway end-to-end verification for the decompose-and-solve change.

Runs one factual question (non-solve guarded path) and one multi-part math
problem (solve branch) through the real pipeline (MySQL + LM Studio), and prints
the key fields the plan asks us to assert. Delete after use.
"""
import asyncio
import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.local")
os.environ["RAG_CACHE"] = "False"  # bypass the semantic cache so the pipeline actually runs
django.setup()

from apps.rag.services.pipeline import answer_question  # noqa: E402


def safe(s) -> str:
    """ASCII-safe for the Windows cp1252 console (the answer may contain emoji)."""
    return str(s).encode("ascii", "replace").decode("ascii")


CASES = [
    ("Given the polynomial P(x) = (x - 3)^2 - 4. "
     "Part 1: expand and simplify P(x). "
     "Part 2: factorise P(x). "
     "Part 3: solve the equation P(x) = 0.", "math", "en", "MULTI-PART PROBLEM (solve branch)"),
]


async def main() -> None:
    for q, subj, lang, tag in CASES:
        print("=" * 70, flush=True)
        print("CASE:", tag, flush=True)
        print("Q:", safe(q[:120]), flush=True)
        a = await answer_question(q, language=lang, subject=subj)
        m = a.metrics
        print("-> status      :", a.status, flush=True)
        print("-> refused     :", a.refused, flush=True)
        print("-> answer_len  :", len(a.answer or ""), flush=True)
        print("-> solved_parts:", m.get("solved_parts"), flush=True)
        print("-> subproblems :", m.get("subproblems"), flush=True)
        print("-> llm_calls   :", m.get("llm_calls"), flush=True)
        print("-> ctx_chunks  :", m.get("context_chunks"), flush=True)
        print("-> citations   :", len(a.citations), flush=True)
        print("-> total_s     :", m.get("latency", {}).get("total_s"), flush=True)
        body = (a.answer or "").replace("\n", " ")
        print("-> ANSWER_FULL :", safe(body), flush=True)
        print("", flush=True)
    print("VERIFY_DONE", flush=True)


asyncio.run(main())
