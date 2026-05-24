"""Throwaway: verify the solve branch triggers for a multi-part worksheet even
when NO subject is selected (mirrors the user's case), cache bypassed. Delete after.
"""
import asyncio
import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.local")
os.environ["RAG_CACHE"] = "False"
django.setup()

from apps.rag.services.pipeline import answer_question  # noqa: E402


def safe(s) -> str:
    return str(s).encode("ascii", "replace").decode("ascii")


# 6-part subset of the user's worksheet; subject + language left to routing.
Q = (
    "1. Arrange in descending order and reduce: P(x)=5x^6-2x^3+9+x^2+4x^3-3x^6+7x-5x^2+1\n"
    "2. Simplify and state the degree: Q(x)=10x^4-6x^2+3x+2x^4+5x^2-8x+11\n"
    "6. Given P(x)=2x^3-5x^2+3x+4, find P(-2).\n"
    "7. Find the values of x that make P(x)=x^2-49 equal to zero.\n"
    "8. Determine whether P(x)=2x^3+4x^2-1+x^3 and Q(x)=3x^3+4x^2-1 are identical.\n"
    "9. Solve (x^2-64)/(x-8)=0 (state excluded values)."
)


async def main() -> None:
    a = await answer_question(Q, language=None, subject=None)
    m = a.metrics
    print("routed_subject:", a.subject, "| language:", a.language)
    print("status       :", a.status, "| refused:", a.refused)
    print("answer_len   :", len(a.answer or ""))
    print("solved_parts :", m.get("solved_parts"))
    print("subproblems  :", m.get("subproblems"))
    print("llm_calls    :", m.get("llm_calls"))
    print("ctx_chunks   :", m.get("context_chunks"))
    print("citations    :", len(a.citations))
    print("total_s      :", m.get("latency", {}).get("total_s"))
    print("ANSWER:")
    print(safe(a.answer or ""))
    print("VERIFY_DONE")


asyncio.run(main())
