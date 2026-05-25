import asyncio
import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.local")
django.setup()

from apps.rag.services.pipeline import answer_question  # noqa: E402


def p(*a):
    print(*a, flush=True)


def safe(s) -> str:
    return str(s).encode("ascii", "replace").decode("ascii")


# Stacked, as the user types it: linear line first, quadratic (the intended numerator) second.
Q = """Solve the equation and state any excluded values:
x-14
x
2
-196
=0"""


async def main():
    a = await answer_question(Q, language="en", subject=None)
    body = a.answer or ""
    low = body.lower()
    p("status:", a.status, "| solved_parts:", a.metrics.get("solved_parts"), "| answer_len:", len(body))
    p("mentions x^2-196 numerator:", "196" in body)
    p("reaches x = -14         :", "-14" in body or "= -14" in body or "x=-14" in body.replace(" ", ""))
    p("says 'no solution'      :", "no solution" in low or "aucune solution" in low)
    p("---- ANSWER ----")
    p(safe(body))
    p("VERIFY_DONE")


asyncio.run(main())
