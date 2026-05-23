"""Context analysing: grade the retrieved passages for relevance and overall
sufficiency in a SINGLE LLM call (cheap on a slow local model). Irrelevant
passages are dropped before generation so the small model isn't distracted;
insufficiency (with a "missing" hint) drives the agent's refine loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from apps.rag.services import prompts
from apps.rag.services.llm import LLMResult, LMStudioClient


@dataclass(slots=True)
class ContextGrade:
    relevant_flags: list[bool]      # aligned to the graded chunks (first max_chunks)
    relevant_fraction: float
    sufficient: bool
    missing: str
    llm_results: list[LLMResult] = field(default_factory=list)


async def grade_context(llm: LMStudioClient, question: str, chunks, *, max_chunks: int,
                        sufficiency_min: float) -> ContextGrade:
    graded = chunks[:max_chunks]
    if not graded:
        return ContextGrade([], 0.0, False, "no context retrieved")

    user = prompts.ANSWER_USER.format(context=prompts.format_context(graded), question=question)
    data, result = await llm.chat_json(
        [{"role": "system", "content": prompts.GRADE_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0, max_tokens=120,
    )

    relevant_numbers = {int(n) for n in (data.get("relevant") or []) if str(n).strip().lstrip("-").isdigit()}
    flags = [(i + 1) in relevant_numbers for i in range(len(graded))]
    fraction = sum(flags) / len(flags) if flags else 0.0
    # No usable grade => fail open (treat all as relevant) so we never wrongly refuse.
    if not relevant_numbers and "relevant" not in data:
        flags = [True] * len(graded)
        fraction = 1.0

    sufficient = bool(data.get("sufficient")) and fraction >= sufficiency_min
    missing = str(data.get("missing") or "").strip()
    return ContextGrade(flags, fraction, sufficient, missing, [result])


def filter_relevant(chunks, grade: ContextGrade):
    """Keep graded-relevant chunks plus any beyond the graded window (ungraded)."""
    flags = grade.relevant_flags
    kept = [c for i, c in enumerate(chunks) if i >= len(flags) or flags[i]]
    return kept or chunks  # never return empty — fall back to the original set
