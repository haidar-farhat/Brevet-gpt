"""Self-verification: check that the answer's claims are supported by the
retrieved context (same logic/prompts as the RAGAS faithfulness metric). The
agent acts on a low score per RAG_VERIFY_ACTION (warn / revise / refuse).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from apps.rag.services.evaluation import _CLAIMS_SYS, _SUPPORT_SYS
from apps.rag.services.llm import LLMResult, LMStudioClient


@dataclass(slots=True)
class Verification:
    faithfulness: float | None      # supported / total claims (None if not checkable)
    unsupported: list[str]
    llm_results: list[LLMResult] = field(default_factory=list)


async def verify_answer(llm: LMStudioClient, answer: str, contexts: list[str], *,
                        max_claims: int) -> Verification:
    if not answer.strip() or not contexts:
        return Verification(None, [])

    data, claims_result = await llm.chat_json(
        [{"role": "system", "content": _CLAIMS_SYS}, {"role": "user", "content": f"ANSWER:\n{answer}"}],
        temperature=0.0, max_tokens=400,
    )
    claims = [c for c in (data.get("claims") or []) if isinstance(c, str) and c.strip()][:max_claims]
    if not claims:
        return Verification(None, [], [claims_result])

    joined = "\n\n".join(contexts)
    results = [claims_result]
    unsupported: list[str] = []
    for claim in claims:
        verdict, r = await llm.chat_json(
            [{"role": "system", "content": _SUPPORT_SYS},
             {"role": "user", "content": f"CONTEXT:\n{joined}\n\nCLAIM: {claim}"}],
            temperature=0.0, max_tokens=10,
        )
        results.append(r)
        if not verdict.get("supported"):
            unsupported.append(claim)

    faithfulness = (len(claims) - len(unsupported)) / len(claims)
    return Verification(faithfulness, unsupported, results)
