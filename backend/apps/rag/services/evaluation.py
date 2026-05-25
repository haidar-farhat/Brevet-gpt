"""RAGAS-faithful evaluation, implemented directly against our LM Studio judge
+ bge-m3 embeddings (no ragas/langchain dependency, robust with local LLMs).

Metrics follow the RAGAS definitions:
* faithfulness        = supported claims / total claims (claims extracted from
                        the answer, each checked against the retrieved context).
* answer_relevancy    = mean cosine similarity between the original question and
                        questions reverse-generated from the answer.
* context_precision   = mean precision@k over the ranks of relevant contexts.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

from apps.rag.services.llm import LMStudioClient

_CLAIMS_SYS = (
    'Extract the distinct factual claims stated in the ANSWER. '
    'Respond JSON only: {"claims": ["...", "..."]}. If there are none, {"claims": []}.'
)
_SUPPORT_SYS = (
    'Decide if the CONTEXT supports the CLAIM, judging ONLY from the context. '
    'Respond JSON only: {"supported": true|false}.'
)
_GENQ_SYS = (
    'Generate up to {n} concise questions that the ANSWER would correctly answer, '
    'in the same language as the answer. Respond JSON only: {{"questions": ["...", "..."]}}.'
)
_CTXREL_SYS = (
    'Decide if the CONTEXT passage is relevant/useful for answering the QUESTION. '
    'Respond JSON only: {"relevant": true|false}.'
)


@dataclass(frozen=True, slots=True)
class RagasScores:
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def _faithfulness(judge: LMStudioClient, answer: str, contexts: list[str]) -> float | None:
    if not answer.strip() or not contexts:
        return None
    data, _ = await judge.chat_json(
        [{"role": "system", "content": _CLAIMS_SYS}, {"role": "user", "content": f"ANSWER:\n{answer}"}],
        temperature=0.0, max_tokens=400,
    )
    claims = [c for c in (data.get("claims") or []) if isinstance(c, str) and c.strip()]
    if not claims:
        return None
    joined = "\n\n".join(contexts)
    supported = 0
    for claim in claims:
        verdict, _ = await judge.chat_json(
            [{"role": "system", "content": _SUPPORT_SYS},
             {"role": "user", "content": f"CONTEXT:\n{joined}\n\nCLAIM: {claim}"}],
            temperature=0.0, max_tokens=10,
        )
        supported += 1 if verdict.get("supported") else 0
    return supported / len(claims)


async def _answer_relevancy(judge: LMStudioClient, embedder, question: str, answer: str,
                            n: int = 3) -> float | None:
    if not answer.strip():
        return None
    data, _ = await judge.chat_json(
        [{"role": "system", "content": _GENQ_SYS.format(n=n)}, {"role": "user", "content": f"ANSWER:\n{answer}"}],
        temperature=0.0, max_tokens=200,
    )
    gen = [q for q in (data.get("questions") or []) if isinstance(q, str) and q.strip()]
    if not gen:
        return None
    vectors = await asyncio.to_thread(embedder.embed, [question, *gen])
    base = vectors[0]
    sims = [_cosine(base, v) for v in vectors[1:]]
    return sum(sims) / len(sims)


async def _context_precision(judge: LMStudioClient, question: str, contexts: list[str]) -> float | None:
    if not contexts:
        return None
    relevance: list[int] = []
    for ctx in contexts:
        verdict, _ = await judge.chat_json(
            [{"role": "system", "content": _CTXREL_SYS},
             {"role": "user", "content": f"QUESTION: {question}\n\nCONTEXT: {ctx}"}],
            temperature=0.0, max_tokens=10,
        )
        relevance.append(1 if verdict.get("relevant") else 0)
    if sum(relevance) == 0:
        return 0.0
    cumulative, precisions = 0, []
    for i, rel in enumerate(relevance, 1):
        if rel:
            cumulative += 1
            precisions.append(cumulative / i)
    return sum(precisions) / sum(relevance)


async def evaluate_sample(judge: LMStudioClient, embedder, *, question: str, answer: str,
                          contexts: list[str]) -> RagasScores:
    faith, relevancy, precision = await asyncio.gather(
        _faithfulness(judge, answer, contexts),
        _answer_relevancy(judge, embedder, question, answer),
        _context_precision(judge, question, contexts),
    )
    return RagasScores(faithfulness=faith, answer_relevancy=relevancy, context_precision=precision)
