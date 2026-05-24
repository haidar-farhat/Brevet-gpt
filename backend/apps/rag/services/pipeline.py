"""Async RAG orchestrator.

Flow: guard -> plan/route (LLM) -> recursive hybrid retrieve -> select/assemble
-> grounding guard -> generate (LLM) -> output guard. Returns the answer with
citations and detailed per-stage metrics (latency, tokens, tokens/sec).

Sync retrieval (Chroma + Django ORM + bge-m3) is offloaded via asyncio.to_thread
so it never touches the event loop; only the LM Studio calls are truly async.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from time import perf_counter

from django.conf import settings

from apps.rag.services import prompts, reformulation
from apps.rag.services.guard import check_question, sanitize_answer
from apps.rag.services.llm import LLMResult, LMStudioClient
from apps.rag.services.retrieval import HybridRetriever

_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Cached retriever (embedder + Chroma collection) — the bge-m3 model loads
    once and is reused across questions. Safe to share: it is only ever called
    inside asyncio.to_thread, never on the event loop."""
    global _retriever
    if _retriever is None:
        from apps.catalog.services.embeddings import build_embedder
        from apps.catalog.services.vectorstore import get_collection

        _retriever = HybridRetriever(
            build_embedder(),
            get_collection(settings.CHROMA_DIR, settings.CHROMA_COLLECTION),
        )
    return _retriever


_UNSET = object()
_reranker = _UNSET


def get_reranker():
    """Cached reranker (loaded once, like the embedder). May be None when
    reranking is disabled or the backend isn't a cross-encoder."""
    global _reranker
    if _reranker is _UNSET:
        from apps.rag.services.rerank import build_reranker

        _reranker = build_reranker()
    return _reranker


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    language: str
    subject: str | None
    queries: list[str]
    citations: list[dict]
    contexts: list[dict]
    metrics: dict
    status: str = "answer"  # answer | clarify | out_of_scope | refused | blocked

    def to_dict(self) -> dict:
        return asdict(self)


def _aggregate_metrics(llm_results: list[LLMResult], generation: LLMResult | None,
                       latency: dict, **extra) -> dict:
    prompt = sum(r.prompt_tokens for r in llm_results)
    completion = sum(r.completion_tokens for r in llm_results)
    return {
        "llm_calls": len(llm_results),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "generation_tokens_per_sec": round(generation.tokens_per_sec, 1) if generation else 0.0,
        "latency": {k: round(v, 3) for k, v in latency.items()},
        **extra,
    }


def nonempty_answer(text: str, language: str) -> str:
    """Guarantee a non-blank answer: when generation produced nothing, return a
    friendly localized fallback so the SSE result.answer is never empty."""
    if text and text.strip():
        return text
    return prompts.EMPTY_FALLBACK.get(language, prompts.EMPTY_FALLBACK["en"])


async def answer_question(question: str, *, language: str | None = None,
                          subject: str | None = None, top_k: int | None = None,
                          on_event=None) -> Answer:
    """Answer a question. If ``on_event`` (async callable) is given, emit live
    log events per stage and stream the answer tokens, for the web terminal."""
    started = perf_counter()
    llm_results: list[LLMResult] = []

    async def emit(event: dict) -> None:
        if on_event is not None:
            await on_event(event)

    guard = check_question(question)
    if not guard.ok:
        lang = language or reformulation._guess_language(question)
        await emit({"type": "log", "stage": "guard", "level": "warn", "message": f"blocked: {guard.reason}"})
        return _terminal(question, lang, subject, _guard_message(lang), started, llm_results,
                         status="blocked", refused=True, reason=guard.reason)
    question = guard.text
    await emit({"type": "log", "stage": "guard", "message": "input accepted"})
    if guard.truncated:
        await emit({"type": "log", "stage": "guard", "level": "warn",
                    "message": f"question was very long — using the first {len(question)} characters"})

    # --- Semantic cache lookup ------------------------------------------
    query_vec = None
    if settings.RAG_CACHE:
        from apps.rag.services.cache import get_cache
        try:
            query_vec = (await asyncio.to_thread(get_retriever().embedder.embed, [question]))[0]
            hit = await asyncio.to_thread(get_cache().get, query_vec, language, subject)
        except Exception:
            hit = None
        if hit is not None:
            await emit({"type": "log", "stage": "cache", "message": "semantic cache hit — returning stored answer"})
            return hit

    llm = LMStudioClient()
    if settings.RAG_AGENTIC:
        from apps.rag.services import agent
        answer = await agent.agentic_answer(
            question, language=language, subject=subject, top_k=top_k,
            on_event=on_event, llm=llm, started=started, llm_results=llm_results,
        )
    else:
        answer = await _simple_answer(
            question, language=language, subject=subject, top_k=top_k,
            on_event=on_event, llm=llm, started=started, llm_results=llm_results,
        )

    # --- Store successful answers in the cache --------------------------
    if settings.RAG_CACHE and query_vec is not None and answer.status == "answer" and not answer.refused:
        try:
            await asyncio.to_thread(get_cache().put, question, query_vec, answer, language, subject)
        except Exception:
            pass
    return answer


async def _simple_answer(question: str, *, language, subject, top_k, on_event,
                         llm: LMStudioClient, started: float, llm_results: list[LLMResult]) -> Answer:
    """The original linear pipeline (used when RAG_AGENTIC=False)."""

    async def emit(event: dict) -> None:
        if on_event is not None:
            await on_event(event)

    # 1. Plan / route -----------------------------------------------------
    t0 = perf_counter()
    plan, plan_result = await reformulation.plan_query(llm, question, language, subject)
    reformulate_s = perf_counter() - t0
    llm_results.append(plan_result)
    await emit({"type": "log", "stage": "route", "language": plan.language, "subject": plan.subject,
                "queries": plan.queries, "latency_s": round(reformulate_s, 3)})

    # 2. Retrieve (with bounded recursive broadening) ---------------------
    retriever = get_retriever()
    candidates, k = settings.RAG_CANDIDATES, top_k or settings.RAG_TOP_K
    budget, min_rel = settings.RAG_MAX_CONTEXT_TOKENS, settings.RAG_MIN_RELEVANCE

    t0 = perf_counter()
    selected, best_sim = await asyncio.to_thread(
        retriever.retrieve, plan.queries, plan.language, plan.subject,
        candidates=candidates, top_k=k, token_budget=budget,
    )
    reformulations = 0
    while best_sim < min_rel and reformulations < settings.RAG_MAX_REFORMULATIONS:
        await emit({"type": "log", "stage": "retrieve", "level": "warn",
                    "message": f"weak match (sim={best_sim:.2f}); broadening query"})
        extra_queries, broaden_result = await reformulation.broaden(llm, question)
        llm_results.append(broaden_result)
        reformulations += 1
        if not extra_queries:
            break
        # drop the subject filter on retries to widen the net
        retry_sel, retry_sim = await asyncio.to_thread(
            retriever.retrieve, extra_queries, plan.language, None,
            candidates=candidates, top_k=k, token_budget=budget,
        )
        if retry_sim > best_sim:
            selected, best_sim = retry_sel, retry_sim
    retrieve_s = perf_counter() - t0
    await emit({"type": "log", "stage": "retrieve", "chunks": len(selected),
                "best_sim": round(best_sim, 3), "reformulations": reformulations,
                "latency_s": round(retrieve_s, 3),
                "sources": [{"n": i, "book": c.book_title, "page": c.page_start,
                             "sim": round(c.dense_sim, 3)} for i, c in enumerate(selected, 1)]})

    # 3. Grounding guard --------------------------------------------------
    if not selected or best_sim < min_rel:
        await emit({"type": "log", "stage": "answer", "level": "warn",
                    "message": "insufficient context — refusing"})
        return _refusal(question, plan.language, plan.subject,
                        prompts.REFUSAL[plan.language], started, llm_results,
                        reason="no relevant context", best_sim=best_sim,
                        reformulations=reformulations,
                        latency={"reformulate_s": reformulate_s, "retrieve_s": retrieve_s})

    # 4. Generate ---------------------------------------------------------
    await emit({"type": "log", "stage": "generate", "message": f"generating with {await llm.model()}"})
    t0 = perf_counter()
    messages = prompts.build_answer_messages(question, selected, plan.language)
    if on_event is not None:
        generation = await llm.chat_stream(messages, lambda delta: emit({"type": "token", "text": delta}))
    else:
        generation = await llm.chat(messages)
    generate_s = perf_counter() - t0
    llm_results.append(generation)
    answer_text = nonempty_answer(sanitize_answer(generation.text), plan.language)

    metrics = _aggregate_metrics(
        llm_results, generation,
        {"reformulate_s": reformulate_s, "retrieve_s": retrieve_s,
         "generate_s": generate_s, "total_s": perf_counter() - started},
        model=await llm.model(),
        best_similarity=round(best_sim, 3),
        reformulations=reformulations,
        context_chunks=len(selected),
    )
    return Answer(
        question=question,
        answer=answer_text,
        refused=False,
        language=plan.language,
        subject=plan.subject,
        queries=plan.queries,
        citations=[_citation(i, c) for i, c in enumerate(selected, 1)],
        contexts=[_context(i, c) for i, c in enumerate(selected, 1)],
        metrics=metrics,
    )


def _citation(n: int, c) -> dict:
    return {"n": n, "book": c.book_title, "subject": c.subject,
            "page_start": c.page_start, "page_end": c.page_end, "heading": c.heading_path}


def _context(n: int, c) -> dict:
    return {**_citation(n, c), "content": c.content, "snippet": c.content[:240],
            "dense_sim": round(c.dense_sim, 3), "score": round(c.score, 4)}


def _guard_message(lang: str) -> str:
    return {
        "en": "I can only help with questions about the course materials, and I can't follow "
              "instructions embedded in the request.",
        "fr": "Je ne peux répondre qu'aux questions portant sur les documents du cours, et je "
              "ne peux pas suivre d'instructions cachées dans la demande.",
    }[lang]


def _terminal(question: str, language: str, subject: str | None, text: str, started: float,
              llm_results: list[LLMResult], *, status: str, refused: bool, reason: str,
              best_sim: float = 0.0, reformulations: int = 0, latency: dict | None = None) -> Answer:
    """Build an early-exit Answer (refusal, off-scope, clarification, blocked)."""
    lat = latency or {}
    lat["total_s"] = perf_counter() - started
    metrics = _aggregate_metrics(llm_results, None, lat, status=status, refused_reason=reason,
                                 best_similarity=round(best_sim, 3),
                                 reformulations=reformulations, context_chunks=0)
    return Answer(question=question, answer=text, refused=refused, language=language,
                  subject=subject, queries=[], citations=[], contexts=[], metrics=metrics,
                  status=status)


def _refusal(question: str, language: str, subject: str | None, text: str, started: float,
             llm_results: list[LLMResult], *, reason: str, best_sim: float = 0.0,
             reformulations: int = 0, latency: dict | None = None) -> Answer:
    return _terminal(question, language, subject, text, started, llm_results,
                     status="refused", refused=True, reason=reason, best_sim=best_sim,
                     reformulations=reformulations, latency=latency)


async def health() -> dict:
    """Liveness of the moving parts, for the API/run command."""
    from apps.catalog.models import Chunk

    out: dict = {}
    try:
        out["llm"] = await LMStudioClient().health()
        out["llm_ok"] = True
    except Exception as exc:
        out["llm_ok"] = False
        out["llm_error"] = str(exc)
    out["chunks"] = await asyncio.to_thread(Chunk.objects.count)
    out["vectors"] = await asyncio.to_thread(get_retriever().collection.count)
    return out
