"""Agentic answer orchestrator (Corrective / Self-RAG).

Flow: plan/route -> retrieve candidates -> rerank -> grade context (analyse) ->
refine+re-retrieve loop (if insufficient) -> reason (problem-solving subjects) ->
generate -> self-verify. Every optional step is gated by a hard LLM-call budget
so a slow CPU model can't blow up, and emits a live SSE log for the web terminal.

Reuses the existing retriever/reranker/prompts/grading/verify/reformulation. The
`Answer` contract and metric aggregation stay in pipeline.py (imported lazily to
avoid the pipeline<->agent import cycle).
"""
from __future__ import annotations

import asyncio
from time import perf_counter

from django.conf import settings

from apps.rag.services import grading, prompts, reformulation, verify
from apps.rag.services.guard import sanitize_answer
from apps.rag.services.llm import LLMResult, LMStudioClient
from apps.rag.services.rerank import rerank_chunks


async def agentic_answer(question: str, *, language, subject, top_k, on_event,
                         llm: LMStudioClient, started: float, llm_results: list[LLMResult]):
    from apps.rag.services.pipeline import (  # lazy: avoid circular import
        Answer, _aggregate_metrics, _citation, _context, _refusal, get_reranker, get_retriever,
    )

    async def emit(event: dict) -> None:
        if on_event is not None:
            await on_event(event)

    def can_call() -> bool:
        return len(llm_results) < settings.RAG_AGENT_LLM_BUDGET

    retriever = get_retriever()
    reranker = get_reranker()
    candidates_k = settings.RAG_CANDIDATES
    k = top_k or settings.RAG_TOP_K
    budget_tokens = settings.RAG_MAX_CONTEXT_TOKENS
    min_rel = settings.RAG_MIN_RELEVANCE

    # 1. Plan / route -----------------------------------------------------
    t0 = perf_counter()
    plan, plan_result = await reformulation.plan_query(llm, question, language, subject)
    reformulate_s = perf_counter() - t0
    llm_results.append(plan_result)
    await emit({"type": "log", "stage": "route", "language": plan.language, "subject": plan.subject,
                "queries": plan.queries, "latency_s": round(reformulate_s, 3)})

    # rebind retrieve to the resolved plan language
    async def do_retrieve(queries, subj):
        cands, sim = await asyncio.to_thread(
            retriever.retrieve_candidates, queries, plan.language, subj, candidates=candidates_k
        )
        reranked = False
        if reranker is not None and cands:
            head = await asyncio.to_thread(
                rerank_chunks, reranker, question, cands[: settings.RAG_RERANK_CANDIDATES],
                dense_weight=settings.RAG_DENSE_SIM_WEIGHT,
            )
            cands = head + cands[settings.RAG_RERANK_CANDIDATES:]
            reranked = True
        return retriever.select_within_budget(cands, top_k=k, token_budget=budget_tokens), sim, reranked

    # 2-4. Retrieve + rerank ---------------------------------------------
    t_retr = perf_counter()
    selected, best_sim, reranked = await do_retrieve(plan.queries, plan.subject)
    if reranked:
        await emit({"type": "log", "stage": "rerank",
                    "message": f"reranked candidates ({settings.RAG_RERANK_BACKEND}) — "
                               f"top {round(selected[0].score, 3) if selected else 0}"})
    await emit({"type": "log", "stage": "retrieve", "chunks": len(selected),
                "best_sim": round(best_sim, 3), "reformulations": 0,
                "sources": [{"n": i, "book": c.book_title, "page": c.page_start,
                             "sim": round(c.dense_sim, 3)} for i, c in enumerate(selected, 1)]})

    # 5. Grade context ----------------------------------------------------
    grade = None
    if settings.RAG_GRADE_CONTEXT and selected and can_call():
        grade = await grading.grade_context(
            llm, question, selected, max_chunks=settings.RAG_GRADE_MAX_CHUNKS,
            sufficiency_min=settings.RAG_SUFFICIENCY_MIN,
        )
        llm_results.extend(grade.llm_results)
        selected = grading.filter_relevant(selected, grade)
        await emit({"type": "log", "stage": "grade",
                    "message": f"{sum(grade.relevant_flags)}/{len(grade.relevant_flags)} relevant, "
                               f"sufficient={grade.sufficient}"
                               + (f" — missing: {grade.missing}" if grade.missing else "")})

    # 6. Refine loop ------------------------------------------------------
    tried = list(plan.queries)
    loops = 0
    while loops < settings.RAG_AGENT_MAX_LOOPS and can_call() and (
        (grade is not None and not grade.sufficient) or (grade is None and best_sim < min_rel)
    ):
        missing = grade.missing if grade else ""
        new_queries, refine_result = await reformulation.refine_queries(
            llm, question, missing=missing, prior_queries=tried, language=plan.language
        )
        llm_results.append(refine_result)
        loops += 1
        await emit({"type": "log", "stage": "refine",
                    "message": f"loop {loops}: missing '{missing or 'context'}' → {new_queries}"})
        if not new_queries:
            break
        tried.extend(new_queries)
        sel2, sim2, _ = await do_retrieve(new_queries, None)  # drop subject filter to widen
        if sel2 and (sim2 >= best_sim or not selected):
            selected, best_sim = sel2, sim2
        if settings.RAG_GRADE_CONTEXT and selected and can_call():
            grade = await grading.grade_context(
                llm, question, selected, max_chunks=settings.RAG_GRADE_MAX_CHUNKS,
                sufficiency_min=settings.RAG_SUFFICIENCY_MIN,
            )
            llm_results.extend(grade.llm_results)
            selected = grading.filter_relevant(selected, grade)
    retrieve_s = perf_counter() - t_retr

    # 7. Grounding guard --------------------------------------------------
    if not selected or best_sim < min_rel:
        await emit({"type": "log", "stage": "answer", "level": "warn",
                    "message": "insufficient context — refusing"})
        return _refusal(question, plan.language, plan.subject, prompts.REFUSAL[plan.language],
                        started, llm_results, reason="no relevant context", best_sim=best_sim,
                        reformulations=loops,
                        latency={"reformulate_s": reformulate_s, "retrieve_s": retrieve_s})

    # 8. Reason (problem-solving subjects) --------------------------------
    reasoning = None
    if settings.RAG_REASON and plan.subject in settings.RAG_REASON_SUBJECTS and can_call():
        await emit({"type": "log", "stage": "reason", "message": f"reasoning over rules ({plan.subject})"})
        reason_result = await llm.chat(
            prompts.build_reason_messages(question, selected, plan.language),
            max_tokens=settings.RAG_REASON_MAX_TOKENS,
        )
        llm_results.append(reason_result)
        reasoning = reason_result.text

    # 9. Generate ---------------------------------------------------------
    await emit({"type": "log", "stage": "generate", "message": f"generating with {await llm.model()}"})
    t0 = perf_counter()
    messages = prompts.build_answer_messages(question, selected, plan.language, reasoning=reasoning)
    if on_event is not None:
        generation = await llm.chat_stream(messages, lambda d: emit({"type": "token", "text": d}))
    else:
        generation = await llm.chat(messages)
    generate_s = perf_counter() - t0
    llm_results.append(generation)
    answer_text = sanitize_answer(generation.text)

    # 10. Self-verify -----------------------------------------------------
    faithfulness = None
    revised = False
    if settings.RAG_VERIFY and can_call():
        v = await verify.verify_answer(llm, answer_text, [c.content for c in selected],
                                       max_claims=settings.RAG_VERIFY_MAX_CLAIMS)
        llm_results.extend(v.llm_results)
        faithfulness = v.faithfulness
        if faithfulness is not None and faithfulness < settings.RAG_VERIFY_MIN:
            action = settings.RAG_VERIFY_ACTION
            await emit({"type": "log", "stage": "verify", "level": "warn",
                        "message": f"faithfulness {faithfulness:.2f} < {settings.RAG_VERIFY_MIN} "
                                   f"({len(v.unsupported)} unsupported) — action={action}"})
            if action == "refuse":
                return _refusal(question, plan.language, plan.subject, prompts.REFUSAL[plan.language],
                                started, llm_results, reason="answer failed verification",
                                best_sim=best_sim, reformulations=loops,
                                latency={"reformulate_s": reformulate_s, "retrieve_s": retrieve_s,
                                         "generate_s": generate_s})
            if action == "revise" and can_call():
                rev = await llm.chat(prompts.build_revise_messages(
                    question, selected, answer_text, v.unsupported, plan.language))
                llm_results.append(rev)
                answer_text = sanitize_answer(rev.text)
                revised = True
        elif faithfulness is not None:
            await emit({"type": "log", "stage": "verify", "message": f"faithfulness {faithfulness:.2f}"})

    # 11. Aggregate + return ---------------------------------------------
    metrics = _aggregate_metrics(
        llm_results, generation,
        {"reformulate_s": reformulate_s, "retrieve_s": retrieve_s,
         "generate_s": generate_s, "total_s": perf_counter() - started},
        model=await llm.model(),
        best_similarity=round(best_sim, 3),
        agentic=True,
        loops=loops,
        rerank_backend=settings.RAG_RERANK_BACKEND if reranker else "none",
        relevant_fraction=round(grade.relevant_fraction, 3) if grade else None,
        context_sufficient=grade.sufficient if grade else None,
        faithfulness=round(faithfulness, 3) if faithfulness is not None else None,
        revised=revised,
        context_chunks=len(selected),
    )
    return Answer(
        question=question,
        answer=answer_text,
        refused=False,
        language=plan.language,
        subject=plan.subject,
        queries=tried,
        citations=[_citation(i, c) for i, c in enumerate(selected, 1)],
        contexts=[_context(i, c) for i, c in enumerate(selected, 1)],
        metrics=metrics,
    )
