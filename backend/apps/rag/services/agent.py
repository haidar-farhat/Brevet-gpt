"""Agentic answer orchestrator (Corrective / Self-RAG).

Flow: plan/route -> retrieve candidates -> rerank -> grade context (analyse) ->
refine+re-retrieve loop (if insufficient) -> then EITHER the SOLVE branch
(decompose a problem into sub-problems and solve each with a small focused
context, for math/physics/chemistry exercises) OR reason -> generate ->
self-verify. Generation is guarded so it never returns a blank answer (it retries
with a leaner prompt, then falls back to a friendly message). Every optional step
is gated by a hard LLM-call budget so a slow CPU model can't blow up, and emits a
live SSE log for the web terminal.

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
from apps.rag.services.rerank import dense_rerank, rerank_chunks


async def agentic_answer(question: str, *, language, subject, top_k, on_event,
                         llm: LMStudioClient, started: float, llm_results: list[LLMResult]):
    from apps.rag.services.pipeline import (  # lazy: avoid circular import
        Answer, _aggregate_metrics, _citation, _context, _refusal, _terminal,
        get_reranker, get_retriever, nonempty_answer,
    )

    async def emit(event: dict) -> None:
        if on_event is not None:
            await on_event(event)

    # LLM-call budget; raised in the solve branch so "solve all parts" is never cut short.
    budget = {"max": settings.RAG_AGENT_LLM_BUDGET}

    def can_call() -> bool:
        return len(llm_results) < budget["max"]

    async def generate_guarded(messages, *, max_tokens=None, label="generate",
                               stream_as="token", rebuild_leaner=None):
        """Generate text, streaming each token as ``stream_as`` when on_event is
        set. Never blanks: if the model streams zero tokens, retry once with a
        leaner prompt (budget permitting). Appends every LLMResult to llm_results.
        Returns (sanitized_text, last_result)."""
        async def run(msgs):
            if on_event is not None:
                return await llm.chat_stream(
                    msgs, lambda d: emit({"type": stream_as, "text": d}), max_tokens=max_tokens)
            return await llm.chat(msgs, max_tokens=max_tokens)

        result = await run(messages)
        llm_results.append(result)
        text = sanitize_answer(result.text)
        if not text and settings.RAG_SOLVE_RETRY_EMPTY and can_call():
            await emit({"type": "log", "stage": label, "level": "warn",
                        "message": "empty output — retrying with a leaner prompt"})
            result = await run(rebuild_leaner() if rebuild_leaner else messages)
            llm_results.append(result)
            text = sanitize_answer(result.text)
        return text, result

    retriever = get_retriever()
    reranker = get_reranker()
    candidates_k = settings.RAG_CANDIDATES
    k = top_k or settings.RAG_TOP_K
    budget_tokens = settings.RAG_MAX_CONTEXT_TOKENS
    min_rel = settings.RAG_MIN_RELEVANCE

    # 1. Analyse / route: scope, clarification, and step-back query plan ---
    catalog = await asyncio.to_thread(reformulation.get_materials_catalog)
    t0 = perf_counter()
    analysis, analysis_result = await reformulation.analyze_query(llm, question, language, subject, catalog=catalog)
    reformulate_s = perf_counter() - t0
    llm_results.append(analysis_result)
    await emit({"type": "log", "stage": "route", "language": analysis.language, "subject": analysis.subject,
                "queries": analysis.queries, "latency_s": round(reformulate_s, 3),
                "in_scope": analysis.in_scope})

    # Off-topic guard: decline early, before spending retrieval/generation.
    if settings.RAG_SCOPE_GUARD and not analysis.in_scope:
        await emit({"type": "log", "stage": "route", "level": "warn", "message": "off-topic — declining"})
        return _terminal(question, analysis.language, analysis.subject,
                         prompts.OUT_OF_SCOPE[analysis.language], started, llm_results,
                         status="out_of_scope", refused=True, reason="off-topic",
                         latency={"reformulate_s": reformulate_s})

    # Clarification: ask back instead of guessing when the request is vague.
    if settings.RAG_CLARIFY and analysis.needs_clarification:
        await emit({"type": "log", "stage": "route", "message": "needs clarification — asking back"})
        return _terminal(question, analysis.language, analysis.subject, analysis.clarification,
                         started, llm_results, status="clarify", refused=False,
                         reason="needs clarification", latency={"reformulate_s": reformulate_s})

    # rebind retrieve to the resolved language
    async def do_retrieve(queries, subj):
        # Only constrain by subject when explicitly trusted; otherwise search all
        # subjects so a mis-route can't exclude the right chunks.
        effective_subj = subj if settings.RAG_SUBJECT_FILTER else None
        cands, sim = await asyncio.to_thread(
            retriever.retrieve_candidates, queries, analysis.language, effective_subj, candidates=candidates_k
        )
        mode = None
        if cands:
            if reranker is not None:  # cross-encoder (opt-in; needs the extra model)
                head = await asyncio.to_thread(
                    rerank_chunks, reranker, question, cands[: settings.RAG_RERANK_CANDIDATES],
                    dense_weight=settings.RAG_DENSE_SIM_WEIGHT,
                )
                cands = head + cands[settings.RAG_RERANK_CANDIDATES:]
                mode = "cross_encoder"
            elif settings.RAG_RERANK and settings.RAG_RERANK_BACKEND.lower() == "dense":
                cands = dense_rerank(cands, dense_weight=settings.RAG_DENSE_SIM_WEIGHT)
                mode = "dense"
        return retriever.select_within_budget(cands, top_k=k, token_budget=budget_tokens), sim, mode

    # 2-4. Retrieve + rerank ---------------------------------------------
    t_retr = perf_counter()
    selected, best_sim, rerank_mode = await do_retrieve(analysis.queries, analysis.subject)
    if rerank_mode:
        await emit({"type": "log", "stage": "rerank",
                    "message": f"reranked candidates ({rerank_mode}) — "
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
    tried = list(analysis.queries)
    loops = 0
    while loops < settings.RAG_AGENT_MAX_LOOPS and can_call() and (
        (grade is not None and not grade.sufficient) or (grade is None and best_sim < min_rel)
    ):
        missing = grade.missing if grade else ""
        new_queries, refine_result = await reformulation.refine_queries(
            llm, question, missing=missing, prior_queries=tried, language=analysis.language
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
        return _refusal(question, analysis.language, analysis.subject, prompts.REFUSAL[analysis.language],
                        started, llm_results, reason="no relevant context", best_sim=best_sim,
                        reformulations=loops,
                        latency={"reformulate_s": reformulate_s, "retrieve_s": retrieve_s})

    # 8. SOLVE branch: decompose a problem and solve each part with a small,
    # focused context. Skips reason + verify on purpose — the per-part solve IS
    # the reasoning, and small prompts avoid the empty-output overload that a
    # monolith prompt causes on a small local model. ---------------------
    # The LLM's is_problem flag is unreliable on a small model, so OR it with a
    # deterministic heuristic (solve-verbs / numbered parts + equations).
    solving = (settings.RAG_SOLVE
               and (analysis.is_problem or reformulation.looks_like_problem(question))
               and analysis.subject in settings.RAG_REASON_SUBJECTS)
    if solving:
        from apps.catalog.services.chunking import count_tokens

        # Decompose with a dedicated call (routing JSON is kept small for
        # reliability); fall back to solving the whole question if it yields nothing.
        parts = list(analysis.sub_problems)
        if not parts and can_call():
            await emit({"type": "log", "stage": "solve", "message": "decomposing the problem into parts"})
            parts, dec_result = await reformulation.decompose_problem(llm, question, language=analysis.language)
            llm_results.append(dec_result)
        if not parts:
            parts = [question]
        capped = len(parts) > settings.RAG_MAX_SUBPROBLEMS
        parts = parts[: settings.RAG_MAX_SUBPROBLEMS]
        budget["max"] = max(budget["max"], len(parts) + 6)  # decompose + per-part + assemble + slack
        multi = len(parts) > 1
        # Every part shares the topic's retrieved rules; trim once to a small window.
        ctx = retriever.select_within_budget(
            selected, top_k=settings.RAG_SOLVE_TOP_K, token_budget=settings.RAG_SOLVE_CONTEXT_TOKENS)
        await emit({"type": "log", "stage": "solve",
                    "message": f"solving {len(parts)} part(s) step by step"})

        t_solve = perf_counter()
        solved: list[str] = []
        last_result: LLMResult | None = None
        for i, part in enumerate(parts, 1):
            if not can_call():
                break
            await emit({"type": "log", "stage": "solve", "message": f"Part {i}/{len(parts)}: {part[:80]}"})
            msgs = prompts.build_solve_messages(part, ctx, analysis.language)
            leaner = (lambda p=part: prompts.build_solve_messages(p, ctx[:1], analysis.language))
            # Multi-part: stream the working into the "Thinking" segment; the clean
            # combined answer is streamed separately below (avoids duplication).
            text, last_result = await generate_guarded(
                msgs, max_tokens=settings.RAG_SOLVE_MAX_TOKENS, label="solve",
                stream_as=("reason_token" if multi else "token"), rebuild_leaner=leaner)
            solved.append(f"## Part {i}\n\n{text}" if multi else text)

        # Assemble: LLM-stitch only when the combined parts are small enough to
        # stay reliable; otherwise join locally (also the path when the budget
        # is spent). Either way the answer is complete and never blank.
        if multi:
            joined = "\n\n".join(solved)
            if can_call() and solved and count_tokens(joined) <= settings.RAG_SOLVE_CONTEXT_TOKENS:
                await emit({"type": "log", "stage": "solve", "message": "assembling the full answer"})
                answer_text, last_result = await generate_guarded(
                    prompts.build_assemble_messages(question, solved, analysis.language),
                    label="solve", stream_as="token")
                if not answer_text:
                    answer_text = joined
            else:
                answer_text = joined
        else:
            answer_text = solved[0] if solved else ""

        if capped:
            answer_text += {"en": f"\n\n_(Showing the first {len(parts)} parts.)_",
                            "fr": f"\n\n_(Affichage des {len(parts)} premières parties.)_"}[analysis.language]
        answer_text = nonempty_answer(answer_text, analysis.language)
        solve_s = perf_counter() - t_solve

        metrics = _aggregate_metrics(
            llm_results, last_result,
            {"reformulate_s": reformulate_s, "retrieve_s": retrieve_s,
             "generate_s": solve_s, "total_s": perf_counter() - started},
            model=await llm.model(),
            best_similarity=round(best_sim, 3),
            agentic=True, loops=loops, rerank_backend=rerank_mode or "none",
            relevant_fraction=round(grade.relevant_fraction, 3) if grade else None,
            context_sufficient=grade.sufficient if grade else None,
            faithfulness=None, revised=False, context_chunks=len(ctx),
            solved_parts=len(solved), subproblems=len(parts),
        )
        return Answer(
            question=question, answer=answer_text, refused=False,
            language=analysis.language, subject=analysis.subject, queries=tried,
            citations=[_citation(i, c) for i, c in enumerate(ctx, 1)],
            contexts=[_context(i, c) for i, c in enumerate(ctx, 1)],
            metrics=metrics,
        )

    # 9. Reason (problem-solving subjects, non-solve path) ----------------
    reasoning = None
    if settings.RAG_REASON and analysis.subject in settings.RAG_REASON_SUBJECTS and can_call():
        await emit({"type": "log", "stage": "reason", "message": f"reasoning over rules ({analysis.subject})"})
        reason_messages = prompts.build_reason_messages(question, selected, analysis.language)
        if on_event is not None:
            # Stream the reasoning so the UI's "Thinking" segment fills live.
            reason_result = await llm.chat_stream(
                reason_messages, lambda d: emit({"type": "reason_token", "text": d}),
                max_tokens=settings.RAG_REASON_MAX_TOKENS,
            )
        else:
            reason_result = await llm.chat(reason_messages, max_tokens=settings.RAG_REASON_MAX_TOKENS)
        llm_results.append(reason_result)
        reasoning = reason_result.text

    # 10. Generate (guarded: retries leaner on empty, never blanks) -------
    await emit({"type": "log", "stage": "generate", "message": f"generating with {await llm.model()}"})
    t0 = perf_counter()
    messages = prompts.build_answer_messages(question, selected, analysis.language, reasoning=reasoning)

    def leaner_answer():
        # Drop the reasoning block and halve the context if the full prompt blanks.
        half = selected[: max(1, len(selected) // 2)]
        return prompts.build_answer_messages(question, half, analysis.language, reasoning=None)

    answer_text, generation = await generate_guarded(
        messages, label="generate", stream_as="token", rebuild_leaner=leaner_answer)
    generate_s = perf_counter() - t0

    # 11. Self-verify -----------------------------------------------------
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
                return _refusal(question, analysis.language, analysis.subject, prompts.REFUSAL[analysis.language],
                                started, llm_results, reason="answer failed verification",
                                best_sim=best_sim, reformulations=loops,
                                latency={"reformulate_s": reformulate_s, "retrieve_s": retrieve_s,
                                         "generate_s": generate_s})
            if action == "revise" and can_call():
                rev = await llm.chat(prompts.build_revise_messages(
                    question, selected, answer_text, v.unsupported, analysis.language))
                llm_results.append(rev)
                answer_text = sanitize_answer(rev.text)
                revised = True
        elif faithfulness is not None:
            await emit({"type": "log", "stage": "verify", "message": f"faithfulness {faithfulness:.2f}"})

    answer_text = nonempty_answer(answer_text, analysis.language)

    # 12. Aggregate + return ---------------------------------------------
    metrics = _aggregate_metrics(
        llm_results, generation,
        {"reformulate_s": reformulate_s, "retrieve_s": retrieve_s,
         "generate_s": generate_s, "total_s": perf_counter() - started},
        model=await llm.model(),
        best_similarity=round(best_sim, 3),
        agentic=True,
        loops=loops,
        rerank_backend=rerank_mode or "none",
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
        language=analysis.language,
        subject=analysis.subject,
        queries=tried,
        citations=[_citation(i, c) for i, c in enumerate(selected, 1)],
        contexts=[_context(i, c) for i, c in enumerate(selected, 1)],
        metrics=metrics,
    )
