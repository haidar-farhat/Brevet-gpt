"""Evaluate the RAG pipeline: RAGAS-faithful quality metrics + performance.

    python manage.py rag_eval                 # full eval set
    python manage.py rag_eval --language fr --limit 3
    python manage.py rag_eval --no-judge      # performance metrics only (no judge)

Quality metrics (0-1): faithfulness, answer relevancy, context precision.
Performance: tokens, generation tokens/sec, per-stage latency, total wall time.
Requires LM Studio running (used both to answer and to judge).
"""
from __future__ import annotations

import asyncio
from time import perf_counter

from django.core.management.base import BaseCommand, CommandError, CommandParser

from apps.rag.data.eval_questions import EVAL_QUESTIONS
from apps.rag.services import evaluation
from apps.rag.services.llm import LLMUnavailable, LMStudioClient
from apps.rag.services.pipeline import answer_question, get_retriever


def _avg(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "  -  "


class Command(BaseCommand):
    help = "Evaluate the RAG pipeline (RAGAS-faithful metrics + performance)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--language", choices=("en", "fr"), default=None)
        parser.add_argument("--subject", default=None)
        parser.add_argument("--no-judge", action="store_true",
                            help="Skip quality metrics; report performance only.")

    def handle(self, *args: object, **options: object) -> None:
        try:
            asyncio.run(self._run(options))
        except LLMUnavailable as exc:
            raise CommandError(str(exc)) from exc

    async def _run(self, options: dict) -> None:
        items = [
            q for q in EVAL_QUESTIONS
            if (not options["language"] or q.get("language") == options["language"])
            and (not options["subject"] or q.get("subject") == options["subject"])
        ]
        if options["limit"]:
            items = items[: options["limit"]]
        if not items:
            raise CommandError("No eval questions match the given filters.")

        judge = LMStudioClient()
        embedder = get_retriever().embedder

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"\nEvaluating {len(items)} question(s)\n"
            f"{'faith':>6} {'relev':>6} {'ctxP':>6} {'tok':>6} {'tok/s':>6} {'gen_s':>6}  question"
        ))

        scores_all, perf = [], []
        wall = perf_counter()
        for item in items:
            answer = await answer_question(item["question"], language=item.get("language"),
                                           subject=item.get("subject"))
            row = answer.metrics
            perf.append(row)
            scores = None
            if not options["no_judge"] and not answer.refused:
                scores = await evaluation.evaluate_sample(
                    judge, embedder, question=item["question"], answer=answer.answer,
                    contexts=[c["content"] for c in answer.contexts],
                )
                scores_all.append(scores)

            lat = row.get("latency", {})
            tag = "REFUSED " if answer.refused else ""
            self.stdout.write(
                f"{_fmt(scores.faithfulness) if scores else '  -  ':>6} "
                f"{_fmt(scores.answer_relevancy) if scores else '  -  ':>6} "
                f"{_fmt(scores.context_precision) if scores else '  -  ':>6} "
                f"{row.get('total_tokens', 0):>6} "
                f"{row.get('generation_tokens_per_sec', 0):>6.1f} "
                f"{lat.get('generate_s', 0):>6.1f}  {tag}{item['question'][:50]}"
            )

        self._summary(scores_all, perf, perf_counter() - wall, options["no_judge"])

    def _summary(self, scores_all, perf, wall: float, no_judge: bool) -> None:
        self.stdout.write(self.style.MIGRATE_HEADING("\n== SUMMARY =="))
        if not no_judge and scores_all:
            self.stdout.write(self.style.SUCCESS(
                f"  faithfulness      {_fmt(_avg([s.faithfulness for s in scores_all]))}\n"
                f"  answer_relevancy  {_fmt(_avg([s.answer_relevancy for s in scores_all]))}\n"
                f"  context_precision {_fmt(_avg([s.context_precision for s in scores_all]))}"
            ))

        def stage(name: str) -> float:
            return _avg([p.get("latency", {}).get(name) for p in perf]) or 0.0

        total_tokens = sum(p.get("total_tokens", 0) for p in perf)
        refused = sum(1 for p in perf if p.get("refused_reason"))
        self.stdout.write(
            f"  questions         {len(perf)} ({refused} refused)\n"
            f"  total tokens      {total_tokens:,}\n"
            f"  mean gen tok/s    {_avg([p.get('generation_tokens_per_sec') for p in perf]) or 0:.1f}\n"
            f"  mean latency      reformulate {stage('reformulate_s'):.2f}s | "
            f"retrieve {stage('retrieve_s'):.2f}s | generate {stage('generate_s'):.2f}s | "
            f"total {stage('total_s'):.2f}s\n"
            f"  wall time         {wall:.1f}s"
        )
