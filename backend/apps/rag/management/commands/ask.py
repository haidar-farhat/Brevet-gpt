"""Ask Brevet-GPT a question from the CLI (one-shot or interactive REPL).

    python manage.py ask "What is photosynthesis?"
    python manage.py ask "Résous 2x + 3 = 7" --language fr --subject math
    python manage.py ask --json "define an isotope"
    python manage.py ask                      # interactive
"""
from __future__ import annotations

import asyncio
import json

from django.core.management.base import BaseCommand, CommandError, CommandParser

from apps.rag.services.llm import LLMUnavailable
from apps.rag.services.pipeline import Answer, answer_question


class Command(BaseCommand):
    help = "Ask Brevet-GPT a question (one-shot or interactive)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("question", nargs="*", help="Question text; omit for interactive mode.")
        parser.add_argument("--language", choices=("en", "fr"), default=None)
        parser.add_argument("--subject", default=None)
        parser.add_argument("--top-k", type=int, default=None, dest="top_k")
        parser.add_argument("--json", action="store_true", dest="as_json")

    def handle(self, *args: object, **options: object) -> None:
        question = " ".join(options["question"]).strip()
        try:
            asyncio.run(self._answer(question, options) if question else self._repl(options))
        except LLMUnavailable as exc:
            raise CommandError(str(exc)) from exc
        except (KeyboardInterrupt, EOFError):
            self.stdout.write("\nbye")

    async def _answer(self, question: str, options: dict) -> None:
        answer = await answer_question(
            question, language=options["language"], subject=options["subject"], top_k=options["top_k"]
        )
        self._render(answer, options["as_json"])

    async def _repl(self, options: dict) -> None:
        self.stdout.write(self.style.MIGRATE_HEADING(
            "Brevet-GPT — ask a question ('exit' or Ctrl-C to quit)"
        ))
        while True:
            question = (await asyncio.to_thread(input, "\n? ")).strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            try:
                await self._answer(question, options)
            except LLMUnavailable as exc:
                self.stdout.write(self.style.ERROR(str(exc)))

    def _render(self, answer: Answer, as_json: bool) -> None:
        if as_json:
            self.stdout.write(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
            return

        self.stdout.write("\n" + answer.answer)
        if answer.citations and not answer.refused:
            self.stdout.write(self.style.HTTP_INFO("\nSources:"))
            for c in answer.citations:
                pages = f"p.{c['page_start']}" + (f"-{c['page_end']}" if c["page_end"] != c["page_start"] else "")
                head = f" — {c['heading']}" if c["heading"] else ""
                self.stdout.write(f"  [{c['n']}] {c['subject']} — {c['book']} {pages}{head}")

        m = answer.metrics
        lat = m.get("latency", {})
        self.stdout.write(self.style.WARNING(
            f"\n[{m.get('model', '?')}] tokens={m.get('total_tokens', 0)} "
            f"(gen {m.get('generation_tokens_per_sec', 0)} tok/s) | "
            f"reformulate {lat.get('reformulate_s', '?')}s retrieve {lat.get('retrieve_s', '?')}s "
            f"generate {lat.get('generate_s', '?')}s total {lat.get('total_s', '?')}s | "
            f"ctx={m.get('context_chunks', 0)} sim={m.get('best_similarity', '?')} "
            f"reforms={m.get('reformulations', 0)}"
        ))
