"""Start Brevet-GPT: health-check the moving parts, then serve the async API
over ASGI (uvicorn).

    python manage.py brevet
    python manage.py brevet --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio

from django.core.management.base import BaseCommand, CommandParser

from apps.rag.services.pipeline import health


class Command(BaseCommand):
    help = "Health-check then run the Brevet-GPT async API server (uvicorn/ASGI)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--reload", action="store_true")

    def handle(self, *args: object, **options: object) -> None:
        self.stdout.write(self.style.MIGRATE_HEADING("\nBrevet-GPT"))
        status = asyncio.run(health())
        self.stdout.write(f"  corpus : {status.get('vectors')} vectors / {status.get('chunks')} chunks")
        if status.get("llm_ok"):
            llm = status["llm"]
            self.stdout.write(self.style.SUCCESS(f"  llm    : {llm['model']} @ {llm['base_url']}"))
        else:
            self.stdout.write(self.style.WARNING(
                f"  llm    : DOWN — {status.get('llm_error', 'unknown')}\n"
                f"           Start LM Studio, load a model, enable the local server."
            ))

        host, port = options["host"], options["port"]
        self.stdout.write(self.style.SUCCESS(
            f"\nServing http://{host}:{port}  —  POST /api/ask · GET /api/health\n"
        ))
        import uvicorn

        uvicorn.run("config.asgi:application", host=host, port=port, reload=options["reload"])
