"""Async HTTP API for Brevet-GPT (plain Django async views)."""
from __future__ import annotations

import json

from django.http import JsonResponse

from apps.rag.services.llm import LLMUnavailable
from apps.rag.services.pipeline import answer_question, health

_JSON = {"ensure_ascii": False}


async def ask_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid JSON body"}, status=400)

    question = (payload.get("question") or "").strip()
    if not question:
        return JsonResponse({"error": "'question' is required"}, status=400)

    try:
        answer = await answer_question(
            question,
            language=payload.get("language"),
            subject=payload.get("subject"),
            top_k=payload.get("top_k"),
        )
    except LLMUnavailable as exc:
        return JsonResponse({"error": str(exc)}, status=503)

    return JsonResponse(answer.to_dict(), json_dumps_params=_JSON)


# Set the attribute directly instead of using @csrf_exempt: the decorator wraps
# the function so Django no longer sees it as a coroutine and calls it
# synchronously (returning an unawaited coroutine). Setting the flag keeps the
# async view intact while CsrfViewMiddleware still skips it.
ask_view.csrf_exempt = True


async def health_view(request):
    return JsonResponse(await health(), json_dumps_params=_JSON)
