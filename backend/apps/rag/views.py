"""Async HTTP API for Brevet-GPT (plain Django async views)."""
from __future__ import annotations

import asyncio
import json

from django.http import JsonResponse, StreamingHttpResponse

from apps.rag.services.llm import LLMUnavailable
from apps.rag.services.pipeline import answer_question, health

_JSON = {"ensure_ascii": False}


def _read_question(request):
    """Parse + validate the {question, language?, subject?, top_k?} body."""
    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return None, JsonResponse({"error": "invalid JSON body"}, status=400)
    if not (payload.get("question") or "").strip():
        return None, JsonResponse({"error": "'question' is required"}, status=400)
    return payload, None


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


async def ask_stream_view(request):
    """Server-Sent Events: streams pipeline log events + answer tokens live,
    ending with a 'result' event carrying citations + metrics."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    payload, error = _read_question(request)
    if error is not None:
        return error

    queue: asyncio.Queue = asyncio.Queue()

    async def on_event(event: dict) -> None:
        await queue.put(event)

    async def run() -> None:
        try:
            answer = await answer_question(
                payload["question"].strip(), language=payload.get("language"),
                subject=payload.get("subject"), top_k=payload.get("top_k"), on_event=on_event,
            )
            await queue.put({"type": "result", **answer.to_dict()})
        except LLMUnavailable as exc:
            await queue.put({"type": "error", "error": str(exc)})
        except Exception as exc:  # surface unexpected errors to the terminal
            await queue.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            await queue.put(None)

    async def stream():
        task = asyncio.create_task(run())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            if not task.done():
                task.cancel()

    response = StreamingHttpResponse(stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"  # disable proxy buffering
    return response


ask_stream_view.csrf_exempt = True


async def health_view(request):
    return JsonResponse(await health(), json_dumps_params=_JSON)
