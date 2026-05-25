"""Async HTTP API for the Manage Materials screen: list / browse / upload (SSE) /
freeze / unfreeze / delete / search. Mirrors the rag app's async + SSE pattern
(``view.csrf_exempt = True`` attribute, NOT the decorator — the decorator breaks
async views). Sync ORM / OCR / embed work is offloaded via ``asyncio.to_thread``.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from django.http import JsonResponse, StreamingHttpResponse

from apps.catalog.models import Book
from apps.catalog.services import materials
from apps.rag.services.guard import check_question

_JSON = {"ensure_ascii": False}


def _err(message: str, status: int = 400):
    return JsonResponse({"error": message}, status=status)


# --- list / detail / delete -------------------------------------------------
async def materials_list_view(request):
    if request.method != "GET":
        return _err("GET required", 405)
    g = request.GET
    books = await asyncio.to_thread(
        materials.list_books,
        status=g.get("status") or None, subject=g.get("subject") or None,
        language=g.get("language") or None, q=g.get("q") or None,
    )
    return JsonResponse(
        {"books": books, "ingest_in_progress": materials.ingest_in_progress()},
        json_dumps_params=_JSON,
    )


materials_list_view.csrf_exempt = True


async def materials_detail_view(request, book_id: int):
    if request.method == "DELETE":
        try:
            return JsonResponse(await asyncio.to_thread(materials.delete_book, book_id))
        except Book.DoesNotExist:
            return _err("book not found", 404)
    if request.method != "GET":
        return _err("GET or DELETE required", 405)
    try:
        offset = max(0, int(request.GET.get("offset", 0)))
        limit = min(max(1, int(request.GET.get("limit", 50))), 200)
    except ValueError:
        offset, limit = 0, 50
    try:
        detail = await asyncio.to_thread(materials.book_detail, book_id, offset=offset, limit=limit)
    except Book.DoesNotExist:
        return _err("book not found", 404)
    return JsonResponse(detail, json_dumps_params=_JSON)


materials_detail_view.csrf_exempt = True


async def materials_status_view(request, book_id: int, action: str):
    if request.method != "POST":
        return _err("POST required", 405)
    status = {"freeze": "frozen", "unfreeze": "active"}.get(action)
    if status is None:
        return _err("unknown action", 404)
    try:
        return JsonResponse(await asyncio.to_thread(materials.set_status, book_id, status))
    except Book.DoesNotExist:
        return _err("book not found", 404)


materials_status_view.csrf_exempt = True


# --- dedup precheck + corpus search ----------------------------------------
async def materials_check_view(request):
    if request.method != "POST":
        return _err("POST required", 405)
    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return _err("invalid JSON body")
    match = await asyncio.to_thread(
        materials.precheck,
        title=payload.get("title", ""), language=payload.get("language", ""),
        subject_code=payload.get("subject", ""),
    )
    return JsonResponse({"match": match}, json_dumps_params=_JSON)


materials_check_view.csrf_exempt = True


def _search(query: str, language, subject) -> list[dict]:
    from django.conf import settings

    from apps.rag.services.pipeline import get_retriever

    selected, _best = get_retriever().retrieve(
        [query], language, subject if settings.RAG_SUBJECT_FILTER else None,
        candidates=settings.RAG_CANDIDATES, top_k=settings.RAG_TOP_K,
        token_budget=settings.RAG_MAX_CONTEXT_TOKENS,
    )
    return [
        {
            "book": c.book_title, "subject": c.subject, "language": c.language,
            "page_start": c.page_start, "page_end": c.page_end, "heading": c.heading_path,
            "sim": round(c.dense_sim, 3), "snippet": c.content[:300],
        }
        for c in selected
    ]


async def materials_search_view(request):
    if request.method != "POST":
        return _err("POST required", 405)
    try:
        payload = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return _err("invalid JSON body")
    guard = check_question(payload.get("q") or "")
    if not guard.ok:
        return _err(guard.reason or "invalid query")
    hits = await asyncio.to_thread(
        _search, guard.text, payload.get("language") or None, payload.get("subject") or None
    )
    return JsonResponse({"hits": hits}, json_dumps_params=_JSON)


materials_search_view.csrf_exempt = True


# --- upload (multipart -> SSE progress) ------------------------------------
async def materials_upload_view(request):
    if request.method != "POST":
        return _err("POST required", 405)
    upload = request.FILES.get("file")
    if upload is None:
        return _err("'file' is required (multipart form field 'file').")
    form = request.POST
    language = (form.get("language") or "").strip()
    subject = (form.get("subject") or "").strip()
    title = (form.get("title") or "").strip()
    level = (form.get("level") or "brevet").strip()
    resolution = form.get("resolution") or None
    target_raw = form.get("target_id") or ""
    target_id = int(target_raw) if target_raw.isdigit() else None

    if language not in ("en", "fr"):
        return _err("'language' must be 'en' or 'fr'.")
    if not subject:
        return _err("'subject' is required.")

    # Save to disk, then validate (magic bytes read from the saved file).
    saved = await asyncio.to_thread(materials.save_upload, language, upload.name, upload.chunks())
    try:
        with open(saved, "rb") as fh:
            head = fh.read(8)
        await asyncio.to_thread(materials.validate_upload, upload.name, upload.size, head)
    except materials.UploadError as exc:
        Path(saved).unlink(missing_ok=True)
        return _err(str(exc))

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_stage(stage: str, message: str, **extra) -> None:
        loop.call_soon_threadsafe(
            queue.put_nowait, {"type": "stage", "stage": stage, "message": message, **extra}
        )

    async def run() -> None:
        try:
            result = await asyncio.to_thread(
                materials.run_upload, src_path=str(saved), language=language, subject_code=subject,
                title=title, level=level, resolution=resolution, target_id=target_id, on_stage=on_stage,
            )
            await queue.put({"type": "done", **result})
        except materials.NeedsDecision as nd:
            await queue.put({"type": "needs_decision", "match": nd.match})
        except materials.UploadError as exc:
            await queue.put({"type": "error", "error": str(exc)})
        except Exception as exc:  # surface unexpected failures to the UI
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
    response["X-Accel-Buffering"] = "no"
    return response


materials_upload_view.csrf_exempt = True
