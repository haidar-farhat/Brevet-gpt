"""Semantic answer cache.

Stores each answered question's bge-m3 embedding + the full Answer (as JSON) in a
dedicated Chroma collection. A new question is embedded and matched against the
cache; a hit (cosine >= RAG_CACHE_MIN_SIM, default 0.97 — only near-identical
phrasings) returns the stored answer and skips the whole pipeline.

Matching is embedding-only: the question embedding already captures language and
topic, and the high threshold prevents serving a cached answer for a genuinely
different question. All operations are sync (run via asyncio.to_thread) and wrapped
so a cache failure never breaks answering.
"""
from __future__ import annotations

import json
import uuid

from django.conf import settings

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        from apps.catalog.services.vectorstore import get_collection

        _collection = get_collection(settings.CHROMA_DIR, settings.RAG_CACHE_COLLECTION)
    return _collection


class SemanticCache:
    def __init__(self, min_sim: float) -> None:
        self.min_sim = min_sim

    def get(self, query_vec, language=None, subject=None, grade=None):
        """Return a cached Answer for a near-identical question, else None.

        ``grade`` is matched exactly (via a Chroma ``where``): unlike language/topic
        — which the embedding already captures — the grade scope narrows the *corpus*
        without changing the question text, so a g7-scoped query must not reuse a
        g9-scoped answer."""
        col = _get_collection()
        try:
            if col.count() == 0:
                return None
            res = col.query(query_embeddings=[query_vec], n_results=1,
                            where={"grade": grade or ""}, include=["distances", "documents"])
        except Exception:
            return None
        ids = (res.get("ids") or [[]])[0]
        if not ids:
            return None
        distance = (res.get("distances") or [[1.0]])[0][0]
        if (1.0 - float(distance)) < self.min_sim:
            return None
        document = (res.get("documents") or [[None]])[0][0]
        if not document:
            return None
        try:
            payload = json.loads(document)
        except json.JSONDecodeError:
            return None

        from apps.rag.services.pipeline import Answer

        payload.setdefault("metrics", {})
        payload["metrics"]["cached"] = True
        try:
            return Answer(**payload)
        except TypeError:
            return None  # schema drift — treat as a miss

    def put(self, question, query_vec, answer, language=None, subject=None, grade=None) -> None:
        col = _get_collection()
        try:
            col.upsert(
                ids=[str(uuid.uuid4())],
                embeddings=[query_vec],
                documents=[json.dumps(answer.to_dict(), ensure_ascii=False)],
                metadatas=[{
                    "language": language or answer.language or "",
                    "subject": subject or answer.subject or "",
                    "grade": grade or "",
                }],
            )
        except Exception:
            pass  # cache writes are best-effort


_cache: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache(settings.RAG_CACHE_MIN_SIM)
    return _cache
