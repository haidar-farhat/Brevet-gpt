"""Hybrid retrieval: Chroma dense (bge-m3) + MySQL FULLTEXT lexical, fused with
Reciprocal Rank Fusion, then selected within a token budget and assembled with
citations. Pure sync — the async pipeline runs it via asyncio.to_thread, which
keeps the Django ORM and Chroma calls off the event loop and thread-safe.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass

from django.db import connection

from apps.catalog.models import Chunk

RRF_K = 60


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: int
    vector_id: str
    book_title: str
    subject: str
    language: str
    page_start: int
    page_end: int
    heading_path: str
    content: str
    dense_sim: float
    score: float


def _uid(value) -> uuid.UUID:
    """Normalize a vector id to a UUID. Chroma stores dashed strings; Django's
    MySQL UUIDField stores 32 hex chars — UUID() accepts both."""
    return uuid.UUID(str(value))


class HybridRetriever:
    def __init__(self, embedder, collection) -> None:
        self.embedder = embedder
        self.collection = collection

    @staticmethod
    def _where(language: str | None, subject: str | None):
        conds = []
        if language:
            conds.append({"language": language})
        if subject:
            conds.append({"subject": subject})
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def _dense(self, query: str, where, k: int) -> list[tuple[uuid.UUID, float]]:
        vector = self.embedder.embed([query])[0]
        res = self.collection.query(
            query_embeddings=[vector], n_results=k, where=where, include=["distances"]
        )
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return [(_uid(i), 1.0 - float(d)) for i, d in zip(ids, dists)]

    def _lexical(self, query: str, language, subject, k: int) -> list[tuple[uuid.UUID, float]]:
        sql = [
            "SELECT vector_id, MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE) AS score",
            "FROM chunks",
            "WHERE MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE)",
        ]
        params: list = [query, query]
        if language:
            sql.append("AND language = %s")
            params.append(language)
        if subject:
            sql.append("AND subject_id = (SELECT id FROM subjects WHERE code = %s)")
            params.append(subject)
        sql.append("ORDER BY score DESC LIMIT %s")
        params.append(k)
        with connection.cursor() as cursor:
            cursor.execute(" ".join(sql), params)
            rows = cursor.fetchall()
        return [(_uid(vid), float(score)) for vid, score in rows if vid is not None]

    def retrieve(self, queries: list[str], language: str | None, subject: str | None, *,
                 candidates: int, top_k: int, token_budget: int) -> tuple[list[RetrievedChunk], float]:
        where = self._where(language, subject)
        dense_sim: dict[uuid.UUID, float] = {}
        rank_lists: list[list[uuid.UUID]] = []

        for query in queries:
            dense_hits = self._dense(query, where, candidates)
            for uid, sim in dense_hits:
                dense_sim[uid] = max(dense_sim.get(uid, -1.0), sim)
            rank_lists.append([uid for uid, _ in dense_hits])
            rank_lists.append([uid for uid, _ in self._lexical(query, language, subject, candidates)])

        fused = _rrf(rank_lists)
        if not fused:
            return [], 0.0

        rrf_score = dict(fused)
        chunks = {
            _uid(c.vector_id): c
            for c in Chunk.objects.filter(vector_id__in=[uid for uid, _ in fused]).select_related("book", "subject")
        }

        selected: list[RetrievedChunk] = []
        used_tokens = 0
        for uid, _ in fused:
            chunk = chunks.get(uid)
            if chunk is None:
                continue
            tokens = chunk.token_count or 0
            if selected and used_tokens + tokens > token_budget:
                continue
            selected.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    vector_id=str(chunk.vector_id),
                    book_title=chunk.book.title,
                    subject=chunk.subject.code,
                    language=chunk.language,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    heading_path=chunk.heading_path,
                    content=chunk.content,
                    dense_sim=dense_sim.get(uid, 0.0),
                    score=rrf_score[uid],
                )
            )
            used_tokens += tokens
            if len(selected) >= top_k:
                break

        return selected, (max(dense_sim.values()) if dense_sim else 0.0)


def _rrf(rank_lists: list[list[uuid.UUID]], k: int = RRF_K) -> list[tuple[uuid.UUID, float]]:
    scores: dict[uuid.UUID, float] = defaultdict(float)
    for ranked in rank_lists:
        for rank, uid in enumerate(ranked):
            scores[uid] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
