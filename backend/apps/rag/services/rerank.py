"""Cross-encoder reranking — the highest-ROI accuracy lever, with zero LLM cost.

The hybrid retriever fuses dense + lexical hits by RRF but never uses the actual
query↔chunk semantic match to order them. A cross-encoder scores each
(query, passage) pair directly; we blend that with the dense similarity and
reorder, so the most relevant chunks land in the (small) context window.

Mirrors LocalEmbedder: the model is loaded lazily on CPU and cached, and is only
ever called inside asyncio.to_thread.
"""
from __future__ import annotations

from collections.abc import Sequence

from apps.rag.services.retrieval import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._model = None

    @property
    def name(self) -> str:
        return self._model_name

    def _model_or_load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device="cpu", max_length=self._max_length)
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        model = self._model_or_load()
        scores = model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [float(s) for s in scores]


def build_reranker() -> CrossEncoderReranker | None:
    """Return the configured reranker, or None when disabled / not a cross-encoder
    (the 'llm' backend is handled by the agent's grading step instead)."""
    from django.conf import settings

    if not settings.RAG_RERANK or settings.RAG_RERANK_BACKEND.lower() != "cross_encoder":
        return None
    return CrossEncoderReranker(settings.RAG_RERANK_MODEL)


def rerank_chunks(reranker: CrossEncoderReranker, query: str, chunks: list[RetrievedChunk], *,
                  dense_weight: float) -> list[RetrievedChunk]:
    """Score chunks with the cross-encoder, min-max normalize, blend with the
    dense similarity, overwrite ``chunk.score``, and return sorted best-first."""
    if not chunks:
        return chunks
    raw = reranker.score(query, [c.content for c in chunks])
    lo, hi = min(raw), max(raw)
    span = hi - lo
    for chunk, s in zip(chunks, raw):
        norm = (s - lo) / span if span else 1.0
        chunk.score = (1.0 - dense_weight) * norm + dense_weight * chunk.dense_sim
    return sorted(chunks, key=lambda c: c.score, reverse=True)


def dense_rerank(chunks: list[RetrievedChunk], *, dense_weight: float) -> list[RetrievedChunk]:
    """No-model reranker: blend the (previously unused) dense similarity with the
    normalized RRF score and reorder. Zero cost, no second torch model — the
    robust default when a cross-encoder can't be co-loaded with the embedder."""
    if not chunks:
        return chunks
    rrf = [c.score for c in chunks]
    lo, hi = min(rrf), max(rrf)
    span = hi - lo
    for chunk in chunks:
        norm = (chunk.score - lo) / span if span else 1.0
        chunk.score = (1.0 - dense_weight) * norm + dense_weight * chunk.dense_sim
    return sorted(chunks, key=lambda c: c.score, reverse=True)
