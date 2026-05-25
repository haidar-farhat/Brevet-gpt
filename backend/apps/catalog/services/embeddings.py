"""Embedding backends behind a single ``.embed(texts) -> list[list[float]]`` API.

* ``LocalEmbedder`` — sentence-transformers on CPU (default; fully offline).
  Default model BAAI/bge-m3: multilingual (FR+EN), 1024-dim, 8k context.
* ``OpenAIEmbedder`` — optional API model (text-embedding-3-large, 3072-dim).

``build_embedder()`` picks one from settings. Heavy deps (torch / openai) are
imported lazily so ``--dry-run`` never needs them.
"""
from __future__ import annotations

from collections.abc import Sequence


class LocalEmbedder:
    """Local CPU embeddings via sentence-transformers. Model is loaded lazily
    and cached after the first call (and downloaded once to the HF cache)."""

    def __init__(self, model_name: str, batch_size: int = 32, normalize: bool = True) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize
        self._model = None

    @property
    def model(self) -> str:
        return self._model_name

    def _model_or_load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._model_or_load()
        vectors = model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str, batch_size: int = 100) -> None:
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is empty. Add it to backend/.env before embedding "
                "(or use --dry-run to validate chunking without calling the API)."
            )
        self._api_key = api_key
        self._model = model
        self._batch_size = batch_size
        self._client = None

    @property
    def model(self) -> str:
        return self._model

    def _client_or_init(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        client = self._client_or_init()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = client.embeddings.create(model=self._model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors


def build_embedder():
    """Return the embedder selected by settings.EMBEDDING_BACKEND ('local'|'openai').

    Raises ValueError on misconfiguration so callers can surface a clean message.
    """
    from django.conf import settings

    backend = settings.EMBEDDING_BACKEND.lower()
    if backend == "local":
        return LocalEmbedder(settings.LOCAL_EMBED_MODEL)
    if backend == "openai":
        return OpenAIEmbedder(settings.OPENAI_API_KEY, settings.OPENAI_EMBED_MODEL)
    raise ValueError(f"Unknown EMBEDDING_BACKEND '{backend}'; expected 'local' or 'openai'.")
