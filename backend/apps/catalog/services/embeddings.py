"""OpenAI embedding client.

``text-embedding-3-large`` is an API model (3072-dim): it requires an API key
and network access and bills per token. The client is created lazily so that
``--dry-run`` paths never need a key or the ``openai`` package.
"""
from __future__ import annotations

from collections.abc import Sequence


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
