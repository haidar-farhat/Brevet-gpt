"""Async client for the LM Studio OpenAI-compatible server.

Captures token usage + latency on every call so the pipeline and the evaluator
can report throughput. The model id is auto-detected from the loaded model when
LMSTUDIO_MODEL is left blank.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from django.conf import settings


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_sec(self) -> float:
        return self.completion_tokens / self.latency if self.latency > 0 else 0.0


class LLMUnavailable(RuntimeError):
    """Raised when LM Studio is unreachable or has no model loaded."""


class LMStudioClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float | None = None) -> None:
        self.base_url = base_url or settings.LMSTUDIO_BASE_URL
        self.api_key = api_key or settings.LMSTUDIO_API_KEY
        self._model = model or settings.LMSTUDIO_MODEL
        self.timeout = timeout or settings.LLM_TIMEOUT
        self._client = None

    def _client_or_init(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        return self._client

    async def model(self) -> str:
        if self._model:
            return self._model
        try:
            listing = await self._client_or_init().models.list()
        except Exception as exc:  # connection refused, etc.
            raise LLMUnavailable(
                f"LM Studio not reachable at {self.base_url}. Start it, load a model, "
                f"and enable the local server. ({exc})"
            ) from exc
        if not listing.data:
            raise LLMUnavailable("LM Studio is running but no model is loaded.")
        self._model = listing.data[0].id
        return self._model

    async def chat(self, messages: list[dict], *, temperature: float | None = None,
                   max_tokens: int | None = None, json_mode: bool = False) -> LLMResult:
        # NB: json_mode is intentionally NOT mapped to response_format. LM Studio
        # rejects OpenAI's {"type": "json_object"} (it wants json_schema/text), so
        # we rely on the "respond JSON only" instruction + the tolerant parser in
        # chat_json. This keeps the client portable across local backends.
        client = self._client_or_init()
        model = await self.model()
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature,
            "max_tokens": settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
        }
        started = time.perf_counter()
        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMUnavailable(f"LM Studio request failed: {exc}") from exc
        latency = time.perf_counter() - started
        usage = response.usage
        return LLMResult(
            text=response.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency=latency,
        )

    async def chat_json(self, messages: list[dict], **kwargs) -> tuple[dict, LLMResult]:
        """Chat expecting a JSON object back; tolerant of code-fenced output."""
        result = await self.chat(messages, json_mode=True, **kwargs)
        return _parse_json(result.text), result

    async def health(self) -> dict:
        model = await self.model()
        return {"base_url": self.base_url, "model": model}


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}
