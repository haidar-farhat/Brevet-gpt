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
    finish_reason: str | None = None  # "stop" | "length" (cut off) | ...
    via_reasoning: bool = False  # text came from the reasoning channel (content was empty)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_sec(self) -> float:
        return self.completion_tokens / self.latency if self.latency > 0 else 0.0


class LLMUnavailable(RuntimeError):
    """Raised when LM Studio is unreachable or has no model loaded."""


def _think_kwargs() -> dict:
    """Disable hybrid-reasoning 'thinking' at the chat-template level — the reliable
    switch for Qwen3 in LM Studio / vLLM. (We deliberately do NOT also inject a
    ``/no_think`` prompt token: combined with this flag it made Qwen3 emit EMPTY
    content.) No-op when LLM_NO_THINK is off or on a backend that ignores the kwarg."""
    if getattr(settings, "LLM_NO_THINK", False):
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}


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
        mt = settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens
        mt = max(mt, getattr(settings, "LLM_MIN_COMPLETION_TOKENS", 0))  # headroom for hidden reasoning
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature,
            "max_tokens": mt,
            **_think_kwargs(),
        }
        started = time.perf_counter()
        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMUnavailable(f"LM Studio request failed: {exc}") from exc
        latency = time.perf_counter() - started
        usage = response.usage
        choice = response.choices[0]
        # Fall back to the reasoning channel if a reasoning model left content empty.
        content = choice.message.content or ""
        text = content or (getattr(choice.message, "reasoning_content", "") or "")
        return LLMResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency=latency,
            finish_reason=getattr(choice, "finish_reason", None),
            via_reasoning=not content and bool(text),
        )

    async def chat_stream(self, messages: list[dict], on_token, *, on_reasoning=None,
                          temperature: float | None = None, max_tokens: int | None = None) -> LLMResult:
        """Stream completion tokens, invoking ``await on_token(delta)`` for each.
        A reasoning model's hidden chain-of-thought (the separate reasoning_content
        channel) is forwarded to ``await on_reasoning(delta)`` when given. Usage
        isn't reliably reported in stream mode, so counts are estimated with tiktoken."""
        from apps.catalog.services.chunking import count_tokens

        client = self._client_or_init()
        model = await self.model()
        mt = settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens
        mt = max(mt, getattr(settings, "LLM_MIN_COMPLETION_TOKENS", 0))  # headroom for hidden reasoning
        started = time.perf_counter()
        parts: list[str] = []
        reasoning: list[str] = []
        finish: str | None = None
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE if temperature is None else temperature,
                max_tokens=mt,
                stream=True,
                **_think_kwargs(),
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish = choice.finish_reason
                rdelta = getattr(choice.delta, "reasoning_content", "") or ""
                if rdelta:
                    reasoning.append(rdelta)
                    if on_reasoning is not None:
                        await on_reasoning(rdelta)
                delta = choice.delta.content or ""
                if delta:
                    parts.append(delta)
                    await on_token(delta)
        except Exception as exc:
            raise LLMUnavailable(f"LM Studio stream failed: {exc}") from exc
        text = "".join(parts)
        via_reasoning = False
        if not text and reasoning:  # reasoning model emitted ONLY into the think channel
            text = "".join(reasoning)  # surfaced via the result event (it holds the worked solution)
            via_reasoning = True
        prompt_text = " ".join(m.get("content", "") for m in messages)
        return LLMResult(
            text=text,
            prompt_tokens=count_tokens(prompt_text),
            completion_tokens=count_tokens(text),
            latency=time.perf_counter() - started,
            finish_reason=finish,
            via_reasoning=via_reasoning,
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
