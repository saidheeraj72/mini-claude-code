"""Streaming Ollama chat client."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator

import ollama

from mcc.config import Config


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    call_id: str = ""


@dataclass
class Chunk:
    """One streamed delta: text, thinking, or the final assembled message."""

    text: str = ""
    thinking: str = ""
    done: bool = False
    tool_calls: list[ToolCall] | None = None
    eval_count: int = 0
    prompt_eval_count: int = 0


class OllamaClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = ollama.Client(host=cfg.ollama_host)
        # True until a model tells us it has no thinking mode, then None.
        self._thinks: bool | None = True

    # ------------------------------------------------------------------ chat
    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[Chunk]:
        """Yield Chunks as the model generates. Final chunk has done=True."""
        options = {
            "num_ctx": self.cfg.num_ctx,
            "temperature": self.cfg.temperature,
        }
        # Thinking is always *requested*, never suppressed. On qwen3, think=False
        # does not stop the model reasoning -- it just stops Ollama separating it,
        # so the reasoning lands in `content` and would be persisted as the
        # assistant's answer. Requesting it keeps `content` clean; whether the
        # user sees it is a display concern (cfg.think).
        kwargs: dict[str, Any] = dict(
            model=self.cfg.model,
            messages=messages,
            tools=tools or None,
            stream=True,
            options=options,
        )
        try:
            stream = self._client.chat(think=self._thinks, **kwargs)
        except ollama.ResponseError as e:
            if self._thinks is not True or "think" not in str(e).lower():
                raise
            # Model has no thinking mode; remember and retry without it.
            self._thinks = None
            stream = self._client.chat(**kwargs)

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        eval_count = prompt_eval = 0

        for part in stream:
            msg = part.get("message") or {}
            thinking = msg.get("thinking") or ""
            content = msg.get("content") or ""
            if content:
                text_parts.append(content)
            if thinking or content:
                yield Chunk(text=content, thinking=thinking)

            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    args = _loads_or_empty(args)
                calls.append(ToolCall(name=fn.get("name", ""), args=dict(args)))

            if part.get("done"):
                eval_count = part.get("eval_count") or 0
                prompt_eval = part.get("prompt_eval_count") or 0

        full_text = "".join(text_parts)

        # Small models frequently ignore the tools API and emit a fenced JSON
        # block instead. Recover those rather than treating them as prose.
        if not calls:
            recovered, full_text = extract_inline_calls(full_text)
            calls = recovered

        yield Chunk(
            done=True,
            text=full_text,
            tool_calls=calls,
            eval_count=eval_count,
            prompt_eval_count=prompt_eval,
        )

    # ------------------------------------------------------------- embedding
    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embed(model=self.cfg.embed_model, input=texts)
        return [list(e) for e in resp["embeddings"]]

    # ------------------------------------------------------------ diagnostic
    def available_models(self) -> list[str]:
        return [m.get("model", "") for m in self._client.list().get("models", [])]


# ---------------------------------------------------------------- fallbacks
import json

_FENCE = re.compile(r"```(?:json|tool_call|tool)?\s*(\{.*?\})\s*```", re.DOTALL)


def _loads_or_empty(raw: str) -> dict[str, Any]:
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract_inline_calls(text: str) -> tuple[list[ToolCall], str]:
    """Pull tool calls out of fenced JSON blocks a model emitted as prose.

    Recognises {"name": ..., "arguments": {...}} and the {"tool": ...,
    "args": {...}} variant. Returns the calls plus the text with those
    blocks stripped out.
    """
    calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []

    for match in _FENCE.finditer(text):
        payload = _loads_or_empty(match.group(1))
        name = payload.get("name") or payload.get("tool") or payload.get("tool_name")
        if not name:
            continue
        args = payload.get("arguments")
        if args is None:
            args = payload.get("args")
        if args is None:
            args = {k: v for k, v in payload.items() if k not in
                    ("name", "tool", "tool_name")}
        if not isinstance(args, dict):
            continue
        calls.append(ToolCall(name=str(name), args=args))
        spans.append(match.span())

    for start, end in reversed(spans):
        text = text[:start] + text[end:]

    return calls, text.strip()
