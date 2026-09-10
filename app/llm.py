"""LLM client.

Talks OpenAI's chat-completions format to whatever `LLM_BASE_URL` points at —
by default the go-ai-gateway, which forwards to Anthropic and enforces
per-key budgets. Swapping to OpenAI or a local model is a config change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class ChatModel(Protocol):
    def complete(self, system: str, user: str) -> Completion: ...


class OpenAICompatibleModel:
    def __init__(self, base_url: str, api_key: str, model: str, max_tokens: int = 512) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> Completion:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0.1,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            model=resp.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
