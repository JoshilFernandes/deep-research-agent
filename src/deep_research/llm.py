"""LLM provider abstraction.

Every agent talks to `LLMClient`, never to a specific vendor SDK. That
means: (1) swapping Groq for OpenAI/Anthropic later is a one-class change,
and (2) tests and the CI eval run against `MockLLM`, so the whole graph is
exercised deterministically with zero network calls and zero API cost.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class LLMResponse:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class LLMClient(ABC):
    model_name: str = "unknown"

    @abstractmethod
    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> LLMResponse:
        """Return a free-text completion."""

    def complete_json(self, system: str, user: str, *, temperature: float = 0.1) -> tuple[dict, LLMResponse]:
        """Ask for a JSON object and parse it, tolerating minor formatting slips
        (code fences, leading prose) that even well-prompted models produce.
        """
        json_system = (
            system
            + "\n\nRespond with a single valid JSON object and nothing else. "
            "No markdown code fences, no commentary before or after the JSON."
        )
        resp = self.complete(json_system, user, temperature=temperature)
        return _parse_json_loose(resp.text), resp


def _parse_json_loose(text: str) -> dict:
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences if the model added them anyway.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Last resort: grab the outermost {...} span.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Model did not return parseable JSON: {text[:200]!r}")


class GroqLLM(LLMClient):
    """Real provider, backed by Groq's OpenAI-compatible chat completions API."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        from groq import Groq  # imported lazily so MockLLM never needs the package configured

        self.model_name = model or os.getenv("DEEP_RESEARCH_MODEL", "llama-3.3-70b-versatile")
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
                "and put it in your .env file (see .env.example)."
            )
        self._client = Groq(api_key=key)

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        choice = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=choice,
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
        )


# Type of a scripted mock handler: (system, user) -> str
MockHandler = Callable[[str, str], str]


class MockLLM(LLMClient):
    """Deterministic, offline stand-in used by tests, CI, and `eval/run_eval.py --mock`.

    A `MockLLM` is built from an ordered list of handlers; each call pops the
    first handler whose `match` substring is found in the user prompt. This
    lets a single test script the planner, researcher, verifier and writer
    turns of one full run without any network access.
    """

    model_name = "mock-llm"

    def __init__(self, handlers: Optional[list[tuple[str, MockHandler]]] = None):
        self._handlers: list[tuple[str, MockHandler]] = handlers or []

    def add(self, match: str, handler: MockHandler) -> "MockLLM":
        self._handlers.append((match, handler))
        return self

    def complete(self, system: str, user: str, *, temperature: float = 0.2) -> LLMResponse:
        for match, handler in self._handlers:
            if match in user or match in system:
                text = handler(system, user)
                return LLMResponse(text=text, tokens_in=len(user) // 4, tokens_out=len(text) // 4)
        raise AssertionError(
            f"MockLLM received a prompt with no matching handler. "
            f"user[:200]={user[:200]!r}"
        )
