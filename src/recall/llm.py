"""LLMClient protocol and reference implementations.

Provides a stable interface for the small LLM operations Recall performs:
  - complete(prompt) — generation (used by bounded_generate)
  - classify(prompt, options) — multiple-choice (used by intent / edge-type classification)
  - split_into_thoughts(text) — utterance decomposition

Reference impls:
  MockLLMClient — for tests; deterministic, zero-dep.
  OpenAIClient  — production; requires `openai` package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ThoughtSpan:
    text: str
    role: str  # 'fact' | 'attempt' | 'decision' | 'pivot' | 'outcome' | 'correction'


class LLMClient(Protocol):
    def complete(self, prompt: str, max_tokens: int = 256) -> str: ...

    def classify(self, prompt: str, options: list[str]) -> str: ...

    def split_into_thoughts(self, text: str) -> list[ThoughtSpan]: ...


class MockLLMClient:
    """Deterministic stub LLM. Splits on punctuation; classifies by lexical hits."""

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        # Strip the BOUNDED_PROMPT scaffolding and echo the context tail.
        # In tests we're checking the API surface, not generation quality.
        if "Context:" in prompt:
            tail = prompt.rsplit("Context:", 1)[-1].strip().splitlines()
            return " ".join(line for line in tail if line.strip())[:max_tokens]
        return prompt[:max_tokens]

    def classify(self, prompt: str, options: list[str]) -> str:
        # Pick the option whose name appears most often (case-insensitive) in the prompt.
        prompt_lc = prompt.lower()
        scored = sorted(options, key=lambda o: -prompt_lc.count(o.lower()))
        return scored[0]

    def split_into_thoughts(self, text: str) -> list[ThoughtSpan]:
        # Sentence-level fallback splitter. Not great, sufficient for tests.
        # Heuristic role assignment by keyword.
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        spans = []
        for s in sentences:
            s = s.strip()
            if len(s) < 5:
                continue
            role = "fact"
            ls = s.lower()
            if any(k in ls for k in ("decided", "switched to", "we will use")):
                role = "decision"
            elif any(k in ls for k in ("tried", "attempted")):
                role = "attempt"
            elif any(k in ls for k in ("but", "however", "instead")):
                role = "pivot"
            elif any(k in ls for k in ("works", "stable", "succeeded", "failed")):
                role = "outcome"
            elif any(k in ls for k in ("actually", "correction", "wrong")):
                role = "correction"
            spans.append(ThoughtSpan(text=s, role=role))
        return spans


class OpenAIClient:
    """Real LLM client backed by OpenAI / OpenRouter / Anthropic via OpenAI-compatible API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None,
                 base_url: str | None = None):
        try:
            import openai  # type: ignore
        except ImportError as e:
            raise ImportError("OpenAIClient requires `pip install recall[llm-openai]`.") from e
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    def classify(self, prompt: str, options: list[str]) -> str:
        full_prompt = f"{prompt}\n\nRespond with EXACTLY one of: {', '.join(options)}."
        out = self.complete(full_prompt, max_tokens=20).strip().lower()
        for opt in options:
            if opt.lower() in out:
                return opt
        return options[0]  # default fallback

    def split_into_thoughts(self, text: str) -> list[ThoughtSpan]:
        prompt = (
            "Split the text below into atomic thoughts. For each thought, output a JSON "
            "object with keys 'text' (string) and 'role' (one of: fact, attempt, decision, "
            "pivot, outcome, correction).\n\n"
            f"Text: {text}\n\n"
            "Output JSON list only."
        )
        out = self.complete(prompt, max_tokens=512)
        try:
            import json

            data = json.loads(out)
            return [ThoughtSpan(text=d["text"], role=d.get("role", "fact")) for d in data]
        except Exception:
            # Fall back to mock splitter on parse failure.
            return MockLLMClient().split_into_thoughts(text)
