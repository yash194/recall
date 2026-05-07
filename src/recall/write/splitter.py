"""Node-split classifier — splits an utterance into 0..N thought spans.

Strategy:
  1. Try LLM split (if LLMClient provided).
  2. Fall back to sentence-level segmentation.
  3. Merge adjacent short sentences into chunks of at least `min_chunk_tokens`
     so the prompt-prefix doesn't dominate the embedding (v0.5 fix).
  4. Filter out chunks too short to be meaningful.

v0.5 background: with prompt-prefixed dual embedding (FORWARD_PROMPT +
BACKWARD_PROMPT prefixes ~50 chars each), short sentences (5-30 chars)
end up with embeddings dominated by the prefix and cluster together
regardless of content. Merging into ~50-100 token chunks restores real
semantic discrimination.
"""
from __future__ import annotations

import re

from recall.llm import LLMClient, ThoughtSpan


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_DEFAULT_MIN_CHUNK_TOKENS = 80
_DEFAULT_MAX_CHUNK_TOKENS = 250


def _coalesce(
    spans: list[ThoughtSpan],
    min_tokens: int,
    max_tokens: int,
) -> list[ThoughtSpan]:
    """Merge adjacent ThoughtSpans into chunks of [min_tokens, max_tokens] words.

    Single sentence longer than `max_tokens` is kept as-is (we don't truncate).
    """
    if not spans:
        return spans
    out: list[ThoughtSpan] = []
    buf_text: list[str] = []
    buf_role: str | None = None
    buf_tokens = 0
    for sp in spans:
        n = len(sp.text.split())
        # If adding this would exceed max and we have enough already, flush
        if buf_tokens >= min_tokens and buf_tokens + n > max_tokens:
            out.append(ThoughtSpan(text=" ".join(buf_text), role=buf_role or "fact"))
            buf_text = []
            buf_tokens = 0
            buf_role = None
        buf_text.append(sp.text)
        buf_tokens += n
        if buf_role is None:
            buf_role = sp.role
    if buf_text:
        out.append(ThoughtSpan(text=" ".join(buf_text), role=buf_role or "fact"))
    # Final pass: drop chunks too short
    return [s for s in out if len(s.text.split()) >= 3]


def sentence_split(
    text: str,
    min_tokens: int = 3,
    min_chunk_tokens: int = _DEFAULT_MIN_CHUNK_TOKENS,
    max_chunk_tokens: int = _DEFAULT_MAX_CHUNK_TOKENS,
) -> list[ThoughtSpan]:
    """Pure-Python sentence-level splitter — fallback when no LLM is available.

    v0.5: adjacent short sentences are coalesced so each output span is at
    least `min_chunk_tokens` words. Pass `min_chunk_tokens=0` to keep the
    legacy per-sentence behavior.
    """
    raw: list[ThoughtSpan] = []
    for s in _SENTENCE_BOUNDARY.split(text.strip()):
        s = s.strip()
        if len(s.split()) < min_tokens:
            continue
        raw.append(ThoughtSpan(text=s, role="fact"))
    if min_chunk_tokens <= 0:
        return raw
    return _coalesce(raw, min_chunk_tokens, max_chunk_tokens)


def split_into_thoughts(
    text: str,
    llm: LLMClient | None = None,
    min_chunk_tokens: int = _DEFAULT_MIN_CHUNK_TOKENS,
    max_chunk_tokens: int = _DEFAULT_MAX_CHUNK_TOKENS,
) -> list[ThoughtSpan]:
    """Top-level entry point. LLM if provided, else sentence-level fallback."""
    if llm is not None:
        try:
            spans = llm.split_into_thoughts(text)
            if spans:
                if min_chunk_tokens > 0:
                    spans = _coalesce(spans, min_chunk_tokens, max_chunk_tokens)
                return spans
        except Exception:
            pass  # fall through to deterministic splitter
    return sentence_split(
        text,
        min_chunk_tokens=min_chunk_tokens,
        max_chunk_tokens=max_chunk_tokens,
    )
