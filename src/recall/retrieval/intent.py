"""Query intent classification — symmetric / directional / hybrid.

Rule-based with optional LLM fallback. Most queries are unambiguous via
keyword analysis (why / what caused / what happens after → directional).
"""
from __future__ import annotations

from recall.llm import LLMClient


_DIRECTIONAL_HINTS = (
    "why",
    "what caused",
    "what causes",
    "because",
    "reason",
    "what led to",
    "what comes after",
    "what happens next",
    "consequence",
    "due to",
    "resulted",
    "led to",
    "follow",
    "succeed",
)

_SYMMETRIC_HINTS = (
    "what is",
    "what's",
    "describe",
    "explain",
    "tell me about",
    "find",
    "locate",
)


def classify_intent(query: str, llm: LLMClient | None = None) -> str:
    """Returns 'directional' | 'symmetric' | 'hybrid'."""
    q_lc = (query or "").lower()
    has_directional = any(h in q_lc for h in _DIRECTIONAL_HINTS)
    has_symmetric = any(h in q_lc for h in _SYMMETRIC_HINTS)

    if has_directional and has_symmetric:
        return "hybrid"
    if has_directional:
        return "directional"
    if has_symmetric:
        return "symmetric"

    # If unclassified by rules and LLM is available, ask
    if llm is not None:
        try:
            return llm.classify(
                f"Is this query directional (asking about cause/effect/sequence) or symmetric "
                f"(asking about properties/descriptions) or hybrid?\n\nQuery: {query}",
                options=["directional", "symmetric", "hybrid"],
            )
        except Exception:
            pass

    # Default: hybrid — try both retrieval modes and fuse
    return "hybrid"
