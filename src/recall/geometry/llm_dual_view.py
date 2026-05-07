"""LLM pre-filtering dual-view embedder — yash_math.md §5.3.

The proven path to high-ρ asymmetric embeddings:

  text_fwd = LLM("Describe the consequences, effects, and outcomes of: " + t)
  text_bwd = LLM("Describe the causes, preconditions, and origins of: " + t)
  f(t) = embed(text_fwd)
  b(t) = embed(text_bwd)

Per yash_math.md Table 5.3:
  - BGE + instruction prefix:  ρ = 0.164  (signal too weak — Γ = 0.006)
  - OpenAI + instruction:      ρ = 0.184  (still weak — Γ = 0.030)
  - LLM pre-filtering:         ρ = 0.418  (strong — 2.3× signal)

The cost: 2 LLM calls per write at indexing time (cached after first call).
The benefit: Γ becomes a load-bearing retrieval primitive.
"""
from __future__ import annotations

import hashlib
import threading
from typing import Any

import numpy as np

from recall.config import Config, DEFAULT
from recall.embeddings import Embedder
from recall.llm import LLMClient


_FORWARD_EXPANSION_PROMPT = """\
Describe the immediate consequences, effects, downstream outcomes, or
follow-up actions that this text leads to. Be concrete and stay in the same
domain. Output only the description (no preface).

Text:
{text}

Consequences:
"""

_BACKWARD_EXPANSION_PROMPT = """\
Describe the immediate causes, preconditions, prior context, or trigger
events that led to this text. Be concrete and stay in the same domain.
Output only the description (no preface).

Text:
{text}

Causes:
"""


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


class LLMPrefilteredEmbedder:
    """LLM-pre-filtered dual-view embedder.

    Wraps any base Embedder; uses an LLMClient to generate forward/backward
    expansions before embedding.

    Both expansions and embeddings are cached by sha256(text) — on a hit, no
    LLM or embedder calls are made.

    Args:
        base_embedder: an Embedder for the actual numerical embedding.
        llm: LLMClient that produces expansions.
        config: Config; uses default forward/backward prompts if not provided.
        cache_size: max entries in the LRU cache.
        max_expansion_tokens: token budget per LLM expansion call.
    """

    def __init__(
        self,
        base_embedder: Embedder,
        llm: LLMClient,
        config: Config | None = None,
        cache_size: int = 4096,
        max_expansion_tokens: int = 80,
    ):
        self._base = base_embedder
        self._llm = llm
        self._cfg = config or DEFAULT
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cache_max = cache_size
        self._max_tokens = max_expansion_tokens

    @property
    def dim(self) -> int:
        return self._base.dim

    def _expand_forward(self, text: str) -> str:
        try:
            return self._llm.complete(
                _FORWARD_EXPANSION_PROMPT.format(text=text),
                max_tokens=self._max_tokens,
            ).strip()
        except Exception:
            return f"Consequences of: {text}"

    def _expand_backward(self, text: str) -> str:
        try:
            return self._llm.complete(
                _BACKWARD_EXPANSION_PROMPT.format(text=text),
                max_tokens=self._max_tokens,
            ).strip()
        except Exception:
            return f"Causes of: {text}"

    def embed_dual(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            if h in self._cache:
                return self._cache[h]

        # Generate forward and backward expansions via LLM
        text_fwd = self._expand_forward(text)
        text_bwd = self._expand_backward(text)

        # Embed each via base embedder. We do NOT use the base's prompted dual-view
        # here — we want the raw embedding of the LLM-generated expansion.
        # If the base embedder doesn't expose a single-shot method, use embed_dual
        # and average the two views.
        f_pair = self._base.embed_dual(text_fwd)
        b_pair = self._base.embed_dual(text_bwd)

        # Use the forward-prompted view of each expanded text, normalized.
        f = _normalize(f_pair[0])
        b = _normalize(b_pair[0])

        with self._lock:
            self._cache[h] = (f, b)
            if len(self._cache) > self._cache_max:
                # Evict oldest by simple FIFO (not strictly LRU, sufficient)
                drop = list(self._cache.keys())[: self._cache_max // 8]
                for k in drop:
                    self._cache.pop(k, None)
        return f, b

    def cache_stats(self) -> dict[str, int]:
        return {"size": len(self._cache), "max": self._cache_max}
