"""Baseline implementations for head-to-head comparison with Recall.

  - CosineRAG: vanilla vector RAG (BGE embeddings + cosine top-k)
  - BM25Index: classic BM25 keyword retrieval

These are the standard baselines published RAG papers compare against.
Both implement the same `add(text, scope)` / `search(query, k)` interface so
the benchmark code can swap them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class RetrievalHit:
    text: str
    score: float
    scope: dict
    meta: dict | None = None


class CosineRAG:
    """Vanilla vector RAG. Ingest text → embed via BGE → cosine top-k."""

    def __init__(self, embedder=None, dim: int = 384):
        if embedder is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            self._embed_fn = lambda t: self._model.encode([t], normalize_embeddings=True)[0]
        else:
            self._embed_fn = embedder
        self._dim = dim
        self._texts: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._scopes: list[dict] = []
        self._metas: list[dict] = []

    def add(self, text: str, scope: dict | None = None, meta: dict | None = None) -> int:
        emb = self._embed_fn(text)
        self._texts.append(text)
        self._embeddings.append(np.asarray(emb, dtype=np.float32))
        self._scopes.append(scope or {})
        self._metas.append(meta or {})
        return len(self._texts) - 1

    def search(self, query: str, k: int = 5, scope_filter: dict | None = None) -> list[RetrievalHit]:
        if not self._embeddings:
            return []
        q_emb = self._embed_fn(query)
        E = np.stack(self._embeddings)
        scores = E @ q_emb
        # apply scope filter if provided
        if scope_filter:
            mask = np.array([
                all(s.get(k) == v for k, v in scope_filter.items())
                for s in self._scopes
            ])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [
            RetrievalHit(
                text=self._texts[i],
                score=float(scores[i]),
                scope=self._scopes[i],
                meta=self._metas[i],
            )
            for i in order if scores[i] != -np.inf
        ]

    def size(self) -> int:
        return len(self._texts)

    def name(self) -> str:
        return "CosineRAG (BGE-small)"


class BM25Index:
    """Classic BM25 keyword retrieval (rank_bm25 implementation)."""

    def __init__(self):
        try:
            from rank_bm25 import BM25Okapi  # noqa: F401
        except ImportError as e:
            raise ImportError("BM25Index requires rank_bm25 (pip install rank_bm25)") from e
        self._BM25 = None
        self._corpus_tokens: list[list[str]] = []
        self._texts: list[str] = []
        self._scopes: list[dict] = []
        self._metas: list[dict] = []
        self._dirty = True

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        return [t for t in re.findall(r"[a-zA-Z0-9]+", (text or "").lower()) if len(t) > 1]

    def add(self, text: str, scope: dict | None = None, meta: dict | None = None) -> int:
        self._texts.append(text)
        self._corpus_tokens.append(self._tokenize(text))
        self._scopes.append(scope or {})
        self._metas.append(meta or {})
        self._dirty = True
        return len(self._texts) - 1

    def _refit(self):
        from rank_bm25 import BM25Okapi
        if not self._corpus_tokens:
            self._BM25 = None
            self._dirty = False
            return
        self._BM25 = BM25Okapi(self._corpus_tokens)
        self._dirty = False

    def search(self, query: str, k: int = 5, scope_filter: dict | None = None) -> list[RetrievalHit]:
        if self._dirty:
            self._refit()
        if self._BM25 is None:
            return []
        q_toks = self._tokenize(query)
        scores = self._BM25.get_scores(q_toks)
        if scope_filter:
            mask = np.array([
                all(s.get(k) == v for k, v in scope_filter.items())
                for s in self._scopes
            ])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [
            RetrievalHit(
                text=self._texts[i],
                score=float(scores[i]),
                scope=self._scopes[i],
                meta=self._metas[i],
            )
            for i in order if scores[i] != -np.inf
        ]

    def size(self) -> int:
        return len(self._texts)

    def name(self) -> str:
        return "BM25Okapi"


class RecallAdapter:
    """Wrap recall.Memory in the baseline interface for apples-to-apples.

    v0.5 parameters:
      mode: 'bulk' uses source='document' which triggers Recall's fast path
            (sentence-only split, no quality buffer scan, no Γ-edge induction).
            Best for ingesting large RAG corpora where retrieval is mostly
            cosine-style anyway.
            'conversation' (default) uses the full pipeline including
            edge induction. Required for path-mode retrieval to add value.
      retrieval_mode: passed to mem.recall() — 'auto' lets the router pick.
    """

    def __init__(
        self,
        mem,
        mode: str = "conversation",
        retrieval_mode: str = "auto",
    ):
        self.mem = mem
        self._source = "document" if mode == "bulk" else "conversation"
        self._retrieval_mode = retrieval_mode
        # v0.7: track whether to over-fetch and re-rank for multi-hop modes.
        # Multi-hop retrieval needs a larger candidate pool because the gold
        # answer node may not be cosine-similar to the question itself.
        self._over_fetch = retrieval_mode in ("multi_hop", "hybrid", "auto")

    def add(self, text: str, scope: dict | None = None, meta: dict | None = None) -> str:
        # Merge user-passed meta into scope so it round-trips through Recall's
        # storage. (Recall doesn't have a dedicated user-meta field; scope is
        # the canonical place for tags that should travel with the node.)
        merged_scope = dict(scope or {})
        if meta:
            for k, v in meta.items():
                if k not in merged_scope:
                    merged_scope[k] = v
        r = self.mem.observe(text, "", scope=merged_scope, source=self._source)
        return r.drawer_id or "(skipped)"

    def search(self, query: str, k: int = 5, scope_filter: dict | None = None) -> list[RetrievalHit]:
        result = self.mem.recall(
            query, scope=scope_filter or {}, mode=self._retrieval_mode, k=k
        )
        return [
            RetrievalHit(
                text=n.text,
                score=n.quality_score,
                scope=n.scope,
                # Replicate user-meta keys back into meta so benchmarks that
                # read h.meta.get(key) work without falling through to scope.
                meta={**(n.scope or {}), "role": n.role, "node_id": n.id},
            )
            for n in result.subgraph_nodes[:k]
        ]

    def size(self) -> int:
        return len(self.mem.storage.all_active_nodes())

    def name(self) -> str:
        return f"Recall (BGE + auto-router, {self._source})"
