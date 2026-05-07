"""Pluggable dual-view embedding backends.

The Embedder protocol returns (f, b) given a text — the forward and backward
views that produce Γ.

Reference impls (in order of fidelity):

    HashEmbedder   — deterministic SHA-derived; for tests and zero-dep installs.
    TfidfEmbedder  — sklearn TF-IDF + prompt-prefix mixing; real lexical semantics.
                     No torch / transformers required.
    BGEEmbedder    — sentence-transformers BAAI/bge-m3 (optional, requires torch).
    VoyageEmbedder — Voyage-4 asymmetric production (optional, API key required).
"""
from __future__ import annotations

import hashlib
import os
import threading
from typing import Protocol

import numpy as np

from recall.config import Config, DEFAULT


class Embedder(Protocol):
    """Embedding contract.

    `embed_dual(text)` returns the (f, b) prompted pair used for Γ-edge
    induction. `embed_symmetric(text)` returns the symmetric retrieval
    vector — by default the average of f and b, but neural embedders
    (e.g. BGE) override it to embed raw text without the forward/backward
    prompt prefixes (v0.6 — prompts dominated short-query embeddings).
    """

    @property
    def dim(self) -> int: ...

    def embed_dual(self, text: str) -> tuple[np.ndarray, np.ndarray]: ...

    def embed_symmetric(self, text: str) -> np.ndarray:
        """v0.6: default falls back to (f + b) / 2 for backward compat."""
        f, b = self.embed_dual(text)
        return (f + b) / 2


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


# -----------------------------------------------------------------------------
# HashEmbedder — deterministic, zero-dep
# -----------------------------------------------------------------------------


class HashEmbedder:
    """Deterministic stub embedder — same input always gives same (f, b).

    The forward / backward prompt prefixes are baked into the hash, so two
    prompted views of the same text produce different vectors but with
    controlled correlation. Good enough for tests, not for production retrieval.
    """

    def __init__(self, dim: int = 256, config: Config | None = None):
        self._dim = dim
        self._cfg = config or DEFAULT

    @property
    def dim(self) -> int:
        return self._dim

    def _hash_to_vec(self, prompt: str, text: str, seed: int = 0) -> np.ndarray:
        composite = f"{prompt}::{text}::{seed}"
        h = hashlib.sha256(composite.encode("utf-8")).digest()
        bytes_needed = self._dim * 4
        repeats = (bytes_needed + len(h) - 1) // len(h)
        buf = (h * repeats)[:bytes_needed]
        arr = np.frombuffer(buf, dtype=np.uint32).astype(np.float32)
        v = (arr / (2**32 - 1)) * 2.0 - 1.0
        return v[: self._dim]

    def embed_dual(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        text_part = self._hash_to_vec("text", text)
        forward_part = self._hash_to_vec(self._cfg.FORWARD_PROMPT, text, seed=1)
        backward_part = self._hash_to_vec(self._cfg.BACKWARD_PROMPT, text, seed=2)
        f = 0.7 * text_part + 0.3 * forward_part
        b = 0.7 * text_part + 0.3 * backward_part
        return _normalize(f), _normalize(b)

    def embed_symmetric(self, text: str) -> np.ndarray:
        """v0.6: fall back to (f + b) / 2 — HashEmbedder doesn't have a
        prompt-prefix issue worth specializing for."""
        f, b = self.embed_dual(text)
        return _normalize((f + b) / 2)


# -----------------------------------------------------------------------------
# TfidfEmbedder — real semantic embedding with no neural-net deps
# -----------------------------------------------------------------------------


class TfidfEmbedder:
    """sklearn TF-IDF + character-ngram backbone + prompt-prefix mixing.

    This produces real lexical-semantic embeddings without requiring torch /
    transformers. Forward and backward views differ via prompt prefixes (which
    the TF-IDF vectorizer treats as part of the document).

    Vocabulary is learned lazily from the texts seen at observe time; we keep a
    rolling buffer and refit on growth. A fixed projection matrix down to
    `dim` is used to keep vector size manageable.

    Recommended for: real benchmarks where you don't want the torch dep.
    """

    def __init__(
        self,
        dim: int = 384,
        max_features: int = 8192,
        config: Config | None = None,
        seed: int = 2026,
    ):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa
        except ImportError as e:
            raise ImportError("TfidfEmbedder requires scikit-learn.") from e
        self._dim = dim
        self._max_features = max_features
        self._cfg = config or DEFAULT
        self._lock = threading.Lock()
        self._fit_count = 0
        self._buffer: list[str] = []
        self._vectorizer = None  # type: ignore
        self._projection: np.ndarray | None = None
        self._seed = seed

    @property
    def dim(self) -> int:
        return self._dim

    def _refit(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        v = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=self._max_features,
            lowercase=True,
            sublinear_tf=True,
        )
        v.fit(self._buffer)
        self._vectorizer = v
        # Fixed random projection (Johnson-Lindenstrauss flavor) to dim
        rng = np.random.default_rng(self._seed)
        self._projection = rng.standard_normal((v.idf_.shape[0], self._dim)).astype(
            np.float32
        ) / np.sqrt(self._dim)
        self._fit_count += 1

    def _ensure_fit(self, text: str) -> None:
        if self._vectorizer is None or len(self._buffer) < 64 or self._fit_count == 0:
            with self._lock:
                self._buffer.append(text)
                if len(self._buffer) >= 32 and self._vectorizer is None:
                    self._refit()

    def _vectorize_one(self, text: str) -> np.ndarray:
        if self._vectorizer is None:
            # Cold start: fall back to a deterministic hash-based vector
            return _normalize(
                HashEmbedder(dim=self._dim, config=self._cfg)._hash_to_vec("tfidf-cold", text)
            )
        sparse = self._vectorizer.transform([text])
        # Project sparse → dense dim
        # sparse shape: (1, n_features), projection: (n_features, dim)
        dense = (sparse @ self._projection).astype(np.float32).ravel()
        return _normalize(dense)

    def embed_dual(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        forward_text = self._cfg.FORWARD_PROMPT + text
        backward_text = self._cfg.BACKWARD_PROMPT + text
        # Add observation to vocabulary buffer
        self._ensure_fit(forward_text)
        self._ensure_fit(backward_text)
        f = self._vectorize_one(forward_text)
        b = self._vectorize_one(backward_text)
        return f, b

    def embed_symmetric(self, text: str) -> np.ndarray:
        """v0.6: TF-IDF doesn't have the BGE prompt-prefix issue, so the
        symmetric vector is just the (f + b) / 2 average. Kept explicit so
        runtime `hasattr(embedder, "embed_symmetric")` reliably returns True."""
        f, b = self.embed_dual(text)
        return _normalize((f + b) / 2)


# -----------------------------------------------------------------------------
# BGEEmbedder — production sentence-transformers
# -----------------------------------------------------------------------------


class BGEEmbedder:
    """Real production embedder using BGE-M3 via sentence-transformers.

    Requires: pip install recall[embed-bge]
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", config: Config | None = None):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "BGEEmbedder requires `pip install recall[embed-bge]` "
                "(sentence-transformers + torch)."
            ) from e
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        self._cfg = config or DEFAULT

    @property
    def dim(self) -> int:
        return self._dim

    def embed_dual(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        forward_prompted = self._cfg.FORWARD_PROMPT + text
        backward_prompted = self._cfg.BACKWARD_PROMPT + text
        embs = self._model.encode(
            [forward_prompted, backward_prompted],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embs[0].astype(np.float32), embs[1].astype(np.float32)

    def embed_symmetric(self, text: str) -> np.ndarray:
        """v0.6: raw-text encoding for symmetric retrieval (no prompt prefix).

        The forward/backward prompts dominate the embedding for short text
        which hurt LongMemEval recall@5 in v0.5. Using the raw text here
        gives Recall the same retrieval quality as a vanilla cosine RAG
        baseline; the prompted f/b are still used by Γ-edge induction.
        """
        emb = self._model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True,
        )[0]
        return emb.astype(np.float32)

    def embed_batch(
        self, texts: list[str],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """v0.6: encode a batch of texts in one BGE call.

        Returns (forward_list, backward_list, symmetric_list). Used by the
        write pipeline to avoid one inference call per span.
        """
        if not texts:
            return [], [], []
        prompted_f = [self._cfg.FORWARD_PROMPT + t for t in texts]
        prompted_b = [self._cfg.BACKWARD_PROMPT + t for t in texts]
        # One inference call covers all three views: f for every text,
        # then b for every text, then raw for every text.
        all_inputs = prompted_f + prompted_b + list(texts)
        embs = self._model.encode(
            all_inputs, normalize_embeddings=True, convert_to_numpy=True,
        )
        n = len(texts)
        f_list = [embs[i].astype(np.float32) for i in range(n)]
        b_list = [embs[n + i].astype(np.float32) for i in range(n)]
        s_list = [embs[2 * n + i].astype(np.float32) for i in range(n)]
        return f_list, b_list, s_list


# -----------------------------------------------------------------------------
# Auto-select helper
# -----------------------------------------------------------------------------


def auto_embedder(prefer: str = "tfidf", **kwargs) -> Embedder:
    """Pick the best available embedder.

    Order: explicit `prefer`, then bge → tfidf → hash, falling back if deps
    aren't installed.
    """
    if prefer == "bge":
        try:
            return BGEEmbedder(**kwargs)
        except ImportError:
            pass  # fall through
    if prefer in ("tfidf", "bge"):
        try:
            return TfidfEmbedder(**kwargs)
        except ImportError:
            pass
    return HashEmbedder(**{k: v for k, v in kwargs.items() if k in ("dim",)})
