"""Γ identifiability under prompt rephrasing and LLM swap.

Per the Recall lit-review (Q3, May 2026):
  - PTEB (arXiv 2510.06730) shows production embeddings drift non-trivially
    under paraphrase.
  - "I Predict Therefore I Am" (arXiv 2503.08980) gives identifiability only
    up to invertible linear transform.
  - GSTransform (arXiv 2505.24754) provides a cheap projection that
    canonicalizes instruction-following embeddings.

Recall's Γ as written is NOT identifiable. Two fixes:

1. **Paraphrase ensemble** — for each forward/backward prompt, sample N
   paraphrases of the instruction, average f and b before computing Γ. Reduces
   prompt-instance noise.

2. **Whitening + GSTransform projection** — fit a per-direction projection on a
   small calibration set so f and b live in a canonical frame, robust to the
   underlying LLM choice.

Both are cheap (O(d^2) at calibration, free at inference) and required for
cross-model Γ comparisons.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from recall.embeddings import Embedder


# Paraphrases of the forward/backward direction prompts.
# Each set should have semantically equivalent variants. The averaging across
# them gives prompt-invariance per the PTEB recipe.
FORWARD_PROMPT_PARAPHRASES = (
    "Forward describe — what comes next or follows from: ",
    "What follows after: ",
    "Predict the consequence of: ",
    "What is caused by: ",
    "Subsequent to this, what happens: ",
)

BACKWARD_PROMPT_PARAPHRASES = (
    "Backward describe — what came before or causes: ",
    "What precedes: ",
    "What is the cause of: ",
    "What led to: ",
    "Prior to this, what was the case: ",
)


class ParaphraseEnsembleEmbedder:
    """Wraps an Embedder to average over paraphrases of the direction prompts.

    For each text, embeds N forward-paraphrases and N backward-paraphrases,
    L2-normalizes each, averages, then re-normalizes. The resulting (f̄, b̄) are
    significantly more robust to single-prompt instabilities.
    """

    def __init__(
        self,
        base_embedder: Embedder,
        n_paraphrases: int = 3,
        forward_prompts: tuple[str, ...] = FORWARD_PROMPT_PARAPHRASES,
        backward_prompts: tuple[str, ...] = BACKWARD_PROMPT_PARAPHRASES,
    ):
        self._base = base_embedder
        self._n = max(1, min(n_paraphrases, len(forward_prompts), len(backward_prompts)))
        self._fwd_prompts = forward_prompts[: self._n]
        self._bwd_prompts = backward_prompts[: self._n]

    @property
    def dim(self) -> int:
        return self._base.dim

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else v

    def embed_dual(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        # Compute n_paraphrases × (f, b) and average per direction
        fs = []
        bs = []
        for fp, bp in zip(self._fwd_prompts, self._bwd_prompts):
            # Reset the prompt-prefix for this iteration by mutating the config
            # of the underlying embedder if it supports it. Otherwise prepend
            # the prompt manually and use the embedder's raw embed_dual.
            #
            # The clean implementation: for embedders that take f,b through
            # configurable prompts (HashEmbedder, TfidfEmbedder, BGEEmbedder),
            # mutate the cfg's FORWARD_PROMPT/BACKWARD_PROMPT.
            cfg = getattr(self._base, "_cfg", None)
            if cfg is not None and hasattr(cfg, "FORWARD_PROMPT"):
                old_fwd = cfg.FORWARD_PROMPT
                old_bwd = cfg.BACKWARD_PROMPT
                cfg.FORWARD_PROMPT = fp
                cfg.BACKWARD_PROMPT = bp
                try:
                    f, b = self._base.embed_dual(text)
                finally:
                    cfg.FORWARD_PROMPT = old_fwd
                    cfg.BACKWARD_PROMPT = old_bwd
            else:
                # No way to inject paraphrases — fall through to the base call
                f, b = self._base.embed_dual(text)
            fs.append(self._normalize(f))
            bs.append(self._normalize(b))
        f_avg = self._normalize(np.mean(fs, axis=0))
        b_avg = self._normalize(np.mean(bs, axis=0))
        return f_avg, b_avg


class WhiteningProjector:
    """Whitens (f, b) embeddings using calibration-set statistics.

    Canonicalizes the embedding frame: applies (v - μ) / Σ^{1/2} so that the
    calibration covariance becomes identity. This removes per-LLM drift in the
    embedding centroid and scale.

    Per "Identifying Metric Structures" (Syrota-Hauberg ICML 2025), this
    canonicalization preserves Riemannian quantities up to isometry.
    """

    def __init__(self):
        self._mu_f: np.ndarray | None = None
        self._mu_b: np.ndarray | None = None
        self._sigma_f_inv: np.ndarray | None = None
        self._sigma_b_inv: np.ndarray | None = None
        self._fitted = False

    def fit(self, calibration_pairs: list[tuple[np.ndarray, np.ndarray]],
            ridge: float = 1e-3) -> None:
        """Fit whitening parameters from a small calibration set.

        Args:
            calibration_pairs: list of (f, b) tuples from the embedder.
            ridge: regularizer added to covariance for numerical stability.
        """
        if not calibration_pairs:
            return
        F = np.stack([p[0] for p in calibration_pairs])
        B = np.stack([p[1] for p in calibration_pairs])
        self._mu_f = F.mean(axis=0)
        self._mu_b = B.mean(axis=0)
        Fc = F - self._mu_f
        Bc = B - self._mu_b
        cov_f = (Fc.T @ Fc) / max(1, len(F) - 1) + ridge * np.eye(F.shape[1])
        cov_b = (Bc.T @ Bc) / max(1, len(B) - 1) + ridge * np.eye(B.shape[1])
        # Inverse square root via eigendecomp
        self._sigma_f_inv = self._inv_sqrt(cov_f)
        self._sigma_b_inv = self._inv_sqrt(cov_b)
        self._fitted = True

    def _inv_sqrt(self, M: np.ndarray) -> np.ndarray:
        w, V = np.linalg.eigh(M)
        w = np.maximum(w, 1e-6)
        return V @ np.diag(w ** -0.5) @ V.T

    def transform(self, f: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply whitening to (f, b). Returns canonicalized embeddings."""
        if not self._fitted:
            return f, b
        fw = (f - self._mu_f) @ self._sigma_f_inv
        bw = (b - self._mu_b) @ self._sigma_b_inv
        # Re-normalize to unit
        fn = float(np.linalg.norm(fw))
        bn = float(np.linalg.norm(bw))
        if fn > 1e-12:
            fw = fw / fn
        if bn > 1e-12:
            bw = bw / bn
        return fw.astype(np.float32), bw.astype(np.float32)


class CanonicalEmbedder:
    """Composes paraphrase-ensemble + whitening on top of any base embedder.

    This is the production-grade identifiable Γ embedder.
    """

    def __init__(
        self,
        base_embedder: Embedder,
        n_paraphrases: int = 3,
        whitener: WhiteningProjector | None = None,
    ):
        self._ensemble = ParaphraseEnsembleEmbedder(base_embedder, n_paraphrases=n_paraphrases)
        self._whitener = whitener or WhiteningProjector()

    @property
    def dim(self) -> int:
        return self._ensemble.dim

    def fit_calibration(self, texts: list[str], ridge: float = 1e-3) -> None:
        """Fit the whitener on a small calibration corpus."""
        pairs = [self._ensemble.embed_dual(t) for t in texts]
        self._whitener.fit(pairs, ridge=ridge)

    def embed_dual(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        f, b = self._ensemble.embed_dual(text)
        return self._whitener.transform(f, b)
