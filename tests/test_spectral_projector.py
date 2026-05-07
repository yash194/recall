"""Tests for spectral causal amplification (yash_math.md §6)."""
from __future__ import annotations

import numpy as np

from recall.geometry.gamma import gamma_score
from recall.geometry.spectral import SpectralProjector, fit_from_embedder
from recall.embeddings import HashEmbedder


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def test_spectral_projector_fit_returns_projector():
    rng = np.random.default_rng(42)
    pairs = []
    for _ in range(100):
        f = _norm(rng.standard_normal(32).astype(np.float32))
        b = _norm(rng.standard_normal(32).astype(np.float32))
        pairs.append((f, b))
    proj = SpectralProjector.fit(pairs, threshold=0.5)
    assert proj.P.shape[0] == 32
    assert proj.P.shape[1] >= 4  # min_dim
    assert len(proj.eigenvalues) == proj.P.shape[1]


def test_spectral_projector_preserves_dim_invariants():
    rng = np.random.default_rng(7)
    pairs = [(rng.standard_normal(16).astype(np.float32),
              rng.standard_normal(16).astype(np.float32)) for _ in range(50)]
    proj = SpectralProjector.fit(pairs)
    f, b = pairs[0]
    spec = proj.gamma_spec(f, b, f, b)
    assert isinstance(spec, float)


def test_fit_from_embedder():
    embedder = HashEmbedder(dim=64)
    texts = [f"sample text number {i}" for i in range(20)]
    proj = fit_from_embedder(embedder, texts)
    assert proj.P.shape[0] == 64


def test_spectral_gamma_diagonal_is_negative():
    """Γ_spec(t→t) reduces to -c̃·c̃ ≤ 0 by construction."""
    rng = np.random.default_rng(99)
    pairs = [(rng.standard_normal(16).astype(np.float32),
              rng.standard_normal(16).astype(np.float32)) for _ in range(30)]
    proj = SpectralProjector.fit(pairs)
    f, b = pairs[0]
    self_gamma = proj.gamma_spec(f, b, f, b)
    # In projected space, c̃·c̃ ≥ 0 → Γ_spec(t,t) ≤ 0
    assert self_gamma <= 1e-3
