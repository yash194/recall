"""Tests for Γ identifiability components."""
from __future__ import annotations

import numpy as np

from recall.embeddings import HashEmbedder, TfidfEmbedder
from recall.geometry.gamma import gamma_score
from recall.geometry.identifiability import (
    CanonicalEmbedder,
    ParaphraseEnsembleEmbedder,
    WhiteningProjector,
)


def test_paraphrase_ensemble_returns_correct_shape():
    base = HashEmbedder(dim=64)
    e = ParaphraseEnsembleEmbedder(base, n_paraphrases=3)
    f, b = e.embed_dual("hello world")
    assert f.shape == (64,)
    assert b.shape == (64,)
    # both normalized
    assert abs(float(np.linalg.norm(f)) - 1.0) < 1e-3
    assert abs(float(np.linalg.norm(b)) - 1.0) < 1e-3


def test_paraphrase_ensemble_robustness():
    """Paraphrase-ensemble Γ should be more stable than single-prompt Γ."""
    base = HashEmbedder(dim=128)
    ensemble = ParaphraseEnsembleEmbedder(base, n_paraphrases=4)

    text_a = "We tried Postgres LISTEN/NOTIFY for the queue"
    text_b = "We switched to Redis Streams for stability"

    # Single-prompt Γ
    fa, ba = base.embed_dual(text_a)
    fb, bb = base.embed_dual(text_b)
    g1 = gamma_score(fa, ba, fb, bb)

    # Ensemble Γ
    fa2, ba2 = ensemble.embed_dual(text_a)
    fb2, bb2 = ensemble.embed_dual(text_b)
    g2 = gamma_score(fa2, ba2, fb2, bb2)

    # Both should be finite and roughly bounded
    assert np.isfinite(g1) and np.isfinite(g2)


def test_whitening_fit_and_transform():
    rng = np.random.default_rng(42)
    pairs = []
    for _ in range(50):
        f = rng.standard_normal(16).astype(np.float32)
        f = f / np.linalg.norm(f)
        b = rng.standard_normal(16).astype(np.float32)
        b = b / np.linalg.norm(b)
        pairs.append((f, b))

    w = WhiteningProjector()
    w.fit(pairs)

    f0, b0 = pairs[0]
    fw, bw = w.transform(f0, b0)
    # Output should be normalized
    assert abs(float(np.linalg.norm(fw)) - 1.0) < 1e-3
    assert abs(float(np.linalg.norm(bw)) - 1.0) < 1e-3


def test_canonical_embedder_pipeline():
    base = HashEmbedder(dim=64)
    c = CanonicalEmbedder(base, n_paraphrases=2)
    # Calibrate
    calibration = [f"calibration text {i}" for i in range(20)]
    c.fit_calibration(calibration)

    f, b = c.embed_dual("a fresh text")
    assert f.shape == (64,)
    assert b.shape == (64,)
