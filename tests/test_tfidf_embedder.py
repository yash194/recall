"""Tests for the real TF-IDF embedder."""
from __future__ import annotations

import numpy as np
import pytest

from recall.embeddings import TfidfEmbedder
from recall.geometry.gamma import gamma_score


def test_tfidf_embedder_returns_correct_dim():
    e = TfidfEmbedder(dim=128)
    f, b = e.embed_dual("hello world this is a test")
    assert f.shape == (128,)
    assert b.shape == (128,)


def test_tfidf_embedder_normalized():
    e = TfidfEmbedder(dim=128)
    # Need to feed a few texts to fit the vectorizer
    for _ in range(40):
        e.embed_dual("this is fixed seed text " + str(_))
    f, b = e.embed_dual("a fresh new sentence about systems")
    assert abs(np.linalg.norm(f) - 1.0) < 1e-3
    assert abs(np.linalg.norm(b) - 1.0) < 1e-3


def test_tfidf_dual_views_differ():
    e = TfidfEmbedder(dim=128)
    for i in range(40):
        e.embed_dual(f"warm up text number {i}")
    f, b = e.embed_dual("real test content")
    # f and b should differ (different prompt prefixes)
    assert not np.allclose(f, b)


def test_tfidf_similar_texts_similar_embeddings():
    e = TfidfEmbedder(dim=128)
    # Warm up
    for i in range(40):
        e.embed_dual(f"warm up text number {i}")
    f1, _ = e.embed_dual("Redis Streams as message queue")
    f2, _ = e.embed_dual("Redis Streams used as the queue")
    f3, _ = e.embed_dual("The cat sat on the mat")
    cos12 = float(np.dot(f1, f2))
    cos13 = float(np.dot(f1, f3))
    assert cos12 > cos13


def test_embed_symmetric_falls_back_to_normalized_average():
    """v0.6: TfidfEmbedder.embed_symmetric returns the L2-normalized
    (f + b) / 2 — the protocol default for non-neural embedders that
    don't have a prompt-prefix issue worth specializing for."""
    e = TfidfEmbedder(dim=128)
    for i in range(40):
        e.embed_dual(f"warm up {i}")
    f, b = e.embed_dual("a real test sentence")
    s = e.embed_symmetric("a real test sentence")
    avg = (f + b) / 2
    norm = float(np.linalg.norm(avg))
    expected = avg / norm if norm > 1e-12 else avg
    np.testing.assert_allclose(s, expected, atol=1e-5)
    # Also verify s is unit-norm
    assert abs(float(np.linalg.norm(s)) - 1.0) < 1e-4
