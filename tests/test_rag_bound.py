"""Tests for RAG-as-noisy-ICL + spectral hallucination bounds."""
from __future__ import annotations

from recall.bound.rag_bound import (
    composite_hallucination_bound,
    estimate_cheeger,
    rag_noisy_icl_bound,
    spectral_hallucination_bound,
)


def test_rag_bound_returns_value():
    b = rag_noisy_icl_bound(
        avg_path_loss=0.3, n_retrieved=10,
        retrieval_distortion=0.2, embedding_dim=384,
    )
    assert b is not None
    assert 0 <= b <= 1


def test_rag_bound_grows_with_distortion():
    b_low = rag_noisy_icl_bound(0.2, 10, 0.05, 384)
    b_high = rag_noisy_icl_bound(0.2, 10, 0.5, 384)
    assert b_high > b_low


def test_rag_bound_loose_with_few_retrieved():
    b_few = rag_noisy_icl_bound(0.2, 1, 0.2, 384)
    b_many = rag_noisy_icl_bound(0.2, 50, 0.2, 384)
    # More retrieved = at least as tight (variance term decreases)
    assert b_few >= b_many - 1e-6


def test_spectral_bound_returns_value():
    b = spectral_hallucination_bound(edge_count=20, cheeger_lower_bound=0.3)
    assert b is not None
    assert 0 <= b <= 1


def test_spectral_bound_tightens_with_edges():
    b_few = spectral_hallucination_bound(edge_count=5, cheeger_lower_bound=0.3)
    b_many = spectral_hallucination_bound(edge_count=100, cheeger_lower_bound=0.3)
    assert b_many < b_few


def test_estimate_cheeger():
    c = estimate_cheeger(n_edges=10, n_nodes=5)
    assert 0 <= c <= 1


def test_composite_bound_returns_dict():
    res = composite_hallucination_bound(
        avg_path_loss=0.2, n_retrieved=5,
        retrieval_distortion=0.1,
        n_edges_in_subgraph=8, n_nodes_in_subgraph=4,
    )
    assert "rag_bound" in res
    assert "spectral_bound" in res
    assert "composite" in res
    assert res["composite"] is not None
    assert 0 <= res["composite"] <= 1
