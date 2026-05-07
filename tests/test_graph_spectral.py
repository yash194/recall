"""Tests for spectral graph theory primitives."""
from __future__ import annotations

import numpy as np
import pytest

from recall.graph.spectral import (
    cheeger_constant,
    graph_health,
    graph_laplacian,
    laplacian_eigenvalues,
    personalized_pagerank,
    spectral_gap,
)
from recall.types import Edge, EdgeType, Node


def _node(i: int) -> Node:
    return Node(
        id=f"n{i}", tenant="t", text=f"node {i}",
        f_embedding=np.ones(8, dtype=np.float32),
        b_embedding=np.ones(8, dtype=np.float32),
        quality_status="promoted",
    )


def _edge(i: int, src: str, dst: str, w: float, t: EdgeType = EdgeType.SUPPORTS) -> Edge:
    return Edge(
        id=f"e{i}", tenant="t", src_node_id=src, dst_node_id=dst,
        edge_type=t, weight=w, gamma_score=w, s_squared=1.0,
    )


def test_laplacian_zero_for_empty():
    L = graph_laplacian([], [])
    assert L.shape == (0, 0)


def test_laplacian_path_graph():
    """3-node path: connectivity gap λ_2 > 0."""
    nodes = [_node(i) for i in range(3)]
    edges = [_edge(0, "n0", "n1", 0.5), _edge(1, "n1", "n2", 0.5)]
    L = graph_laplacian(nodes, edges)
    assert L.shape == (3, 3)
    evals = laplacian_eigenvalues(nodes, edges, k=3)
    assert len(evals) >= 2
    assert evals[0] < 1e-6  # λ_1 = 0 (constant eigenvector)
    assert evals[1] > 0  # λ_2 > 0 (connected)


def test_spectral_gap_disconnected():
    """Two disconnected components: λ_2 = 0 (multiplicity 2 of zero)."""
    nodes = [_node(i) for i in range(4)]
    edges = [
        _edge(0, "n0", "n1", 0.5),
        _edge(1, "n2", "n3", 0.5),
    ]
    gap = spectral_gap(nodes, edges)
    # Two components → λ_1 = λ_2 = 0
    assert gap < 1e-3


def test_cheeger_returns_lower_upper():
    nodes = [_node(i) for i in range(5)]
    edges = [_edge(i, f"n{i}", f"n{(i+1) % 5}", 0.5) for i in range(5)]  # 5-cycle
    lower, upper = cheeger_constant(nodes, edges)
    assert lower >= 0
    assert upper >= lower
    # Cheeger inequality: lower ≤ upper


def test_personalized_pagerank_centers_on_seed():
    nodes = [_node(i) for i in range(4)]
    edges = [
        _edge(0, "n0", "n1", 0.5),
        _edge(1, "n0", "n2", 0.5),
        _edge(2, "n0", "n3", 0.5),
    ]
    ppr = personalized_pagerank(nodes, edges, seed_node_ids=["n0"])
    assert "n0" in ppr
    assert ppr["n0"] >= max(ppr["n1"], ppr["n2"], ppr["n3"])


def test_graph_health_returns_full_summary():
    nodes = [_node(i) for i in range(4)]
    edges = [_edge(i, f"n{i}", f"n{(i+1) % 4}", 0.5) for i in range(4)]
    health = graph_health(nodes, edges)
    assert health["n_nodes"] == 4
    assert health["n_edges"] == 4
    assert "spectral_gap_lambda2" in health
    assert "cheeger_lower_bound" in health
