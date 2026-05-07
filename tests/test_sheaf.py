"""Tests for cellular sheaf inconsistency detection."""
from __future__ import annotations

import numpy as np

from recall.graph.sheaf import (
    harmonic_dimension,
    inconsistency_score,
    sheaf_laplacian,
    signed_incidence,
)
from recall.types import Edge, EdgeType, Node


def _node(i: int) -> Node:
    return Node(
        id=f"n{i}", tenant="t", text=f"node {i}",
        f_embedding=np.ones(8, dtype=np.float32),
        b_embedding=np.ones(8, dtype=np.float32),
        quality_status="promoted",
    )


def _edge(i, src, dst, t=EdgeType.SUPPORTS):
    return Edge(
        id=f"e{i}", tenant="t", src_node_id=src, dst_node_id=dst,
        edge_type=t, weight=0.5, gamma_score=0.5, s_squared=1.0,
    )


def test_signed_incidence_shape():
    nodes = [_node(i) for i in range(3)]
    edges = [_edge(0, "n0", "n1"), _edge(1, "n1", "n2")]
    inc, idx = signed_incidence(nodes, edges)
    assert inc.shape == (2, 3)


def test_sheaf_laplacian_consistent_supports_only():
    """3-node chain with all supports edges → 1 globally consistent section."""
    nodes = [_node(i) for i in range(3)]
    edges = [_edge(0, "n0", "n1"), _edge(1, "n1", "n2")]
    h0 = harmonic_dimension(nodes, edges)
    # Connected graph with all consistent signs → H¹ = 0, H⁰ = 1
    assert h0 >= 1


def test_inconsistency_with_contradicts_cycle():
    """3-cycle with one CONTRADICTS edge — frustrated triangle."""
    nodes = [_node(i) for i in range(3)]
    edges = [
        _edge(0, "n0", "n1", EdgeType.SUPPORTS),
        _edge(1, "n1", "n2", EdgeType.SUPPORTS),
        _edge(2, "n2", "n0", EdgeType.CONTRADICTS),
    ]
    score = inconsistency_score(nodes, edges)
    # Frustrated cycle → not globally consistent
    # (specifically: H¹ >= 1 since the signs around the loop don't compose to identity)
    assert score["frustration_score"] > 0.0
    assert "is_globally_consistent" in score


def test_inconsistency_score_returns_dict():
    nodes = [_node(i) for i in range(3)]
    edges = [_edge(0, "n0", "n1"), _edge(1, "n1", "n2")]
    score = inconsistency_score(nodes, edges)
    assert "n_consistent_sections" in score
    assert "smallest_eigenvalue" in score
    assert "is_globally_consistent" in score
    assert "frustration_score" in score


def test_empty_graph():
    score = inconsistency_score([], [])
    assert score["n_consistent_sections"] == 0
