"""Tests for Ollivier-Ricci curvature primitives."""
from __future__ import annotations

import numpy as np

from recall.graph.curvature import (
    compute_ollivier_ricci,
    curvature_pruning_signal,
    curvature_summary,
)
from recall.types import Edge, EdgeType, Node


def _node(i: int) -> Node:
    return Node(
        id=f"n{i}", tenant="t", text=f"node {i}",
        f_embedding=np.ones(8, dtype=np.float32),
        b_embedding=np.ones(8, dtype=np.float32),
        quality_status="promoted",
    )


def _edge(i: int, src: str, dst: str, w: float = 0.5, t: EdgeType = EdgeType.SUPPORTS) -> Edge:
    return Edge(
        id=f"e{i}", tenant="t", src_node_id=src, dst_node_id=dst,
        edge_type=t, weight=w, gamma_score=w, s_squared=1.0,
    )


def test_curvature_empty():
    assert compute_ollivier_ricci([], []) == {}
    assert curvature_pruning_signal([], []) == []


def test_curvature_returns_per_edge():
    """5-cycle: each edge gets a curvature value."""
    nodes = [_node(i) for i in range(5)]
    edges = [_edge(i, f"n{i}", f"n{(i+1) % 5}") for i in range(5)]
    curv = compute_ollivier_ricci(nodes, edges)
    assert len(curv) == 5
    for v in curv.values():
        assert -2.0 <= v <= 2.0  # bounded


def test_curvature_summary_returns_metrics():
    nodes = [_node(i) for i in range(4)]
    edges = [_edge(i, f"n{i}", f"n{(i+1) % 4}") for i in range(4)]
    summary = curvature_summary(nodes, edges)
    assert summary["n_edges"] == 4
    assert "mean_curvature" in summary
    assert "n_bottleneck_edges" in summary
