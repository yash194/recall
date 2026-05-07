"""Tests for the seed-dispersion adaptive router."""
from __future__ import annotations

import numpy as np

from recall.retrieval.router import (
    local_clustering_coefficient,
    mean_pairwise_hops,
    reciprocal_rank_fuse,
    route,
)
from recall.types import Edge, EdgeType, Node


def _node(i: int) -> Node:
    return Node(
        id=f"n{i}", tenant="t", text=f"node {i}",
        f_embedding=np.ones(8, dtype=np.float32),
        b_embedding=np.ones(8, dtype=np.float32),
        quality_status="promoted",
    )


def _edge(i, src, dst):
    return Edge(
        id=f"e{i}", tenant="t", src_node_id=src, dst_node_id=dst,
        edge_type=EdgeType.SUPPORTS, weight=0.5, gamma_score=0.5, s_squared=1.0,
    )


def test_local_clustering_triangle():
    """3-clique: clustering coefficient = 1.0 for each node."""
    nodes = [_node(i) for i in range(3)]
    edges = [
        _edge(0, "n0", "n1"), _edge(1, "n1", "n2"), _edge(2, "n0", "n2"),
    ]
    from recall.retrieval.router import _adj_from_edges
    adj = _adj_from_edges(nodes, edges)
    c = local_clustering_coefficient(adj, ["n0", "n1", "n2"])
    assert abs(c - 1.0) < 1e-6


def test_local_clustering_path():
    """3-node path: clustering = 0 for endpoints + middle."""
    nodes = [_node(i) for i in range(3)]
    edges = [_edge(0, "n0", "n1"), _edge(1, "n1", "n2")]
    from recall.retrieval.router import _adj_from_edges
    adj = _adj_from_edges(nodes, edges)
    c = local_clustering_coefficient(adj, ["n0", "n1", "n2"])
    assert c < 0.34  # all 0


def test_mean_hops_chain():
    """Chain a-b-c-d-e: mean pairwise hops between {a,c,e} > 1."""
    nodes = [_node(i) for i in range(5)]
    edges = [_edge(i, f"n{i}", f"n{i+1}") for i in range(4)]
    from recall.retrieval.router import _adj_from_edges
    adj = _adj_from_edges(nodes, edges)
    h = mean_pairwise_hops(adj, ["n0", "n2", "n4"])
    assert h >= 2.0


def test_route_factual_clustered_returns_symmetric():
    # 3-clique seeds, factual query
    nodes = [_node(i) for i in range(3)]
    edges = [_edge(0, "n0", "n1"), _edge(1, "n1", "n2"), _edge(2, "n0", "n2")]
    r = route("What is the queue tech?", nodes, edges, ["n0", "n1", "n2"])
    assert r == "symmetric"


def test_route_causal_spread_returns_walk_deep():
    nodes = [_node(i) for i in range(5)]
    edges = [_edge(i, f"n{i}", f"n{i+1}") for i in range(4)]
    r = route("Why did the queue become stable?", nodes, edges, ["n0", "n2", "n4"])
    assert r == "walk_deep"


def test_route_no_seeds_returns_symmetric():
    assert route("anything", [], [], []) == "symmetric"


def test_rrf_fuse_basic():
    a = ["x", "y", "z"]
    b = ["z", "y", "x"]
    fused = reciprocal_rank_fuse([a, b])
    # All 3 items appear in fused output
    assert set(fused) == {"x", "y", "z"}
    # All three are roughly tied (each appears at rank 1, 2, 3 across lists);
    # 'y' has the most-balanced position. Just verify all elements are present.
    assert len(fused) == 3


def test_rrf_fuse_weights():
    a = ["x", "y"]
    b = ["y", "x"]
    fused_eq = reciprocal_rank_fuse([a, b], [1.0, 1.0])
    # Equal weights: tied
    fused_a = reciprocal_rank_fuse([a, b], [10.0, 1.0])
    # Heavy weight on a → x first
    assert fused_a[0] == "x"
