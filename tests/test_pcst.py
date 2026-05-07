"""Tests for PCST signed-weight subgraph extraction."""
from __future__ import annotations

import numpy as np

from recall.retrieval.pcst import pcst_extract
from recall.types import Edge, EdgeType, Node, Path


def _n(i):
    return Node(
        id=f"n{i}", tenant="t1", text=f"node {i}",
        f_embedding=np.ones(4, dtype=np.float32),
        b_embedding=np.ones(4, dtype=np.float32),
        quality_status="promoted",
    )


def _e(i, src, dst, w):
    return Edge(
        id=f"e{i}", tenant="t1", src_node_id=src, dst_node_id=dst,
        edge_type=EdgeType.SUPPORTS, weight=w, gamma_score=w, s_squared=1.0,
    )


def test_pcst_empty_returns_empty():
    nodes, edges = pcst_extract([], budget=10.0)
    assert nodes == [] and edges == []


def test_pcst_picks_positive_weight_chain():
    n1, n2, n3 = _n(1), _n(2), _n(3)
    e12 = _e(1, "n1", "n2", 0.6)
    e23 = _e(2, "n2", "n3", 0.4)
    p = Path(nodes=[n1, n2, n3], edges=[e12, e23])
    nodes, edges = pcst_extract([p], budget=10.0)
    ids = {n.id for n in nodes}
    assert {"n1", "n2", "n3"}.issubset(ids)


def test_pcst_avoids_high_cost_edge_when_budget_low():
    n1, n2 = _n(1), _n(2)
    expensive = _e(1, "n1", "n2", -5.0)  # cost 5
    p = Path(nodes=[n1, n2], edges=[expensive])
    # With small budget, the cost can't be afforded
    nodes, edges = pcst_extract([p], budget=1.0)
    # Should at minimum have the seed; cost-5 edge should not be in selected_edges
    assert all(e.id != "e1" for e in edges)
