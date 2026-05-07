"""Tests for Prize-Collecting Steiner Forest with negative-edge anti-prize."""
from __future__ import annotations

import numpy as np

from recall.retrieval.pcsf import pcsf_extract
from recall.types import Edge, EdgeType, Node, Path


def _n(i):
    return Node(
        id=f"n{i}", tenant="t", text=f"node {i}",
        f_embedding=np.ones(4, dtype=np.float32),
        b_embedding=np.ones(4, dtype=np.float32),
        quality_status="promoted",
    )


def _e(i, src, dst, w, t=EdgeType.SUPPORTS):
    return Edge(
        id=f"e{i}", tenant="t", src_node_id=src, dst_node_id=dst,
        edge_type=t, weight=w, gamma_score=w, s_squared=1.0,
    )


def test_pcsf_empty_returns_empty():
    assert pcsf_extract([], budget=10.0) == ([], [])


def test_pcsf_picks_positive_chain():
    n1, n2, n3 = _n(1), _n(2), _n(3)
    e12 = _e(1, "n1", "n2", 0.6)
    e23 = _e(2, "n2", "n3", 0.5)
    nodes, edges = pcsf_extract([Path(nodes=[n1, n2, n3], edges=[e12, e23])], budget=10.0)
    ids = {n.id for n in nodes}
    assert {"n1", "n2", "n3"}.issubset(ids)


def test_pcsf_routes_around_contradicts():
    # Two paths to n3: one direct via positive, one via negative
    n1, n2, n3 = _n(1), _n(2), _n(3)
    pos = _e(1, "n1", "n2", 0.5, EdgeType.SUPPORTS)
    pos2 = _e(2, "n2", "n3", 0.5, EdgeType.SUPPORTS)
    neg = _e(3, "n1", "n3", -2.0, EdgeType.CONTRADICTS)
    p1 = Path(nodes=[n1, n2, n3], edges=[pos, pos2])
    p2 = Path(nodes=[n1, n3], edges=[neg])
    nodes, edges = pcsf_extract([p1, p2], budget=5.0)
    selected_edge_ids = {e.id for e in edges}
    # The negative edge should not be selected (high anti-prize)
    assert "e3" not in selected_edge_ids


def test_pcsf_handles_disconnected_components():
    """PCSF should handle multiple connected components (the 'F' in PCSF)."""
    # Component A: n1 - n2
    n1, n2 = _n(1), _n(2)
    eA = _e(1, "n1", "n2", 0.5)
    # Component B: n3 - n4 (disjoint)
    n3, n4 = _n(3), _n(4)
    eB = _e(2, "n3", "n4", 0.5)
    p = Path(nodes=[n1, n2, n3, n4], edges=[eA, eB])
    nodes, edges = pcsf_extract([p], budget=10.0)
    # Both components should have at least one node
    assert any(n.id in ("n1", "n2") for n in nodes)
    assert any(n.id in ("n3", "n4") for n in nodes)
