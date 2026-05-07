"""Tests for Γ-walk retrieval."""
from __future__ import annotations

import numpy as np
import pytest

from recall.core.storage import SQLiteStorage
from recall.retrieval.walk import gamma_walk
from recall.types import Edge, EdgeType, Node


def _node(id_, text="t", f=None, b=None):
    if f is None:
        f = np.ones(8, dtype=np.float32)
    if b is None:
        b = np.ones(8, dtype=np.float32)
    return Node(id=id_, tenant="t1", text=text, f_embedding=f, b_embedding=b,
                quality_status="promoted")


@pytest.fixture
def storage():
    s = SQLiteStorage(tenant="t1", db_path=":memory:", embed_dim=8)
    yield s
    s.close()


def test_walk_returns_seed_when_no_edges(storage):
    n = _node("n1")
    storage.insert_node(n)
    paths = gamma_walk(storage, n, depth=3, weight_threshold=0.0, beam_width=4)
    # Single path with just the seed
    assert len(paths) >= 1
    assert paths[0].nodes[0].id == "n1"
    assert len(paths[0].edges) == 0


def test_walk_traverses_one_edge(storage):
    n1 = _node("n1")
    n2 = _node("n2")
    storage.insert_node(n1)
    storage.insert_node(n2)
    e = Edge(id="e1", tenant="t1", src_node_id="n1", dst_node_id="n2",
             edge_type=EdgeType.SUPPORTS, weight=0.5, gamma_score=0.5, s_squared=1.0)
    storage.insert_edge(e)

    paths = gamma_walk(storage, n1, depth=3, weight_threshold=0.0, beam_width=4)
    # At least one path should reach n2
    reached = [p for p in paths if any(n.id == "n2" for n in p.nodes)]
    assert len(reached) >= 1


def test_walk_skips_deprecated_edges(storage):
    n1 = _node("n1")
    n2 = _node("n2")
    storage.insert_node(n1)
    storage.insert_node(n2)
    e = Edge(id="e1", tenant="t1", src_node_id="n1", dst_node_id="n2",
             edge_type=EdgeType.SUPPORTS, weight=0.5, gamma_score=0.5, s_squared=1.0)
    storage.insert_edge(e)
    storage.deprecate_edge("e1", reason="test")
    paths = gamma_walk(storage, n1, depth=3, weight_threshold=0.0, beam_width=4)
    reached = [p for p in paths if any(n.id == "n2" for n in p.nodes)]
    assert len(reached) == 0  # deprecated edge → not traversed


def test_walk_respects_weight_threshold(storage):
    n1 = _node("n1")
    n2 = _node("n2")
    storage.insert_node(n1)
    storage.insert_node(n2)
    e = Edge(id="e1", tenant="t1", src_node_id="n1", dst_node_id="n2",
             edge_type=EdgeType.SUPPORTS, weight=0.1, gamma_score=0.1, s_squared=1.0)
    storage.insert_edge(e)

    paths_low = gamma_walk(storage, n1, depth=2, weight_threshold=0.05, beam_width=4)
    reached_low = [p for p in paths_low if any(n.id == "n2" for n in p.nodes)]
    assert len(reached_low) >= 1

    paths_high = gamma_walk(storage, n1, depth=2, weight_threshold=0.5, beam_width=4)
    reached_high = [p for p in paths_high if any(n.id == "n2" for n in p.nodes)]
    assert len(reached_high) == 0  # below threshold


def test_walk_does_not_cycle(storage):
    n1 = _node("n1")
    n2 = _node("n2")
    storage.insert_node(n1)
    storage.insert_node(n2)
    storage.insert_edge(Edge(id="e1", tenant="t1", src_node_id="n1", dst_node_id="n2",
                             edge_type=EdgeType.SUPPORTS, weight=0.5, gamma_score=0.5,
                             s_squared=1.0))
    storage.insert_edge(Edge(id="e2", tenant="t1", src_node_id="n2", dst_node_id="n1",
                             edge_type=EdgeType.SUPPORTS, weight=0.5, gamma_score=0.5,
                             s_squared=1.0))
    paths = gamma_walk(storage, n1, depth=10, weight_threshold=0.0, beam_width=4)
    for p in paths:
        ids = [n.id for n in p.nodes]
        assert len(ids) == len(set(ids))  # no repeats
