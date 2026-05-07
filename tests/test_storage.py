"""Tests for SQLite storage layer."""
from __future__ import annotations

import numpy as np
import pytest

from recall.core.storage import SQLiteStorage
from recall.types import AuditEntry, Drawer, Edge, EdgeType, Node


@pytest.fixture
def storage():
    s = SQLiteStorage(tenant="t1", db_path=":memory:", embed_dim=8)
    yield s
    s.close()


def test_drawer_roundtrip(storage):
    d = Drawer(id="d1", tenant="t1", text="hello world", source="conversation",
               scope={"project": "P1"})
    storage.insert_drawer(d)
    assert storage.has_drawer("d1")
    fetched = storage.get_drawer("d1")
    assert fetched is not None
    assert fetched.text == "hello world"
    assert fetched.scope == {"project": "P1"}


def test_drawer_dedup(storage):
    d = Drawer(id="d1", tenant="t1", text="x", source="conversation")
    storage.insert_drawer(d)
    storage.insert_drawer(d)  # idempotent
    # If we got this far, the INSERT OR IGNORE worked


def test_node_roundtrip(storage):
    f = np.ones(8, dtype=np.float32)
    b = np.ones(8, dtype=np.float32) * -1
    n = Node(
        id="n1", tenant="t1", text="some thought",
        f_embedding=f, b_embedding=b,
        quality_score=0.7, quality_status="promoted",
        scope={"team": "platform"},
    )
    storage.insert_node(n)
    fetched = storage.get_node("n1")
    assert fetched is not None
    assert fetched.text == "some thought"
    np.testing.assert_allclose(fetched.f_embedding, f)
    np.testing.assert_allclose(fetched.b_embedding, b)


def test_edge_roundtrip(storage):
    n_a = Node(id="na", tenant="t1", text="A",
               f_embedding=np.ones(8, dtype=np.float32),
               b_embedding=np.ones(8, dtype=np.float32),
               quality_status="promoted")
    n_b = Node(id="nb", tenant="t1", text="B",
               f_embedding=np.ones(8, dtype=np.float32),
               b_embedding=np.ones(8, dtype=np.float32),
               quality_status="promoted")
    storage.insert_node(n_a)
    storage.insert_node(n_b)

    e = Edge(id="e1", tenant="t1", src_node_id="na", dst_node_id="nb",
             edge_type=EdgeType.SUPPORTS, weight=0.5, gamma_score=0.5,
             gamma_anti=0.2, s_squared=1.0)
    storage.insert_edge(e)
    fetched = storage.get_edge("e1")
    assert fetched is not None
    assert fetched.weight == 0.5
    assert fetched.edge_type == EdgeType.SUPPORTS

    edges_from_a = storage.get_edges_from("na")
    assert len(edges_from_a) == 1
    assert edges_from_a[0].id == "e1"


def test_deprecate_excludes_from_edges(storage):
    n_a = Node(id="na", tenant="t1", text="A",
               f_embedding=np.ones(8, dtype=np.float32),
               b_embedding=np.ones(8, dtype=np.float32),
               quality_status="promoted")
    n_b = Node(id="nb", tenant="t1", text="B",
               f_embedding=np.ones(8, dtype=np.float32),
               b_embedding=np.ones(8, dtype=np.float32),
               quality_status="promoted")
    storage.insert_node(n_a)
    storage.insert_node(n_b)
    e = Edge(id="e1", tenant="t1", src_node_id="na", dst_node_id="nb",
             edge_type=EdgeType.SUPPORTS, weight=0.5, gamma_score=0.5, s_squared=1.0)
    storage.insert_edge(e)

    storage.deprecate_edge("e1", reason="test")
    assert len(storage.get_edges_from("na")) == 0  # deprecated → excluded


def test_topk_cosine_returns_active_only(storage):
    # Insert two nodes; query embedding aligns with the second
    n_a = Node(id="na", tenant="t1", text="A",
               f_embedding=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
               b_embedding=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
               quality_status="promoted",
               scope={"s": "x"})
    n_b = Node(id="nb", tenant="t1", text="B",
               f_embedding=np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
               b_embedding=np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
               quality_status="promoted",
               scope={"s": "x"})
    storage.insert_node(n_a)
    storage.insert_node(n_b)

    q = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    results = storage.topk_cosine(q, scope={"s": "x"}, k=2)
    assert len(results) == 2
    assert results[0].id == "nb"  # cosine match


def test_audit_append_and_query(storage):
    storage.append_audit(
        AuditEntry(
            tenant="t1", operation="WRITE", actor="system",
            target_type="node", target_id="n1",
            payload={"k": "v"}, reason="test",
        )
    )
    storage.append_audit(
        AuditEntry(
            tenant="t1", operation="FORGET", actor="user",
            target_type="node", target_id="n1",
            payload={}, reason="outdated",
        )
    )
    entries = storage.query_audit(target_id="n1")
    assert len(entries) == 2
    ops = [e.operation for e in entries]
    assert ops == ["WRITE", "FORGET"]  # append-only ordering


def test_scope_subset_semantics_topk(storage):
    """v0.5: query scope must be a SUBSET of stored scope, not exact match.

    This is the bug LongMemEval exposed: storing with two-key scope and
    querying with one of them used to return 0 nodes (string-equality match).
    Now it should return both nodes.
    """
    n_a = Node(id="qa-1", tenant="t1", text="answer A",
               f_embedding=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
               b_embedding=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
               quality_status="promoted",
               scope={"qid": "q1", "session_id": "s1"})
    n_b = Node(id="qa-2", tenant="t1", text="answer B",
               f_embedding=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
               b_embedding=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
               quality_status="promoted",
               scope={"qid": "q1", "session_id": "s2"})
    n_other = Node(id="qa-3", tenant="t1", text="other Q",
                   f_embedding=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                   b_embedding=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                   quality_status="promoted",
                   scope={"qid": "q2", "session_id": "s1"})
    storage.insert_node(n_a)
    storage.insert_node(n_b)
    storage.insert_node(n_other)

    q = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    # Query scope is subset of stored scope for qa-1 and qa-2 (qid=q1)
    results = storage.topk_cosine(q, scope={"qid": "q1"}, k=5)
    ids = {r.id for r in results}
    assert ids == {"qa-1", "qa-2"}

    # Query that matches only one
    results = storage.topk_cosine(q, scope={"qid": "q2"}, k=5)
    assert {r.id for r in results} == {"qa-3"}

    # Empty scope returns all
    results = storage.topk_cosine(q, scope={}, k=5)
    assert {r.id for r in results} == {"qa-1", "qa-2", "qa-3"}


def test_scope_subset_semantics_all_active(storage):
    """all_active_nodes also uses subset semantics in v0.5."""
    n_a = Node(id="x", tenant="t1", text="x",
               f_embedding=np.ones(8, dtype=np.float32),
               b_embedding=np.ones(8, dtype=np.float32),
               quality_status="promoted",
               scope={"qid": "q1", "session_id": "s1"})
    n_b = Node(id="y", tenant="t1", text="y",
               f_embedding=np.ones(8, dtype=np.float32),
               b_embedding=np.ones(8, dtype=np.float32),
               quality_status="promoted",
               scope={"qid": "q2"})
    storage.insert_node(n_a)
    storage.insert_node(n_b)
    assert {n.id for n in storage.all_active_nodes(scope={"qid": "q1"})} == {"x"}
    assert {n.id for n in storage.all_active_nodes(scope={})} == {"x", "y"}
    assert {n.id for n in storage.all_active_nodes(scope=None)} == {"x", "y"}


def test_s_embedding_persists_through_roundtrip(storage):
    """v0.6: Node.s_embedding must round-trip through SQLite."""
    f = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    b = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    s = np.array([0.5, 0.5, 0.0, 0, 0, 0, 0, 0], dtype=np.float32)
    n = Node(
        id="se", tenant="t1", text="x",
        f_embedding=f, b_embedding=b, s_embedding=s,
        quality_status="promoted",
    )
    storage.insert_node(n)
    fetched = storage.get_node("se")
    assert fetched is not None
    assert fetched.s_embedding is not None
    np.testing.assert_allclose(fetched.s_embedding, s)


def test_topk_cosine_uses_s_embedding_when_present(storage):
    """v0.6: when s_embedding is set, retrieval uses it directly (not (f+b)/2).

    We craft a node where s_embedding points one direction but (f+b)/2 points
    a different direction, then verify the query in the s_embedding direction
    retrieves it.
    """
    f = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    b = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)  # (f+b)/2 = (0.5, 0.5, ...)
    s_truth = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    n = Node(
        id="t", tenant="t1", text="x",
        f_embedding=f, b_embedding=b, s_embedding=s_truth,
        quality_status="promoted",
    )
    storage.insert_node(n)

    # Query in the s_embedding direction → should retrieve
    q_match = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    hits = storage.topk_cosine(q_match, scope={}, k=1)
    assert len(hits) == 1 and hits[0].id == "t"


def test_adjacency_cache_invalidates_on_edge_deprecate(storage):
    """v0.6: deprecate_edge invalidates the cached adjacency."""
    n_a = Node(id="a", tenant="t1", text="A",
               f_embedding=np.ones(8, dtype=np.float32),
               b_embedding=np.ones(8, dtype=np.float32),
               quality_status="promoted")
    n_b = Node(id="b", tenant="t1", text="B",
               f_embedding=np.ones(8, dtype=np.float32),
               b_embedding=np.ones(8, dtype=np.float32),
               quality_status="promoted")
    storage.insert_node(n_a)
    storage.insert_node(n_b)
    e = Edge(id="e1", tenant="t1", src_node_id="a", dst_node_id="b",
             edge_type=EdgeType.SUPPORTS, weight=0.5, gamma_score=0.5, s_squared=1.0)
    storage.insert_edge(e)

    # Build the adjacency cache, confirm a↔b are connected
    adj = storage.adjacency()
    assert "b" in adj.get("a", set())
    assert "a" in adj.get("b", set())
    assert storage.n_active_edges() == 1

    # Deprecate the edge — adjacency should refresh
    storage.deprecate_edge("e1", reason="test")
    adj_after = storage.adjacency()
    assert "b" not in adj_after.get("a", set())
    assert storage.n_active_edges() == 0


def test_deprecate_node_invalidates_index(storage):
    """v0.5: deprecating a node must remove it from topk_cosine results."""
    n = Node(id="z", tenant="t1", text="z",
             f_embedding=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
             b_embedding=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
             quality_status="promoted")
    storage.insert_node(n)
    q = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    assert any(r.id == "z" for r in storage.topk_cosine(q, scope={}, k=5))
    storage.deprecate_node("z", reason="test")
    assert not any(r.id == "z" for r in storage.topk_cosine(q, scope={}, k=5))


def test_all_active_nodes_excludes_rejected_and_deprecated(storage):
    n_promoted = Node(id="p", tenant="t1", text="P",
                      f_embedding=np.ones(8, dtype=np.float32),
                      b_embedding=np.ones(8, dtype=np.float32),
                      quality_status="promoted")
    n_rejected = Node(id="r", tenant="t1", text="R",
                      f_embedding=np.ones(8, dtype=np.float32),
                      b_embedding=np.ones(8, dtype=np.float32),
                      quality_status="rejected")
    n_pending = Node(id="d", tenant="t1", text="D",
                     f_embedding=np.ones(8, dtype=np.float32),
                     b_embedding=np.ones(8, dtype=np.float32),
                     quality_status="pending")
    storage.insert_node(n_promoted)
    storage.insert_node(n_rejected)
    storage.insert_node(n_pending)
    actives = storage.all_active_nodes()
    ids = {n.id for n in actives}
    assert ids == {"p"}  # only promoted shows
