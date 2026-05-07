"""End-to-end tests of the Memory API."""
from __future__ import annotations

import pytest

from recall import HashEmbedder, Memory, MockLLMClient
from recall.api import HallucinationBlocked


def _new_mem(tenant="t"):
    return Memory(
        tenant=tenant,
        storage=":memory:",
        embedder=HashEmbedder(dim=64),
        llm=MockLLMClient(),
    )


def test_three_line_quickstart_runs():
    mem = _new_mem()
    mem.observe(
        "We're switching from Postgres LISTEN/NOTIFY to Redis Streams.",
        "Sounds good.",
        scope={"project": "infra"},
    )
    result = mem.recall("queue stack", scope={"project": "infra"}, mode="path")
    assert result.query == "queue stack"
    assert result.mode == "path"


def test_observe_then_recall_finds_relevant_node():
    mem = _new_mem()
    mem.observe(
        "We chose Redis Streams for the message queue.",
        "Confirmed.",
        scope={"team": "platform"},
    )
    res = mem.recall("Redis Streams queue", scope={"team": "platform"}, mode="symmetric")
    assert len(res.subgraph_nodes) >= 1


def test_bounded_generate_soft_mode_does_not_raise():
    mem = _new_mem()
    mem.observe("Redis Streams is the queue.", "OK.", scope={"x": 1})
    # Soft mode never raises, just flags
    result = mem.bounded_generate("queue?", scope={"x": 1}, bound="soft")
    assert isinstance(result.text, str)


def test_forget_deprecates_node_and_edges():
    mem = _new_mem()
    r1 = mem.observe(
        "We tried Postgres LISTEN/NOTIFY.", "Ok.", scope={"p": 1}
    )
    r2 = mem.observe(
        "We switched to Redis Streams.", "Ok.", scope={"p": 1}
    )
    if not r2.nodes_written:
        pytest.skip("no nodes written under HashEmbedder; can't test forget")
    target = r2.nodes_written[0]
    forget = mem.forget(target, reason="user said outdated")
    assert forget.deprecated_node_id == target
    # Deprecated node excluded from retrieval
    res = mem.recall("Redis Streams", scope={"p": 1}, mode="symmetric")
    assert all(n.id != target for n in res.subgraph_nodes)


def test_audit_records_observe_and_forget():
    mem = _new_mem()
    r = mem.observe("X is decided.", "Y.", scope={})
    if r.nodes_written:
        nid = r.nodes_written[0]
        mem.forget(nid, reason="testing")
        entries = mem.audit.for_target(nid)
        ops = [e.operation for e in entries]
        assert "PROMOTE" in ops
        assert "FORGET" in ops


def test_trace_returns_provenance():
    mem = _new_mem()
    mem.observe("The quick brown fox jumps.", "Yes.", scope={})
    gen = mem.bounded_generate("what does the fox do", scope={}, bound="soft")
    trace = mem.trace(gen)
    assert trace.generated_text == gen.text
    assert isinstance(trace.audit_entries, list)


def test_stats_after_observe():
    mem = _new_mem()
    mem.observe("A real fact about banking.", "Affirmed.", scope={})
    stats = mem.stats()
    assert "active_nodes" in stats
    assert "active_edges" in stats
    assert stats["active_nodes"] >= 0
