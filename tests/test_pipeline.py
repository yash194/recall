"""Tests for the write pipeline (observe path)."""
from __future__ import annotations

import pytest

from recall import HashEmbedder, MockLLMClient
from recall.audit.log import AuditLog
from recall.config import DEFAULT
from recall.core.storage import SQLiteStorage
from recall.write.pipeline import WritePipeline


@pytest.fixture
def pipeline():
    storage = SQLiteStorage(tenant="t1", db_path=":memory:", embed_dim=64)
    embedder = HashEmbedder(dim=64)
    audit = AuditLog(storage)
    llm = MockLLMClient()
    return WritePipeline(
        tenant="t1", storage=storage, embedder=embedder, audit=audit, llm=llm
    ), storage, audit


def test_observe_writes_drawer_and_nodes(pipeline):
    pipe, storage, audit = pipeline
    result = pipe.observe(
        "We tried Postgres LISTEN/NOTIFY for the queue.",
        "AGENT: That can lose messages under load.",
        scope={"project": "infra"},
    )
    assert result.drawer_id is not None
    assert storage.has_drawer(result.drawer_id)
    assert len(result.nodes_written) >= 1


def test_provenance_firewall_rejects_recall_artifact(pipeline):
    pipe, storage, _ = pipeline
    result = pipe.observe(
        "user said something", "agent echo",
        scope={}, source="recall_artifact",
    )
    assert result.skipped_recall_loop is True
    assert result.drawer_id is None
    assert len(result.nodes_written) == 0


def test_observe_dedup_on_repeat(pipeline):
    pipe, storage, _ = pipeline
    pipe.observe("a fresh fact about widgets", "ok", scope={"x": 1})
    initial_nodes = len(storage.all_active_nodes())
    # Repeat exactly — should hit drawer-level dedup
    pipe.observe("a fresh fact about widgets", "ok", scope={"x": 1})
    final_nodes = len(storage.all_active_nodes())
    assert final_nodes == initial_nodes  # no new nodes written


def test_quality_gate_rejects_boilerplate(pipeline):
    pipe, storage, _ = pipeline
    result = pipe.observe(
        "Sure, I can help with that.",
        "As an AI language model, I cannot do that.",
        scope={},
    )
    # Both halves should be rejected by quality gate (template-match)
    assert all("low_quality" in reason for (_, reason) in result.nodes_rejected)


def test_observe_creates_edges_when_neighbors_exist(pipeline):
    pipe, storage, _ = pipeline
    # First observation creates baseline nodes
    pipe.observe(
        "We tried Postgres LISTEN/NOTIFY for messaging.",
        "It lost messages.",
        scope={"project": "infra"},
    )
    # Second observation should induce edges to the first
    result = pipe.observe(
        "We switched to Redis Streams for messaging.",
        "Stable since.",
        scope={"project": "infra"},
    )
    # At least some edges should be induced (Γ-driven)
    edges = storage.all_active_edges()
    assert len(edges) >= 1


def test_bulk_mode_skips_edge_induction(pipeline):
    """v0.5: source='document' (or fast=True) should skip Γ-edge induction.

    This unblocks bulk-document ingestion (MemoryAgentBench, RAG corpora).
    """
    pipe, storage, _ = pipeline
    # Two semantically-related observations under bulk mode
    pipe.observe(
        "We tried Postgres LISTEN/NOTIFY for messaging.",
        "It lost messages.",
        scope={"project": "infra"},
        source="document",
    )
    pipe.observe(
        "We switched to Redis Streams for messaging.",
        "Stable since.",
        scope={"project": "infra"},
        source="document",
    )
    # No edges should be induced in bulk mode (compare with the test above)
    assert len(storage.all_active_edges()) == 0
    # Nodes still get written
    assert len(storage.all_active_nodes()) >= 2


def test_explicit_fast_flag_overrides_source(pipeline):
    """fast=True wins over default source='conversation'."""
    pipe, storage, _ = pipeline
    pipe.observe(
        "We tried Postgres LISTEN/NOTIFY for messaging.",
        "It lost messages.",
        scope={"project": "infra"},
        fast=True,
    )
    pipe.observe(
        "We switched to Redis Streams for messaging.",
        "Stable since.",
        scope={"project": "infra"},
        fast=True,
    )
    assert len(storage.all_active_edges()) == 0


def test_audit_log_records_writes(pipeline):
    pipe, storage, audit = pipeline
    pipe.observe("Decided to use Redis Streams.", "Confirmed.", scope={"p": 1})
    entries = audit.since(_epoch())
    ops = {e.operation for e in entries}
    assert "WRITE" in ops  # drawer write
    assert any(e.operation in ("PROMOTE", "REJECT") for e in entries)


def _epoch():
    from datetime import datetime, timezone
    return datetime(1970, 1, 1, tzinfo=timezone.utc)
