"""Tests for sleep-time consolidator (BMRS + mean-field + motif)."""
from __future__ import annotations

import numpy as np
import pytest

from recall import Memory, HashEmbedder, MockLLMClient
from recall.consolidate.mean_field import mean_field_iterate
from recall.consolidate.motif import find_recurring_subgraphs
from recall.types import Edge, EdgeType, Node


def _node(i, text="t"):
    return Node(
        id=f"n{i}", tenant="t1", text=f"{text} {i}",
        f_embedding=np.ones(8, dtype=np.float32),
        b_embedding=np.ones(8, dtype=np.float32),
        quality_status="promoted",
    )


def _edge(i, s, d, w, t=EdgeType.SUPPORTS):
    return Edge(
        id=f"e{i}", tenant="t1", src_node_id=s, dst_node_id=d,
        edge_type=t, weight=w, gamma_score=w, gamma_anti=w/2, s_squared=1.0,
    )


def test_mean_field_smooths_outlier():
    """A single outlier weight should pull toward neighborhood average."""
    edges = [
        _edge(1, "a", "b", 0.1, EdgeType.SUPPORTS),
        _edge(2, "b", "c", 0.5, EdgeType.SUPPORTS),  # outlier
        _edge(3, "c", "d", 0.1, EdgeType.SUPPORTS),
    ]
    initial_w2 = edges[1].weight
    mean_field_iterate(edges, T=10, alpha=0.5, beta=0.0)
    # Outlier weight should have moved toward neighborhood avg
    assert edges[1].weight < initial_w2


def test_motif_detects_repeated_chain():
    """Length-2 chain repeated should produce a motif."""
    edges = [
        _edge(1, "a1", "b1", 0.5, EdgeType.SUPPORTS),
        _edge(2, "b1", "c1", 0.5, EdgeType.PIVOTS),
        _edge(3, "a2", "b2", 0.5, EdgeType.SUPPORTS),
        _edge(4, "b2", "c2", 0.5, EdgeType.PIVOTS),
        _edge(5, "x1", "y1", 0.5, EdgeType.AGREES),  # different pattern
    ]
    region = {"a1", "a2", "b1", "b2", "c1", "c2", "x1", "y1"}
    motifs = find_recurring_subgraphs(region, edges, min_occurrences=2)
    patterns = [m["pattern"] for m in motifs]
    assert any("supports → pivots" in p for p in patterns)


def test_consolidator_runs_end_to_end():
    mem = Memory(tenant="con_test", embedder=HashEmbedder(dim=64), llm=MockLLMClient())
    for i in range(5):
        mem.observe(
            f"We made decision {i} about the queue and switched to Redis.",
            f"Confirmed step {i}.",
            scope={"p": "infra"},
        )
    stats = mem.consolidate(budget=5)
    assert stats.regions_processed >= 1


def test_consolidator_induces_edges_after_bulk_ingest():
    """v0.5: bulk-mode ingest skips edges; consolidate(induce_edges=True) adds them.

    Whether the induced edges survive subsequent BMRS pruning is a separate
    decision (BMRS prunes weakly-supported edges by design); the contract here
    is just that induction *runs* on bulk-ingested isolated nodes.
    """
    from recall.config import Config
    cfg = Config()
    cfg.THRESH_GAMMA = 0.005  # match scale_stress / longmemeval setting
    mem = Memory(
        tenant="con_induce",
        embedder=HashEmbedder(dim=64),
        llm=MockLLMClient(),
        config=cfg,
    )
    # Bulk ingest — fast path skips edge induction
    facts = [
        "Postgres LISTEN/NOTIFY had message loss issues at scale.",
        "We switched to Redis Streams for the messaging queue.",
        "Redis Streams gave us at-least-once delivery semantics.",
        "The queue migration to Redis Streams was completed in March 2024.",
        "Engineer Priya owns the queue infrastructure.",
    ]
    for t in facts:
        mem.observe(t, "", scope={"p": "infra"}, source="document")
    # No edges induced at write time (bulk path)
    assert len(mem.storage.all_active_edges()) == 0
    # Run consolidation with edge induction enabled
    stats = mem.consolidate(budget=5, induce_edges=True)
    # Edges should have been induced (BMRS may prune some afterward; that's fine)
    assert stats.edges_induced >= 1
