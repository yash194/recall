"""Tests for PMED experience scoring."""
from __future__ import annotations

import numpy as np

from recall.consolidate.pmed_score import (
    PMEDComponents,
    compute_pmed_components,
    correction_depth,
    debate_uplift,
    disagreement_collapse_rate,
    pmed_priority,
    rarity_boost,
    reasoning_path_divergence,
    sycophancy_penalty,
)
from recall.types import Edge, EdgeType, Node


def _node_with_emb(i, dim=16, seed=42, role="fact"):
    rng = np.random.default_rng(seed + i)
    f = rng.standard_normal(dim).astype(np.float32)
    f = f / float(np.linalg.norm(f))
    b = rng.standard_normal(dim).astype(np.float32)
    b = b / float(np.linalg.norm(b))
    return Node(
        id=f"n{i}", tenant="t", text=f"node {i}",
        f_embedding=f, b_embedding=b,
        role=role, quality_score=0.5, quality_status="promoted",
    )


def _edge(i, src, dst, w=0.5, t=EdgeType.SUPPORTS):
    return Edge(
        id=f"e{i}", tenant="t", src_node_id=src, dst_node_id=dst,
        edge_type=t, weight=w, gamma_score=w, s_squared=1.0,
    )


def test_components_empty():
    pc = compute_pmed_components([], [], 0)
    assert pc.D_RPD == 0.0
    assert pc.DCR == 0.0


def test_components_full():
    nodes = [_node_with_emb(i) for i in range(5)]
    edges = [_edge(i, f"n{i}", f"n{i+1}") for i in range(4)]
    pc = compute_pmed_components(nodes, edges, total_nodes_in_memory=20)
    assert isinstance(pc, PMEDComponents)
    assert 0.0 <= pc.D_RPD <= 2.0
    assert 0.0 <= pc.DCR <= 1.0
    assert 0.0 <= pc.Q_rare <= 1.0


def test_pmed_priority_sums_correctly():
    pc = PMEDComponents(D_RPD=1.0, DCR=0.5, P_syco=0.2, Q_corr=0.3, Q_eff=0.4, Q_rare=0.5)
    score = pmed_priority(pc)
    # 1.0 * 1.0 - 1.0 * 0.5 - 1.0 * 0.2 + 1.0 * 0.3 + 0.5 * 0.4 + 0.5 * 0.5
    expected = 1.0 - 0.5 - 0.2 + 0.3 + 0.2 + 0.25
    assert abs(score - expected) < 1e-6


def test_correction_depth_with_pivot_edge_boosted():
    n0 = _node_with_emb(0, seed=10)
    n1 = _node_with_emb(1, seed=20)
    n0.quality_score = 0.3
    n1.quality_score = 0.7
    edges = [_edge(0, "n0", "n1", 0.5, EdgeType.PIVOTS)]
    q = correction_depth([n0, n1], edges)
    assert q > 0


def test_rarity_decreases_with_size():
    n_few = [_node_with_emb(0)]
    n_many = [_node_with_emb(i) for i in range(20)]
    rare_few = rarity_boost(n_few, all_nodes_count=100)
    rare_many = rarity_boost(n_many, all_nodes_count=100)
    assert rare_few > rare_many


def test_sycophancy_returns_value():
    nodes = [_node_with_emb(0, seed=5), _node_with_emb(1, seed=5)]  # nearly identical
    edges = [_edge(0, "n0", "n1", 0.5, EdgeType.AGREES)]
    p = sycophancy_penalty(nodes, edges)
    assert 0.0 <= p <= 1.0
