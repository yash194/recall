"""Tests for persistent-homology / topology primitives."""
from __future__ import annotations

import numpy as np
import pytest

from recall.graph.topology import (
    persistent_homology_summary,
    topological_signature,
)
from recall.types import Node


def _make_nodes(n: int, dim: int = 16, seed: int = 42) -> list[Node]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        f = rng.standard_normal(dim).astype(np.float32)
        f = f / max(1e-6, float(np.linalg.norm(f)))
        b = rng.standard_normal(dim).astype(np.float32)
        b = b / max(1e-6, float(np.linalg.norm(b)))
        out.append(Node(
            id=f"n{i}", tenant="t", text=f"node {i}",
            f_embedding=f, b_embedding=b, quality_status="promoted",
        ))
    return out


def test_topology_empty_input():
    summary = persistent_homology_summary([])
    assert summary["n_nodes"] == 0


def test_topology_summary_structure():
    nodes = _make_nodes(20)
    summary = persistent_homology_summary(nodes, max_edge_length=2.0)
    assert "n_nodes" in summary
    assert "backend" in summary
    assert "betti_0" in summary
    # If gudhi worked, betti_0 is final component count (eventually 1 for a complete graph at length 2)
    assert summary["betti_0"] >= 1


def test_topological_signature_shape():
    nodes = _make_nodes(15)
    sig = topological_signature(nodes)
    assert sig.shape == (6,)


def test_topology_robust_to_no_embeddings():
    nodes = [Node(id="n1", tenant="t", text="x", quality_status="promoted")]
    summary = persistent_homology_summary(nodes)
    # Should not crash
    assert "n_nodes" in summary
