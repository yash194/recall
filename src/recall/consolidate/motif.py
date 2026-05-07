"""Motif extraction — Mosaic-of-Motifs adapted to typed-edge memory graphs.

Implements MATH.md §4.5 — Selvan-Bakhtiarifard et al. 2026.

A motif is a recurring subgraph pattern. We look for length-2 and length-3
typed-edge chains that appear in ≥ k regions. v1 uses pattern fingerprints
based on edge-type sequences; v2 will use full subgraph isomorphism.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from recall.types import Edge, EdgeType


def _edge_type_str(e: Edge) -> str:
    return e.edge_type.value if hasattr(e.edge_type, "value") else str(e.edge_type)


def find_recurring_subgraphs(
    region: set[str], edges: list[Edge], min_occurrences: int = 2
) -> list[dict]:
    """Identify recurring length-2 typed-edge chains in a region.

    Returns a list of {pattern: 'a → b → c', count: N, instances: [(n1, n2, n3), ...]}.
    """
    # Build forward adjacency restricted to region
    fwd: dict[str, list[Edge]] = defaultdict(list)
    for e in edges:
        if e.src_node_id in region and e.dst_node_id in region and e.weight > 0:
            fwd[e.src_node_id].append(e)

    # Length-2 chains: a -[t1]-> b -[t2]-> c
    chain_patterns: Counter = Counter()
    chain_instances: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for a, edges_a in fwd.items():
        for e1 in edges_a:
            b = e1.dst_node_id
            for e2 in fwd.get(b, []):
                c = e2.dst_node_id
                if c == a:
                    continue
                pattern = f"{_edge_type_str(e1)} → {_edge_type_str(e2)}"
                chain_patterns[pattern] += 1
                chain_instances[pattern].append((a, b, c))

    motifs: list[dict] = []
    for pattern, count in chain_patterns.items():
        if count >= min_occurrences:
            motifs.append({
                "pattern": pattern,
                "count": count,
                "instances": chain_instances[pattern],
            })
    return motifs
