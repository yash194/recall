"""Mean-field GNN refinement of edge probabilities.

Implements MATH.md §4.6 — Selvan's mean-field approach (Graph Refinement
Airway Extraction, MedIA 2020). Iteratively smooths edge weights toward a
fixed point where each edge's probability is consistent with its neighborhood.

For each edge e:
    p(e) ← σ( Σ_{e' ∈ N(e)} α(e, e') · p(e') + β(e) )

Here we use a simplified version: edge weights are pulled toward the average
of their (typed-edge) neighbors, with a regularizer.
"""
from __future__ import annotations

import math
from collections import defaultdict

from recall.types import Edge


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def mean_field_iterate(edges: list[Edge], T: int = 5, alpha: float = 0.3, beta: float = 1.0) -> int:
    """Iterative mean-field refinement of edge weights.

    Args:
        edges: list of Edge objects (mutated in place — `weight` updated).
        T: number of iterations.
        alpha: neighbor influence rate.
        beta: self-prior strength.

    Returns:
        Number of edges whose weights changed by > 0.01 (refined).
    """
    if not edges:
        return 0

    # Build neighborhood index: edges that share a node
    neighbors: dict[str, list[Edge]] = defaultdict(list)
    by_id: dict[str, Edge] = {e.id: e for e in edges}
    for e in edges:
        neighbors[e.src_node_id].append(e)
        neighbors[e.dst_node_id].append(e)

    initial_weights = {e.id: e.weight for e in edges}

    for _ in range(T):
        new_weights: dict[str, float] = {}
        for e in edges:
            # Same-typed neighbors (incident through src or dst)
            same_type_neighbors = []
            for n_id in (e.src_node_id, e.dst_node_id):
                for ne in neighbors[n_id]:
                    if ne.id == e.id:
                        continue
                    if ne.edge_type == e.edge_type:
                        same_type_neighbors.append(ne.weight)
            if same_type_neighbors:
                neighbor_avg = sum(same_type_neighbors) / len(same_type_neighbors)
            else:
                neighbor_avg = 0.0
            # Update: pull toward neighbor average, anchored by initial weight
            new = (1 - alpha) * e.weight + alpha * neighbor_avg
            # Soft regularizer toward initial (β factor)
            new = (new + beta * initial_weights[e.id]) / (1 + beta)
            new_weights[e.id] = new

        for e in edges:
            e.weight = new_weights[e.id]

    refined = 0
    for e in edges:
        if abs(e.weight - initial_weights[e.id]) > 0.01:
            refined += 1
    return refined
