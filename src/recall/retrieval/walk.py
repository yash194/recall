"""Γ-walk — beam-search retrieval over Γ-weighted edges.

Implements ARCHITECTURE.md §5.3. Given a seed node, traverses outgoing edges
with weight ≥ threshold up to a fixed depth, retaining the top-K paths by
cumulative weight.
"""
from __future__ import annotations

from recall.core.storage import Storage
from recall.types import Edge, Node, Path


def gamma_walk(
    storage: Storage,
    seed: Node,
    depth: int = 4,
    weight_threshold: float = 0.0,
    beam_width: int = 8,
) -> list[Path]:
    """Beam-search forward Γ-walk from `seed`.

    Returns up to `beam_width` paths of length ≤ depth.
    """
    paths: list[Path] = [Path(nodes=[seed], edges=[])]
    for _ in range(depth):
        next_paths: list[Path] = []
        for p in paths:
            tail = p.nodes[-1]
            outgoing = storage.get_edges_from(tail.id)
            extended_any = False
            for edge in outgoing:
                if edge.deprecated_at is not None:
                    continue
                if edge.weight < weight_threshold:
                    continue
                next_node = storage.get_node(edge.dst_node_id)
                if next_node is None or not next_node.is_active():
                    continue
                # avoid cycles in the same path
                if any(n.id == next_node.id for n in p.nodes):
                    continue
                next_paths.append(p.extend(next_node, edge))
                extended_any = True
            if not extended_any:
                # keep dead-end path; it has whatever weight it accumulated
                next_paths.append(p)
        # Beam: keep top-`beam_width` paths by cumulative weight
        next_paths.sort(key=lambda pp: -pp.cum_weight)
        paths = next_paths[:beam_width]
    return paths
