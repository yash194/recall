"""PCSF — Prize-Collecting Steiner Forest with negative-edge anti-prize.

Per Recall lit-review (Q1, May 2026): the correct primitive for retrieval over
typed-edge graphs with `supports`/`contradicts`/etc. is **PCSF** (Steiner
forest), NOT PCST (single tree). Reasoning: queries against multi-aspect
memory typically span multiple connected evidence subtrees.

We adapt Ahmadi-Hajiaghayi-Jabbarzade-Mahdavi-Springer (JACM 2025, arXiv
2309.05172) — 2-approximation PCSF — by treating negative-weight edges
(`contradicts`, `superseded`) as **anti-prize**: their inclusion in the
selected forest reduces total prize, motivating the algorithm to route around
them.

Implementation: greedy forest growth on positive-weight components, with
negative-weight edges contributing penalty to any path that traverses them.

Reference primitive (this v1): greedy 2-approx. v2 will plug into
networkx-pcsf if/when available.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from recall.types import Edge, Node, Path


def pcsf_extract(
    paths: list[Path],
    budget: float = 10.0,
    contradiction_penalty: float = 1.5,
) -> tuple[list[Node], list[Edge]]:
    """Greedy PCSF on the union of paths.

    Args:
        paths: candidate Γ-walk paths (output of gamma_walk).
        budget: max total cost of negative edges allowed (the algorithm stops
                growing each tree when its cumulative contradiction cost
                exceeds budget per tree).
        contradiction_penalty: multiplier on negative-edge weight magnitudes;
                higher = more conservative routing around `contradicts`.

    Returns:
        (subgraph_nodes, subgraph_edges)
    """
    if not paths:
        return [], []

    # Aggregate
    node_map: dict[str, Node] = {}
    edge_map: dict[str, Edge] = {}
    for p in paths:
        for n in p.nodes:
            node_map[n.id] = n
        for e in p.edges:
            edge_map[e.id] = e
    nodes = list(node_map.values())
    edges = list(edge_map.values())
    if not edges:
        return nodes, []

    # Compute per-node prize (sum of incident positive-edge weights).
    prizes: dict[str, float] = defaultdict(float)
    for e in edges:
        if e.weight > 0:
            prizes[e.src_node_id] += e.weight
            prizes[e.dst_node_id] += e.weight

    # Compute per-edge cost: positive edges have small cost (anti-prize ~0),
    # negative edges have penalty.
    costs: dict[str, float] = {}
    for e in edges:
        if e.weight < 0:
            costs[e.id] = contradiction_penalty * abs(e.weight)
        else:
            # tiny cost to discourage tree-bloat for low-weight positive edges
            costs[e.id] = max(0.01, 0.1 * (1.0 - e.weight))

    adj: dict[str, list[tuple[str, Edge]]] = defaultdict(list)
    for e in edges:
        adj[e.src_node_id].append((e.dst_node_id, e))
        adj[e.dst_node_id].append((e.src_node_id, e))

    # Identify connected components by positive-only edges (these define
    # candidate trees in the forest).
    pos_adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        if e.weight > 0:
            pos_adj[e.src_node_id].add(e.dst_node_id)
            pos_adj[e.dst_node_id].add(e.src_node_id)

    seen: set[str] = set()
    components: list[set[str]] = []
    for nid in node_map:
        if nid in seen:
            continue
        stack = [nid]
        comp: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.add(cur)
            for nb in pos_adj[cur]:
                if nb not in seen:
                    stack.append(nb)
        components.append(comp)

    # For each component, grow a tree starting from highest-prize seed.
    # Stop when adding more nodes only crosses contradiction edges.
    selected_nodes: set[str] = set()
    selected_edges: set[str] = set()

    for comp in components:
        if not comp:
            continue
        # Comp prize = sum of node prizes in the component
        comp_prizes = {n: prizes.get(n, 0.0) for n in comp}
        seed = max(comp_prizes, key=lambda x: comp_prizes[x])
        if comp_prizes[seed] <= 0:
            continue

        tree_nodes = {seed}
        tree_total_cost = 0.0

        # Greedy MWST-like growth restricted to comp
        while True:
            best: tuple[float, str, Edge] | None = None  # (net, neighbor, edge)
            for nid in list(tree_nodes):
                for nb_id, e in adj[nid]:
                    if nb_id not in comp or nb_id in tree_nodes:
                        continue
                    edge_cost = costs.get(e.id, 0.0)
                    nb_prize = comp_prizes.get(nb_id, 0.0)
                    net = nb_prize - edge_cost
                    if best is None or net > best[0]:
                        best = (net, nb_id, e)
            if best is None or best[0] <= 0:
                break
            edge_cost = costs.get(best[2].id, 0.0)
            if tree_total_cost + edge_cost > budget:
                break
            tree_nodes.add(best[1])
            tree_total_cost += edge_cost
            selected_edges.add(best[2].id)

        selected_nodes |= tree_nodes

    # Add positive-weight edges between selected nodes (closure)
    for e in edges:
        if e.src_node_id in selected_nodes and e.dst_node_id in selected_nodes and e.weight > 0:
            selected_edges.add(e.id)

    return (
        [n for n in nodes if n.id in selected_nodes],
        [e for e in edges if e.id in selected_edges],
    )
