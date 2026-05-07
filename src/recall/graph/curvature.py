"""Curvature on memory graphs — bottleneck detection for the consolidator.

Two complementary curvature notions:

1. **Forman-Ricci** (combinatorial; v0.7 default for bottleneck detection).
   For an edge e=(u,v) in an unweighted graph:
       κ_F(e) = 4 - deg(u) - deg(v) + 2 · |N(u) ∩ N(v)|
   κ_F is strongly negative for bridge edges between dense components and
   positive for edges embedded in triangle-rich communities. Reference:
   Forman 2003, Sreejith et al. 2016 (Forman curvature for complex networks).

2. **Ollivier-Ricci** (transport-based; lazy random walk).
   κ_O(u, v) = 1 - W_1(μ_u, μ_v) / d(u, v)
   The Recall implementation uses a fast TV-distance approximation
   (W_1 ≈ 1 - overlap) which is exact only for adjacent supports —
   accurate for cluster-internal edges, blind to bottlenecks. Use
   ``compute_forman_ricci`` for bottleneck protection. Reference:
   Ollivier 2009, Sandhu et al. 2015, Topping et al. ICLR 2022.

Applications in Recall:
  1. Curvature-aware pruning — bottleneck edges are essential connectors.
     The consolidator skips them in BMRS pruning even when their weight
     is low. This is *complementary* to BMRS, which prunes by edge-weight
     evidence; curvature prunes by graph-topological role.
  2. Community detection — group nodes connected by high-curvature edges.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from recall.types import Edge, Node


def compute_ollivier_ricci(
    nodes: list[Node], edges: list[Edge], alpha: float = 0.5
) -> dict[str, float]:
    """Compute Ollivier-Ricci curvature for each edge.

    Args:
        alpha: lazy random-walk parameter (1 - alpha is staying-put prob).

    Returns:
        Dict edge_id → curvature κ ∈ [-1, 1] (approximate; bounded by Wasserstein).
    """
    if not nodes or not edges:
        return {}

    idx = {n.id: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Build undirected weighted adjacency (using |weight|)
    adj: list[dict[int, float]] = [defaultdict(float) for _ in range(n)]
    for e in edges:
        i = idx.get(e.src_node_id)
        j = idx.get(e.dst_node_id)
        if i is None or j is None or i == j:
            continue
        w = abs(e.weight)
        adj[i][j] = max(adj[i][j], w)
        adj[j][i] = max(adj[j][i], w)

    # Lazy random-walk distribution at node i
    def measure(i: int) -> dict[int, float]:
        nbrs = adj[i]
        deg = sum(nbrs.values())
        if deg <= 0:
            return {i: 1.0}
        m = {i: 1.0 - alpha}
        for j, w in nbrs.items():
            m[j] = m.get(j, 0.0) + alpha * w / deg
        return m

    out: dict[str, float] = {}
    for e in edges:
        i = idx.get(e.src_node_id)
        j = idx.get(e.dst_node_id)
        if i is None or j is None or i == j:
            continue
        mu_i = measure(i)
        mu_j = measure(j)

        # 1-Wasserstein on the integer support; approximated by sorted-list trick
        # using shortest-path graph metric. For efficiency we use the simpler
        # bound: W_1 ≈ 1 - sum_k min(μ_i[k], μ_j[k])  (transportation cap)
        common_overlap = sum(min(mu_i.get(k, 0.0), mu_j.get(k, 0.0))
                             for k in set(mu_i) | set(mu_j))
        # Approx W_1 = 1 - overlap (since both measures sum to 1)
        w1 = 1.0 - common_overlap
        edge_len = 1.0 / max(adj[i][j], 0.01)
        # Normalize so that for d(u,v)=1, kappa = 1 - W_1
        kappa = 1.0 - w1 / max(edge_len, 1.0)
        out[e.id] = float(kappa)
    return out


def compute_forman_ricci(
    nodes: list[Node], edges: list[Edge]
) -> dict[str, float]:
    """Combinatorial Forman-Ricci curvature for each edge.

    For an undirected edge e=(u,v):
        κ_F(e) = 4 - deg(u) - deg(v) + 2 · |N(u) ∩ N(v)|

    Strongly negative for *bridge* edges (deg high, no shared neighbors);
    positive for edges inside a triangle-rich community.

    This is exact and fast: O(deg(u) + deg(v)) per edge. Use this for
    bottleneck protection in BMRS — the TV-approximation Ollivier-Ricci
    in this module is blind to bottleneck topology.
    """
    if not nodes or not edges:
        return {}
    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        if e.src_node_id != e.dst_node_id:
            adj[e.src_node_id].add(e.dst_node_id)
            adj[e.dst_node_id].add(e.src_node_id)
    out: dict[str, float] = {}
    for e in edges:
        u, v = e.src_node_id, e.dst_node_id
        if u == v:
            out[e.id] = 0.0
            continue
        deg_u = len(adj[u])
        deg_v = len(adj[v])
        triangles = len(adj[u] & adj[v])
        out[e.id] = float(4 - deg_u - deg_v + 2 * triangles)
    return out


def curvature_pruning_signal(
    nodes: list[Node], edges: list[Edge], threshold: float = 0.0,
    method: str = "forman",
) -> list[str]:
    """Identify edges that BMRS shouldn't prune even if their weight is low.

    Negative-curvature edges are bottlenecks — removing them disconnects
    communities. We surface them as "protected" so the consolidator skips
    them in BMRS pruning.

    v0.7: default switched from Ollivier-Ricci (the TV approximation in
    this module is blind to bottleneck topology — a bridge between two
    K4 components scored κ ≈ +0.74 instead of strongly negative) to
    Forman-Ricci, which correctly returns negative values for bridges.
    Pass ``method='ollivier'`` to use the legacy curvature.

    Args:
        threshold: edges with curvature below ``-threshold`` are protected.
            Default 0.0 protects any negative-curvature edge. For Forman,
            higher threshold → require stronger bottleneck.
        method: 'forman' (default) or 'ollivier'.

    Returns:
        List of edge ids that are bottleneck connectors and should be
        kept by the consolidator.
    """
    if method == "ollivier":
        curv = compute_ollivier_ricci(nodes, edges)
    else:
        curv = compute_forman_ricci(nodes, edges)
    return [eid for eid, k in curv.items() if k < -threshold]


def curvature_summary(nodes: list[Node], edges: list[Edge]) -> dict[str, float]:
    """High-level curvature health metrics for the graph."""
    curv = compute_ollivier_ricci(nodes, edges)
    if not curv:
        return {"n_edges": 0}
    vals = list(curv.values())
    return {
        "n_edges": len(curv),
        "mean_curvature": float(np.mean(vals)),
        "median_curvature": float(np.median(vals)),
        "min_curvature": float(np.min(vals)),
        "max_curvature": float(np.max(vals)),
        "n_bottleneck_edges": int(sum(1 for v in vals if v < 0)),
        "n_community_edges": int(sum(1 for v in vals if v > 0)),
    }
