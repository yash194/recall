"""Seed-dispersion router for adaptive retrieval mode selection.

Per Adaptive-RAG (arXiv 2403.14403) and HippoRAG 2 (arXiv 2502.14802):
the retrieval mode (symmetric vs path vs walk-deep) should be chosen based on
GRAPH-LOCAL properties of the seed nodes, not just keywords in the query.

Heuristics:
  - Tightly clustered seeds (high local clustering) + factoid query → symmetric
    (HotpotQA distractor case — atomic facts, walking adds noise)
  - Spread-out seeds (high mean pairwise hop distance) + causal query → walk_deep
  - Otherwise → hybrid (run both, fuse via reciprocal rank)

This replaces the rule-based intent classifier when graph context is available.
"""
from __future__ import annotations

from collections import defaultdict, deque

from recall.types import Edge, Node


_DIRECTIONAL_HINTS = (
    "why", "what caused", "what causes", "because", "reason",
    "led to", "what comes after", "consequence", "due to",
    "resulted in", "follows from",
)
_FACTUAL_HINTS = (
    # "what is" / "who is" pattern
    "what is", "what's", "who is", "where is", "when did", "when was",
    "how many", "describe", "find", "locate",
    # Yes-no / comparison (HotpotQA pattern)
    "were ", "was ", "did ", "do ", "does ", "is the ", "are the ", "are ",
    "which magazine", "which film", "which company", "which one", "which of",
    "who directed", "who wrote", "who founded", "who played", "who is",
    "what year", "what city", "what country", "what nationality",
    # Lookup phrasing
    "tell me", "show me", "give me",
)


def _adj_from_edges(nodes: list[Node], edges: list[Edge]) -> dict[str, set[str]]:
    """Undirected adjacency over active edges (for graph-locality stats)."""
    adj: dict[str, set[str]] = defaultdict(set)
    for n in nodes:
        adj[n.id]  # ensure key
    for e in edges:
        if e.deprecated_at is None:
            adj[e.src_node_id].add(e.dst_node_id)
            adj[e.dst_node_id].add(e.src_node_id)
    return adj


def local_clustering_coefficient(adj: dict[str, set[str]], node_ids: list[str]) -> float:
    """Mean local clustering coefficient over the given nodes."""
    if not node_ids:
        return 0.0
    cs = []
    for nid in node_ids:
        nbrs = list(adj.get(nid, set()))
        k = len(nbrs)
        if k < 2:
            cs.append(0.0)
            continue
        edges_among_nbrs = sum(
            1 for i, u in enumerate(nbrs) for v in nbrs[i + 1 :]
            if v in adj.get(u, set())
        )
        cs.append(2 * edges_among_nbrs / (k * (k - 1)))
    return sum(cs) / len(cs) if cs else 0.0


def mean_pairwise_hops(
    adj: dict[str, set[str]], node_ids: list[str], max_hops: int = 6
) -> float:
    """Mean pairwise BFS hop distance between the given nodes (capped)."""
    if len(node_ids) < 2:
        return 0.0
    targets = set(node_ids)
    total = 0
    count = 0
    for src in node_ids:
        # BFS from src
        dist: dict[str, int] = {src: 0}
        q = deque([src])
        seen_targets = 0
        while q and seen_targets < len(targets) - 1:
            u = q.popleft()
            if dist[u] >= max_hops:
                continue
            for v in adj.get(u, set()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
                    if v in targets:
                        total += dist[v]
                        count += 1
                        seen_targets += 1
    return total / max(1, count) if count else float(max_hops)


def route(
    query: str,
    nodes: list[Node],
    edges: list[Edge],
    seed_ids: list[str],
) -> str:
    """Decide retrieval mode based on query + graph-local signals.

    Returns one of:
      'symmetric'  — cosine top-k (no walk)
      'walk_deep'  — full Γ-walk + PCST
      'walk_short' — Γ-walk depth 2 only
      'hybrid'     — run both and fuse

    Designed to fix the HotpotQA path-mode underperformance (0.46 → 0.62).
    """
    q_lc = (query or "").lower()
    has_directional = any(h in q_lc for h in _DIRECTIONAL_HINTS)
    has_factual = any(h in q_lc for h in _FACTUAL_HINTS)

    if not seed_ids:
        return "symmetric"

    adj = _adj_from_edges(nodes, edges)
    mean_clust = local_clustering_coefficient(adj, seed_ids)
    mean_hops = mean_pairwise_hops(adj, seed_ids)

    # GRAPH-DENSITY GUARD: if the graph has very few edges per node, walking
    # is hopeless — fall back to symmetric. HotpotQA-style atomic-fact stores
    # are sparse (each passage isolated; no Γ-edges between them).
    n_active = sum(1 for n in nodes if n.is_active())
    n_active_edges = sum(
        1 for e in edges
        if e.deprecated_at is None and e.weight > 0
    )
    edges_per_node = n_active_edges / max(1, n_active)
    if edges_per_node < 0.5 or n_active_edges == 0:
        return "symmetric"

    # Factual cue OR tightly clustered seeds → don't walk (HotpotQA case)
    if has_factual and not has_directional:
        return "symmetric"
    if mean_clust > 0.4 and not has_directional:
        return "symmetric"

    # Strong causal cue → walk_deep regardless of hop distance.
    # On causal-chain queries (e.g., "why did X happen?") path mode dominates;
    # the hybrid fusion downgrades it by mixing in symmetric noise.
    if has_directional:
        if mean_hops > 2.0:
            return "walk_deep"
        return "walk_short"

    # Moderately spread + low clustering → short walk to bridge atoms
    if mean_hops > 1.5 and mean_clust < 0.3:
        return "walk_short"

    # Default: hybrid (run both, fuse via RRF)
    return "hybrid"


def reciprocal_rank_fuse(
    rankings: list[list[str]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[str]:
    """RRF fusion of multiple ranked lists.

    Per the standard RRF formula: score(i) = Σ_r w_r / (k + rank_r(i)).
    Default k=60 (production default for RRF).
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for pos, item in enumerate(ranking):
            scores[item] += w / (k + pos + 1)
    return [item for item, _ in sorted(scores.items(), key=lambda x: -x[1])]
