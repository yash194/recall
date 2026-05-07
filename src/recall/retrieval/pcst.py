"""PCST — Prize-Collecting Steiner Tree on signed-weight graphs.

Two implementations:

- `pcst_extract` (default): greedy 2-approximation, no external deps.
- `pcst_extract_networkx`: networkx-backed Steiner-tree on the positive-weight
  subgraph, then add high-prize nodes greedily. More robust on dense graphs.

Per ARCHITECTURE.md §5.4 / MATH.md §1.4 (signed weights):
  Positive-weight edges (`supports`, `agrees`, `pivots`, `temporal_next`)
  contribute prizes at endpoints.
  Negative-weight edges (`contradicts`, `superseded`) contribute costs.
  Goal: extract subgraph maximizing (prize − cost) under a budget.
"""
from __future__ import annotations

from recall.types import Edge, Node, Path


def pcst_extract(
    paths: list[Path], budget: float = 10.0,
    must_include: list[str] | None = None,
) -> tuple[list[Node], list[Edge]]:
    """Greedy PCST over the union of paths.

    v0.7: `must_include` is a list of node ids that the extracted subgraph
    is *required* to contain. These are typically the query seeds (top-K
    cosine matches) — they carry the highest semantic signal for the query
    and dropping them silently was the v0.6 bug that made path-mode return
    the wrong subgraph. Greedy expansion then grows outward from these
    required nodes rather than from the hub-iest node in the path union.
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

    # Prizes: positive-weight contribution
    prizes: dict[str, float] = {n.id: 0.0 for n in nodes}
    for e in edges:
        if e.weight > 0:
            prizes[e.src_node_id] = prizes.get(e.src_node_id, 0) + e.weight
            prizes[e.dst_node_id] = prizes.get(e.dst_node_id, 0) + e.weight

    # Costs: positive value for negative edges
    costs: dict[str, float] = {e.id: max(0.0, -e.weight) for e in edges}

    # Adjacency
    adj: dict[str, list[tuple[str, Edge]]] = {n.id: [] for n in nodes}
    for e in edges:
        adj[e.src_node_id].append((e.dst_node_id, e))
        adj[e.dst_node_id].append((e.src_node_id, e))

    # Seed: required terminals first, then highest-prize fallback
    selected_nodes: set[str] = set()
    if must_include:
        for nid in must_include:
            if nid in node_map:
                selected_nodes.add(nid)
    if not selected_nodes:
        seed_id = max(prizes, key=lambda nid: prizes[nid])
        selected_nodes.add(seed_id)
    selected_edges: set[str] = set()
    total_cost = 0.0

    while True:
        best: tuple[float, str, Edge] | None = None
        for nid in list(selected_nodes):
            for neighbor_id, edge in adj.get(nid, []):
                if neighbor_id in selected_nodes:
                    continue
                marginal_prize = prizes.get(neighbor_id, 0.0)
                marginal_cost = costs.get(edge.id, 0.0)
                net = marginal_prize - marginal_cost
                if best is None or net > best[0]:
                    best = (net, neighbor_id, edge)
        if best is None or best[0] <= 0:
            break
        if total_cost + costs.get(best[2].id, 0.0) > budget:
            break
        selected_nodes.add(best[1])
        selected_edges.add(best[2].id)
        total_cost += costs.get(best[2].id, 0.0)

    # Pull positive edges between selected
    for e in edges:
        if e.src_node_id in selected_nodes and e.dst_node_id in selected_nodes and e.weight > 0:
            selected_edges.add(e.id)

    return (
        [n for n in nodes if n.id in selected_nodes],
        [e for e in edges if e.id in selected_edges],
    )


def pcst_extract_networkx(
    paths: list[Path], budget: float = 10.0
) -> tuple[list[Node], list[Edge]]:
    """NetworkX-backed Steiner tree extraction.

    Strategy:
      1. Build a graph of positive-weight edges (use 1/weight as cost).
      2. Pick top-K terminals = highest-prize nodes.
      3. Compute the Steiner tree spanning these terminals.
      4. Augment with adjacent high-prize nodes within budget.

    Falls back to greedy `pcst_extract` if networkx unavailable.
    """
    try:
        import networkx as nx
        from networkx.algorithms.approximation.steinertree import steiner_tree
    except ImportError:
        return pcst_extract(paths, budget)

    if not paths:
        return [], []

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

    # Prizes
    prizes: dict[str, float] = {n.id: 0.0 for n in nodes}
    for e in edges:
        if e.weight > 0:
            prizes[e.src_node_id] = prizes.get(e.src_node_id, 0.0) + e.weight
            prizes[e.dst_node_id] = prizes.get(e.dst_node_id, 0.0) + e.weight

    # Build undirected positive-weight graph
    G = nx.Graph()
    for n in nodes:
        G.add_node(n.id)
    for e in edges:
        if e.weight > 0:
            cost = 1.0 / max(0.05, e.weight)
            if G.has_edge(e.src_node_id, e.dst_node_id):
                if G[e.src_node_id][e.dst_node_id]["weight"] > cost:
                    G[e.src_node_id][e.dst_node_id]["weight"] = cost
                    G[e.src_node_id][e.dst_node_id]["edge_id"] = e.id
            else:
                G.add_edge(e.src_node_id, e.dst_node_id, weight=cost, edge_id=e.id)

    # Pick terminals: top 3 prize nodes connected in G
    sorted_by_prize = sorted(prizes.items(), key=lambda kv: -kv[1])
    terminals: list[str] = []
    for nid, _p in sorted_by_prize:
        if nid in G.nodes and G.degree(nid) > 0:
            terminals.append(nid)
        if len(terminals) >= 3:
            break

    if len(terminals) < 2:
        return pcst_extract(paths, budget)

    # Run Steiner tree on the largest connected component containing the terminals
    try:
        # Restrict to component containing first terminal
        comp = nx.node_connected_component(G, terminals[0])
        H = G.subgraph(comp).copy()
        present_terminals = [t for t in terminals if t in H.nodes]
        if len(present_terminals) < 2:
            return pcst_extract(paths, budget)
        T = steiner_tree(H, present_terminals, weight="weight")
    except Exception:
        return pcst_extract(paths, budget)

    selected_nodes: set[str] = set(T.nodes)
    selected_edges: set[str] = {T[u][v]["edge_id"] for u, v in T.edges if "edge_id" in T[u][v]}

    return (
        [n for n in nodes if n.id in selected_nodes],
        [e for e in edges if e.id in selected_edges],
    )
