"""Optimal Transport on memory graphs.

Two scales of comparison:

  - **Wasserstein** between two distributions of node embeddings: how much
    "embedding mass" needs to move to transform memory state A into B?
  - **Gromov-Wasserstein** between two graphs (with their structure): how
    similar are the *graphs themselves*, not just the embeddings?

Use cases in Recall:
  1. Measuring drift between consolidation runs (W(state_t, state_{t+1}))
  2. Cross-tenant or cross-agent memory similarity (GW between two memory
     graphs of different agents on the same domain)
  3. Detecting "memory regression" — sudden large W after a consolidation
     pass means the consolidator destroyed something it shouldn't have.

POT (python-optimal-transport) handles both with O(n^2 log n) Sinkhorn.
"""
from __future__ import annotations

import numpy as np

from recall.types import Edge, Node


def _has_pot() -> bool:
    try:
        import ot  # noqa: F401
        return True
    except ImportError:
        return False


def wasserstein_graph_distance(
    nodes_a: list[Node],
    nodes_b: list[Node],
    reg: float = 0.05,
    method: str = "sinkhorn",
) -> float:
    """Wasserstein-2 distance between the s-component clouds of two memories.

    Args:
        nodes_a, nodes_b: two sets of nodes (e.g. from two consolidation snapshots).
        reg: Sinkhorn regularization parameter.
        method: 'sinkhorn' (entropic, fast) or 'emd' (exact, slower).

    Returns:
        Wasserstein-2 distance (≥ 0). 0 means identical distributions.
    """
    Sa = _stack_s(nodes_a)
    Sb = _stack_s(nodes_b)
    if Sa is None or Sb is None or len(Sa) == 0 or len(Sb) == 0:
        return 0.0

    # Cost = squared Euclidean
    M = _pairwise_sq_dist(Sa, Sb)

    # Uniform marginals
    a = np.ones(len(Sa)) / len(Sa)
    b = np.ones(len(Sb)) / len(Sb)

    if not _has_pot():
        # Fallback: mean of nearest-neighbor squared distance (lower bound on W2)
        nn_ab = M.min(axis=1).mean()
        nn_ba = M.min(axis=0).mean()
        return float(0.5 * (nn_ab + nn_ba))

    import ot
    if method == "emd":
        return float(ot.emd2(a, b, M))
    return float(ot.sinkhorn2(a, b, M, reg=reg))


def gromov_wasserstein_distance(
    nodes_a: list[Node],
    edges_a: list[Edge],
    nodes_b: list[Node],
    edges_b: list[Edge],
    reg: float = 0.005,
    n_iter: int = 50,
) -> float:
    """Gromov-Wasserstein distance between two graphs.

    GW compares two metric spaces (here: two graphs with their internal
    distance structure). Unlike Wasserstein, doesn't need the embeddings to
    live in the same space.

    Args:
        Two (nodes, edges) pairs.
        reg: entropic regularization.
        n_iter: max Sinkhorn iterations.

    Returns:
        GW distance ≥ 0.
    """
    if not _has_pot():
        # Fallback: structural similarity (edge density gap)
        a_density = len(edges_a) / max(1, len(nodes_a))
        b_density = len(edges_b) / max(1, len(nodes_b))
        return float(abs(a_density - b_density))

    import ot

    Da = _intra_graph_distance(nodes_a, edges_a)
    Db = _intra_graph_distance(nodes_b, edges_b)
    if Da.shape[0] < 2 or Db.shape[0] < 2:
        return 0.0

    p = np.ones(Da.shape[0]) / Da.shape[0]
    q = np.ones(Db.shape[0]) / Db.shape[0]
    try:
        gw = ot.gromov.entropic_gromov_wasserstein2(
            Da, Db, p, q, "square_loss", epsilon=reg, max_iter=n_iter
        )
        return float(gw)
    except Exception:
        return float(abs(len(edges_a) / max(1, len(nodes_a))
                         - len(edges_b) / max(1, len(nodes_b))))


# --- Helpers ---


def _stack_s(nodes: list[Node]) -> np.ndarray | None:
    out = []
    for n in nodes:
        if n.f_embedding is None or n.b_embedding is None:
            continue
        out.append(0.5 * (n.f_embedding + n.b_embedding))
    if not out:
        return None
    return np.stack(out)


def _pairwise_sq_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = (A * A).sum(1)[:, None]
    bb = (B * B).sum(1)[None, :]
    ab = A @ B.T
    return aa + bb - 2.0 * ab


def _intra_graph_distance(nodes: list[Node], edges: list[Edge]) -> np.ndarray:
    """Shortest-path distance matrix on the (positive-weight) graph."""
    n = len(nodes)
    if n == 0:
        return np.zeros((0, 0))
    idx = {node.id: i for i, node in enumerate(nodes)}
    INF = 1e9
    D = np.full((n, n), INF, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    for e in edges:
        i = idx.get(e.src_node_id)
        j = idx.get(e.dst_node_id)
        if i is None or j is None:
            continue
        if e.weight <= 0:
            continue
        # treat 1/weight as edge length
        length = 1.0 / max(e.weight, 0.05)
        D[i, j] = min(D[i, j], length)
        D[j, i] = min(D[j, i], length)

    # Floyd-Warshall (small graphs)
    for k in range(n):
        D = np.minimum(D, D[:, k:k+1] + D[k:k+1, :])

    # Replace INF with twice the max finite distance
    finite = D[D < INF]
    cap = 2.0 * float(finite.max()) if len(finite) else 1.0
    D[D >= INF] = cap
    return D
