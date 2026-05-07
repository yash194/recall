"""Spectral graph theory primitives for the memory graph.

Implements:
  - graph_laplacian:        L = D - A  (or normalized variant)
  - laplacian_eigenvalues:  full or top-k via scipy.sparse.linalg
  - spectral_gap:           λ_2 - λ_1 (algebraic connectivity)
  - cheeger_constant:       isoperimetric proxy via spectral gap (Cheeger inequality)
  - heat_kernel_signature:  exp(-tL) acting on indicator vector → diffusion
  - personalized_pagerank:  signed-weight aware PageRank

The spectral gap λ_2 controls the Cheeger constant h(G):
    λ_2 / 2  ≤  h(G)  ≤  √(2 λ_2)

We use λ_2 to estimate connectivity for the spectral hallucination bound.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from recall.types import Edge, Node


def _build_undirected_adjacency(
    nodes: list[Node], edges: list[Edge], use_abs_weight: bool = True
) -> tuple[scipy.sparse.csr_matrix, dict[str, int]]:
    """Build a symmetric weighted adjacency matrix (n × n)."""
    n = len(nodes)
    if n == 0:
        return scipy.sparse.csr_matrix((0, 0)), {}
    idx = {node.id: i for i, node in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for e in edges:
        i = idx.get(e.src_node_id)
        j = idx.get(e.dst_node_id)
        if i is None or j is None:
            continue
        w = abs(e.weight) if use_abs_weight else e.weight
        rows.append(i); cols.append(j); vals.append(w)
        rows.append(j); cols.append(i); vals.append(w)  # symmetric
    A = scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n, n), dtype=np.float64
    )
    return A, idx


def graph_laplacian(
    nodes: list[Node], edges: list[Edge], normalized: bool = True
) -> scipy.sparse.csr_matrix:
    """Compute the (optionally normalized) graph Laplacian.

    Args:
        nodes, edges: the memory subgraph.
        normalized: if True, returns L_sym = I - D^{-1/2} A D^{-1/2}.
                    if False, returns L = D - A.

    Returns:
        Sparse Laplacian (n × n).
    """
    A, _ = _build_undirected_adjacency(nodes, edges)
    if A.shape[0] == 0:
        return scipy.sparse.csr_matrix((0, 0))
    deg = np.array(A.sum(axis=1)).ravel()
    if not normalized:
        D = scipy.sparse.diags(deg)
        return (D - A).tocsr()
    # L_sym = I - D^{-1/2} A D^{-1/2}
    deg_safe = np.where(deg > 0, deg, 1.0)
    d_inv_sqrt = scipy.sparse.diags(1.0 / np.sqrt(deg_safe))
    L = scipy.sparse.eye(A.shape[0]) - d_inv_sqrt @ A @ d_inv_sqrt
    return L.tocsr()


def laplacian_eigenvalues(
    nodes: list[Node], edges: list[Edge], k: int = 6, normalized: bool = True
) -> np.ndarray:
    """Compute the smallest k Laplacian eigenvalues.

    These tell you about graph connectivity:
      - λ_1 = 0 always (constant eigenvector)
      - λ_2 = "spectral gap" / algebraic connectivity
      - Multiplicity of 0 = number of connected components

    Returns:
        Array of k smallest eigenvalues, sorted ascending.
    """
    L = graph_laplacian(nodes, edges, normalized=normalized)
    n = L.shape[0]
    if n == 0:
        return np.array([])
    if n <= k:
        # dense small problem
        evals = np.linalg.eigvalsh(L.toarray())
        return np.sort(np.real(evals))
    try:
        # smallest k via shift-invert
        evals = scipy.sparse.linalg.eigsh(
            L, k=min(k, n - 2), which="SM", return_eigenvectors=False, maxiter=300
        )
        return np.sort(np.real(evals))
    except Exception:
        # Fallback to dense on small graphs
        evals = np.linalg.eigvalsh(L.toarray())
        return np.sort(np.real(evals))[:k]


def spectral_gap(nodes: list[Node], edges: list[Edge]) -> float:
    """λ_2 of the normalized Laplacian — algebraic connectivity.

    Higher λ_2 = better connected = retrieval paths are more reliable.
    """
    evals = laplacian_eigenvalues(nodes, edges, k=4, normalized=True)
    if len(evals) < 2:
        return 0.0
    return float(evals[1])  # second-smallest


def cheeger_constant(nodes: list[Node], edges: list[Edge]) -> tuple[float, float]:
    """Returns (lower_bound, upper_bound) on the Cheeger constant via the
    Cheeger inequality:

        λ_2 / 2  ≤  h(G)  ≤  √(2 λ_2)

    The Cheeger constant measures how easy it is to "cut" the graph into two
    components — high h(G) = well-connected, hard to fragment.
    """
    lam2 = spectral_gap(nodes, edges)
    if lam2 <= 0:
        return (0.0, 0.0)
    return (lam2 / 2.0, float(np.sqrt(2.0 * lam2)))


def heat_kernel_signature(
    nodes: list[Node], edges: list[Edge], times: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Heat kernel diagonal: HKS_i(t) = sum_k exp(-t λ_k) v_k[i]^2.

    Captures multi-scale neighborhood structure of node i. Useful for
    motif detection at different scales.

    Args:
        times: array of diffusion times; default = [0.1, 1.0, 10.0].

    Returns:
        Dict mapping node_id → array of len(times) HKS values.
    """
    if times is None:
        times = np.array([0.1, 1.0, 10.0])
    L = graph_laplacian(nodes, edges, normalized=True)
    n = L.shape[0]
    out: dict[str, np.ndarray] = {}
    if n == 0:
        return out
    if n <= 32:
        evals, evecs = np.linalg.eigh(L.toarray())
    else:
        try:
            k = min(n - 2, 32)
            evals, evecs = scipy.sparse.linalg.eigsh(
                L, k=k, which="SM", maxiter=300
            )
        except Exception:
            evals, evecs = np.linalg.eigh(L.toarray())

    # HKS_i(t) = Σ_k exp(-t·λ_k) · v_k[i]^2
    evec_sq = evecs ** 2  # (n × k)
    for ni, node in enumerate(nodes):
        hks = np.zeros(len(times))
        for ti, t in enumerate(times):
            hks[ti] = float(np.sum(np.exp(-t * evals) * evec_sq[ni, :]))
        out[node.id] = hks
    return out


def personalized_pagerank(
    nodes: list[Node],
    edges: list[Edge],
    seed_node_ids: list[str],
    damping: float = 0.85,
    n_iter: int = 50,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Signed-weight aware Personalized PageRank.

    For positive-weight edges, propagates rank forward.
    For negative-weight edges (contradicts/superseded), subtracts rank.

    Args:
        seed_node_ids: nodes to teleport to (the personalization vector).
        damping: standard PageRank damping (0.85).

    Returns:
        Dict node_id → PPR score (can be negative for nodes blocked by contradicts).
    """
    n = len(nodes)
    if n == 0 or not seed_node_ids:
        return {}
    idx = {node.id: i for i, node in enumerate(nodes)}

    # Build directed weighted adjacency, signed
    A = scipy.sparse.lil_matrix((n, n), dtype=np.float64)
    for e in edges:
        i = idx.get(e.src_node_id)
        j = idx.get(e.dst_node_id)
        if i is None or j is None:
            continue
        A[i, j] = e.weight
    # Row-normalize positive part
    row_sums = np.array(np.abs(A).sum(axis=1)).ravel()
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    A_norm = scipy.sparse.diags(1.0 / row_sums) @ A.tocsr()

    # Personalization vector
    p = np.zeros(n)
    for sid in seed_node_ids:
        if sid in idx:
            p[idx[sid]] = 1.0
    if p.sum() == 0:
        return {}
    p = p / p.sum()

    r = p.copy()
    for _ in range(n_iter):
        r_new = damping * (A_norm.T @ r) + (1 - damping) * p
        if np.linalg.norm(r_new - r, 1) < tol:
            r = r_new
            break
        r = r_new

    return {node.id: float(r[i]) for node, i in zip(nodes, [idx[n.id] for n in nodes])}


def graph_health(nodes: list[Node], edges: list[Edge]) -> dict[str, float]:
    """One-shot summary of spectral graph health."""
    if not nodes:
        return {"n_nodes": 0, "n_edges": 0}
    lam2 = spectral_gap(nodes, edges)
    cheeger_low, cheeger_up = cheeger_constant(nodes, edges)
    return {
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "spectral_gap_lambda2": lam2,
        "cheeger_lower_bound": cheeger_low,
        "cheeger_upper_bound": cheeger_up,
        "edge_density": len(edges) / max(1, len(nodes)),
    }
