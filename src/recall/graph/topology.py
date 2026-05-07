"""Persistent homology of the memory graph.

Uses gudhi to compute Vietoris-Rips persistence diagrams from node
embeddings. Persistence captures the topology of the embedding cloud:
  - β_0 = number of connected components
  - β_1 = number of independent loops (cycles)
  - persistence intervals = "lifetimes" of features

For a memory substrate, this is useful as a SUMMARY signature:
  - long-persistence loops in β_1 = stable reasoning structures
  - many short-lived components = noisy / un-consolidated memory
  - one persistent component = well-consolidated
  - missing expected loops = "missing reasoning" detection signal

We compute this on the s-component (semantic side) of the embeddings.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from recall.types import Node


def _has_gudhi() -> bool:
    try:
        import gudhi  # noqa: F401
        return True
    except ImportError:
        return False


def persistent_homology_summary(
    nodes: list[Node],
    max_dimension: int = 1,
    max_edge_length: float = 1.0,
) -> dict[str, Any]:
    """Compute persistent-homology summary statistics over the s-embeddings.

    Args:
        nodes: memory nodes with f, b embeddings.
        max_dimension: highest homology dim to compute (1 = up to loops).
        max_edge_length: Vietoris-Rips threshold (in cosine-distance space).

    Returns:
        Dict with keys:
          n_nodes
          backend            ('gudhi' or 'numpy_fallback')
          betti_0            β_0 at full filtration (final component count)
          betti_1            β_1 at full filtration (final loop count)
          mean_persistence_dim0/1
          max_persistence_dim0/1
          total_persistence_dim0/1
    """
    if not nodes:
        return {"n_nodes": 0}

    # Build s-component matrix and use 1-cosine as distance
    s_vecs = []
    for n in nodes:
        if n.f_embedding is None or n.b_embedding is None:
            continue
        s = 0.5 * (n.f_embedding + n.b_embedding)
        nrm = float(np.linalg.norm(s))
        if nrm > 1e-12:
            s = s / nrm
        s_vecs.append(s)

    if len(s_vecs) < 3:
        return {"n_nodes": len(nodes), "backend": "insufficient_data"}

    S = np.stack(s_vecs)  # (n × d)
    # Pairwise cosine distance: 1 - S S^T (S already row-normalized)
    sim = S @ S.T
    dist = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)

    if _has_gudhi():
        return _gudhi_persistence(dist, max_dimension, max_edge_length)
    return _numpy_persistence(dist, max_edge_length)


def _gudhi_persistence(
    dist: np.ndarray, max_dimension: int, max_edge_length: float
) -> dict[str, Any]:
    import gudhi

    rips = gudhi.RipsComplex(distance_matrix=dist, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=max_dimension + 1)
    diag = st.persistence()

    by_dim: dict[int, list[tuple[float, float]]] = {0: [], 1: []}
    for dim, (birth, death) in diag:
        if dim in (0, 1):
            d = float("inf") if death == float("inf") else float(death)
            by_dim[dim].append((float(birth), d))

    def stats(intervals: list[tuple[float, float]], filt: float):
        # Replace inf with the filtration threshold for finite stats
        durations = []
        alive_at_end = 0
        for b, d in intervals:
            if d == float("inf") or d > filt:
                alive_at_end += 1
                durations.append(filt - b)
            else:
                durations.append(d - b)
        if not durations:
            return 0.0, 0.0, 0.0, alive_at_end
        return (float(np.mean(durations)),
                float(np.max(durations)),
                float(np.sum(durations)),
                alive_at_end)

    mean0, max0, total0, b0_alive = stats(by_dim[0], max_edge_length)
    mean1, max1, total1, b1_alive = stats(by_dim[1], max_edge_length)

    return {
        "n_nodes": dist.shape[0],
        "backend": "gudhi",
        "betti_0": b0_alive,
        "betti_1": b1_alive,
        "mean_persistence_dim0": mean0,
        "max_persistence_dim0": max0,
        "total_persistence_dim0": total0,
        "mean_persistence_dim1": mean1,
        "max_persistence_dim1": max1,
        "total_persistence_dim1": total1,
        "n_intervals_dim0": len(by_dim[0]),
        "n_intervals_dim1": len(by_dim[1]),
    }


def _numpy_persistence(dist: np.ndarray, max_edge_length: float) -> dict[str, Any]:
    """Fallback: numpy-only β_0 trace using a union-find at increasing edge thresholds.

    Doesn't compute β_1 (would need the full simplicial complex). Suitable for
    detecting connectivity but not loops.
    """
    n = dist.shape[0]
    # Get sorted edge list
    iu, ju = np.triu_indices(n, k=1)
    weights = dist[iu, ju]
    order = np.argsort(weights)
    iu = iu[order]
    ju = ju[order]
    weights = weights[order]

    # Union-find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    persistence_intervals = []  # (birth, death)
    n_components = n
    for i, j, w in zip(iu, ju, weights):
        if w > max_edge_length:
            break
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        # smaller-component dies, gets recorded
        persistence_intervals.append((0.0, float(w)))
        parent[ri] = rj
        n_components -= 1

    if persistence_intervals:
        durations = [d - b for b, d in persistence_intervals]
        mean0 = float(np.mean(durations))
        max0 = float(np.max(durations))
        total0 = float(np.sum(durations))
    else:
        mean0 = max0 = total0 = 0.0

    return {
        "n_nodes": n,
        "backend": "numpy_fallback",
        "betti_0": n_components,
        "betti_1": None,
        "mean_persistence_dim0": mean0,
        "max_persistence_dim0": max0,
        "total_persistence_dim0": total0,
    }


def topological_signature(nodes: list[Node]) -> np.ndarray:
    """Return a fixed-size topological signature vector for the memory state.

    Useful for comparing memory across time (does the memory have stable
    topology between consolidation runs?).
    """
    summary = persistent_homology_summary(nodes)
    keys = [
        "betti_0", "betti_1",
        "mean_persistence_dim0", "max_persistence_dim0",
        "mean_persistence_dim1", "max_persistence_dim1",
    ]
    sig = []
    for k in keys:
        v = summary.get(k)
        sig.append(float(v) if v is not None else 0.0)
    return np.array(sig, dtype=np.float32)
