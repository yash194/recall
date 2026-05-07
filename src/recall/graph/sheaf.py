"""Cellular sheaf Laplacian for typed-edge inconsistency detection.

Per Hansen-Ghrist (2019) and Wei et al. arXiv 2501.19207 (closed-form
restriction-map learning, Jan 2025), the cellular sheaf Laplacian's
H¹(F) = ker(δ¹) / im(δ⁰) counts the number of *globally inconsistent* sections
on the graph — internal contradictions inference cannot remove.

For Recall's typed-edge memory, this gives:
  - **dim H¹ = 0** ⇒ memory is logically consistent (no frustrated cycles)
  - **dim H¹ > 0** ⇒ memory contains inconsistencies (e.g., cycle of supports +
    contradicts that can't be coherently labelled)

Why this matters: heuristic pairwise contradiction detection misses
*cycle-level* inconsistencies. If A supports B, B supports C, and C
contradicts A, no pairwise check catches the global frustration. H¹ catches
it.

Implementation: 1-d stalks at each node, with restriction maps assigned by
edge type:
    supports / agrees / temporal_next / pivots → +1
    contradicts / superseded                  → -1
    corrects                                   → trained / heuristic
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from recall.types import Edge, EdgeType, Node


# Restriction-map sign by edge type. For 1-d stalks, this is just the sign
# applied when transporting a section across the edge.
_EDGE_TYPE_SIGN: dict[str, float] = {
    "supports": 1.0,
    "agrees": 1.0,
    "temporal_next": 1.0,
    "pivots": 1.0,
    "contradicts": -1.0,
    "superseded": -1.0,
    "corrects": -0.5,  # half-strength flip
    "pending": 0.5,
}


def _et_sign(t) -> float:
    s = t.value if hasattr(t, "value") else str(t)
    return _EDGE_TYPE_SIGN.get(s, 0.5)


def signed_incidence(nodes: list[Node], edges: list[Edge]) -> tuple[scipy.sparse.csr_matrix, dict[str, int]]:
    """Build the signed incidence δ¹: 0-cochains → 1-cochains.

    For each (positive-weight) edge e = (u → v) with sign s_e:
        (δ¹ x)[e] = s_e · x[v] - x[u]

    Equivalently, signed incidence rows have +s_e at column v and -1 at column u.
    Returns the |E|×|V| matrix.
    """
    n = len(nodes)
    if n == 0:
        return scipy.sparse.csr_matrix((0, 0)), {}
    idx = {node.id: i for i, node in enumerate(nodes)}

    # Filter active typed edges
    active = [
        e for e in edges
        if e.deprecated_at is None and e.src_node_id in idx and e.dst_node_id in idx
    ]
    m = len(active)
    if m == 0:
        return scipy.sparse.csr_matrix((0, n)), idx

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for e_idx, e in enumerate(active):
        u = idx[e.src_node_id]
        v = idx[e.dst_node_id]
        s = _et_sign(e.edge_type)
        # row e_idx: -1 at u, +s at v
        rows.append(e_idx); cols.append(u); vals.append(-1.0)
        rows.append(e_idx); cols.append(v); vals.append(float(s))
    inc = scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(m, n), dtype=np.float64
    )
    return inc, idx


def sheaf_laplacian(nodes: list[Node], edges: list[Edge]) -> scipy.sparse.csr_matrix:
    """Sheaf Laplacian L_F = (δ¹)ᵀ δ¹ on 0-cochains."""
    inc, _ = signed_incidence(nodes, edges)
    if inc.shape[0] == 0:
        return scipy.sparse.csr_matrix((inc.shape[1], inc.shape[1]))
    return (inc.T @ inc).tocsr()


def harmonic_dimension(
    nodes: list[Node], edges: list[Edge], tol: float = 1e-6
) -> int:
    """dim H⁰(F) = dim ker(L_F) — the number of globally consistent sections.

    For a connected graph with consistent edge signs, this is 1.
    For a balanced signed graph (Cartwright-Harary 1956), this is 1 if
    consistent, otherwise 0.
    """
    L = sheaf_laplacian(nodes, edges)
    n = L.shape[0]
    if n == 0:
        return 0
    if n <= 32:
        evals = np.linalg.eigvalsh(L.toarray())
    else:
        try:
            k = min(n - 2, 16)
            evals = scipy.sparse.linalg.eigsh(
                L, k=k, which="SM", return_eigenvectors=False, maxiter=300
            )
        except Exception:
            evals = np.linalg.eigvalsh(L.toarray())
    return int(np.sum(np.abs(evals) < tol))


def inconsistency_score(nodes: list[Node], edges: list[Edge]) -> dict[str, float]:
    """Return diagnostic inconsistency metrics from the sheaf Laplacian.

    Returns:
        n_components_consistent: dim H⁰ = number of globally consistent regions
        smallest_eigenvalue: λ_min of L_F (closer to 0 = more consistent)
        is_globally_consistent: H⁰ = number of weakly-connected components of G
        frustration_score: ratio of inconsistent edges (heuristic upper bound)
    """
    L = sheaf_laplacian(nodes, edges)
    n = L.shape[0]
    if n == 0:
        return {
            "n_consistent_sections": 0,
            "smallest_eigenvalue": 0.0,
            "is_globally_consistent": True,
            "frustration_score": 0.0,
        }

    # Count weakly-connected components (using unsigned graph)
    if not edges:
        wcc = n
    else:
        adj = defaultdict(set)
        for e in edges:
            if e.deprecated_at is None:
                adj[e.src_node_id].add(e.dst_node_id)
                adj[e.dst_node_id].add(e.src_node_id)
        seen: set[str] = set()
        wcc = 0
        for node in nodes:
            if node.id in seen:
                continue
            stack = [node.id]
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                for v in adj[u]:
                    if v not in seen:
                        stack.append(v)
            wcc += 1

    h0 = harmonic_dimension(nodes, edges)

    if n <= 32:
        evals = np.linalg.eigvalsh(L.toarray())
    else:
        try:
            evals = scipy.sparse.linalg.eigsh(
                L, k=min(n - 2, 4), which="SM", return_eigenvectors=False, maxiter=300
            )
        except Exception:
            evals = np.linalg.eigvalsh(L.toarray())
    smallest = float(np.min(evals)) if len(evals) else 0.0

    # Heuristic frustration: fraction of negative-sign edges
    n_neg_edges = sum(
        1 for e in edges
        if e.deprecated_at is None and _et_sign(e.edge_type) < 0
    )
    n_active = sum(1 for e in edges if e.deprecated_at is None)
    frustration = (n_neg_edges / max(n_active, 1)) if n_active > 0 else 0.0

    return {
        "n_consistent_sections": int(h0),
        "smallest_eigenvalue": smallest,
        "is_globally_consistent": bool(h0 == wcc),
        "frustration_score": frustration,
        "n_components": wcc,
    }
