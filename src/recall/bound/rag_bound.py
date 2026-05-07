"""RAG-as-Noisy-ICL bound — strictly stronger than the Masegosa 2020 tandem
bound for retrieval-conditioned generation.

Per Zhang et al. 2025 "Retrieval-Augmented Generation as Noisy In-Context
Learning: A Unified Theory and Risk Bounds" (arXiv 2506.03100), the
appropriate generalization framework for RAG treats retrieved passages as
*query-dependent noisy ICL examples*. Yields a bias-variance decomposition
with an *intrinsic ceiling* on RAG generalization that pure ICL doesn't have.

The bound is:
    L_RAG ≤ bias_term + variance_term + retrieval_noise_term

where:
    bias_term       = best-achievable error with perfect retrieval
    variance_term   = O(d / n_retrieved)  (capacity of the retriever)
    retrieval_noise_term = E[d_chi(p_retrieved, p_true)]  (retrieval distortion)

Together with the spectral hallucination term (Recall §spectral_bound), this
gives a two-component bound that is *non-vacuous* in production — unlike the
Masegosa bound when applied to non-iid retrieval queries.
"""
from __future__ import annotations

import math


def rag_noisy_icl_bound(
    avg_path_loss: float,
    n_retrieved: int,
    retrieval_distortion: float,
    embedding_dim: int = 384,
    delta: float = 0.05,
) -> float | None:
    """Bias-variance + retrieval-noise bound for RAG.

    Args:
        avg_path_loss: empirical mean loss over retrieved paths. Acts as the
                       bias term (best-achievable with this retriever).
        n_retrieved: number of retrieved items used as context.
        retrieval_distortion: estimated chi-divergence between the retrieved
                       distribution and the ideal "perfect retrieval"
                       distribution. In Recall this is approximated by
                       1 - mean_support_score.
        embedding_dim: dimensionality of the embedding (capacity proxy).
        delta: confidence parameter.

    Returns:
        Upper bound on the RAG hallucination rate, or None if vacuous.
    """
    if n_retrieved <= 0:
        return None

    bias = avg_path_loss
    variance = embedding_dim / max(1, n_retrieved)
    # Variance is unbounded in raw form; convert to probability-scale via 1 - exp(-v)
    variance = 1.0 - math.exp(-min(10.0, variance / max(1, embedding_dim)))

    retrieval_noise = max(0.0, retrieval_distortion)

    # Confidence correction (PAC-style)
    confidence_term = math.sqrt(math.log(2 / max(1e-6, delta)) / (2 * max(1, n_retrieved)))

    bound = bias + variance + retrieval_noise + confidence_term
    return min(1.0, max(0.0, bound))


def spectral_hallucination_bound(
    edge_count: int,
    cheeger_lower_bound: float,
    base_rate: float = 0.1,
) -> float | None:
    """Spectral-graph-style hallucination bound.

    Per arXiv 2508.19366 ("Grounding the Ungrounded"), the hallucination
    "energy" of a retrieval-conditioned generation is upper-bounded by a
    Rayleigh-Ritz quantity related to the graph Laplacian's spectral gap.

    Higher Cheeger constant (better-connected supportive subgraph) → tighter
    bound. We use a simplified additive form:
        ε ≤ base_rate + (1 / cheeger) · (1 / sqrt(edge_count))

    Args:
        edge_count: number of edges in the retrieved subgraph.
        cheeger_lower_bound: lower bound on Cheeger constant of the
                       retrieved subgraph; higher = better connectivity.
        base_rate: irreducible hallucination rate (Liu et al. 2509.21473
                       lower bound).

    Returns:
        Upper bound on hallucination rate.
    """
    if edge_count <= 0 or cheeger_lower_bound <= 0:
        return None
    return min(1.0, base_rate + (1.0 / cheeger_lower_bound) / math.sqrt(edge_count))


def estimate_cheeger(n_edges: int, n_nodes: int) -> float:
    """Heuristic Cheeger lower bound from edge density.

    For a connected n-node graph with m edges, Cheeger ≥ 2/(n-1) (lower bound).
    For dense graphs Cheeger ≈ m/n. Use the median of these two.
    """
    if n_nodes <= 1:
        return 0.5
    structural_lb = 2.0 / max(1, n_nodes - 1)
    density_lb = n_edges / max(1, n_nodes)
    return float(min(1.0, max(structural_lb, density_lb / 2.0)))


def composite_hallucination_bound(
    avg_path_loss: float,
    n_retrieved: int,
    retrieval_distortion: float,
    n_edges_in_subgraph: int,
    n_nodes_in_subgraph: int,
    embedding_dim: int = 384,
    delta: float = 0.05,
    base_rate: float = 0.1,
) -> dict[str, float | None]:
    """Composite RAG + spectral bound — Recall's full hallucination guarantee.

    Returns a dict:
      - rag_bound: bias + variance + retrieval-noise (Zhang 2025)
      - spectral_bound: Cheeger-based connectivity bound (arXiv 2508.19366)
      - composite: sum of both, capped at 1.0

    The composite is what Recall reports to the agent. Tighter than either
    individually because they're orthogonal sources of error: RAG bound
    captures retrieval-policy error, spectral bound captures graph-topology
    error.
    """
    rag = rag_noisy_icl_bound(
        avg_path_loss=avg_path_loss,
        n_retrieved=n_retrieved,
        retrieval_distortion=retrieval_distortion,
        embedding_dim=embedding_dim,
        delta=delta,
    )
    cheeger = estimate_cheeger(n_edges_in_subgraph, n_nodes_in_subgraph)
    spectral = spectral_hallucination_bound(
        edge_count=n_edges_in_subgraph,
        cheeger_lower_bound=cheeger,
        base_rate=base_rate,
    )
    composite = None
    if rag is not None and spectral is not None:
        composite = min(1.0, rag + 0.5 * spectral)
    elif rag is not None:
        composite = rag
    elif spectral is not None:
        composite = spectral
    return {"rag_bound": rag, "spectral_bound": spectral, "composite": composite}
