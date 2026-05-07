"""PMED experience scoring — Yash's original PMED-Draft framework.

Implements per-trajectory / per-region experience score:

  S(τ) = α·D_RPD − β·DCR − γ·P_syco + δ·Q_corr + η·Q_eff + κ·Q_rare

Where:
  D_RPD  = mean Reasoning Path Divergence (debate disagreement)
  DCR    = Disagreement Collapse Rate (fraction of trajectory that converged)
  P_syco = Sycophancy Penalty (agreement without evidence)
  Q_corr = Correction Depth × confidence delta
  Q_eff  = Debate Uplift (debate vs solo - cost penalty)
  Q_rare = Inverse-frequency rarity boost

Recall integrates this in two places:
  1. **Region pruning at sleep-time** — regions with low S(τ) get pruned harder.
  2. **Edge-weight reweighting** — edges in high-rarity, high-correction regions
     get boost factor in retrieval scoring.

This converts PMED from a research framework into a production knob.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from recall.types import Edge, Node


@dataclass(slots=True)
class PMEDComponents:
    """Decomposed PMED score components for a region/trajectory."""

    D_RPD: float = 0.0  # mean reasoning path divergence
    DCR: float = 0.0    # disagreement collapse rate
    P_syco: float = 0.0  # sycophancy penalty
    Q_corr: float = 0.0  # correction depth × confidence delta
    Q_eff: float = 0.0   # debate uplift
    Q_rare: float = 0.0  # rarity boost

    def total(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        delta: float = 1.0,
        eta: float = 0.5,
        kappa: float = 0.5,
    ) -> float:
        return (
            alpha * self.D_RPD
            - beta * self.DCR
            - gamma * self.P_syco
            + delta * self.Q_corr
            + eta * self.Q_eff
            + kappa * self.Q_rare
        )


def _embedding_cos(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def reasoning_path_divergence(nodes: list[Node]) -> float:
    """D_RPD = mean pairwise (1 - cos) over consecutive nodes' s-components.

    High D_RPD means the trajectory explored disparate ideas — useful experience.
    """
    if len(nodes) < 2:
        return 0.0
    s_vecs = []
    for n in nodes:
        if n.f_embedding is not None and n.b_embedding is not None:
            s_vecs.append(0.5 * (n.f_embedding + n.b_embedding))
    if len(s_vecs) < 2:
        return 0.0
    div = 0.0
    cnt = 0
    for i in range(len(s_vecs) - 1):
        div += 1.0 - _embedding_cos(s_vecs[i], s_vecs[i + 1])
        cnt += 1
    return float(div / cnt) if cnt else 0.0


def disagreement_collapse_rate(nodes: list[Node], eps: float = 0.05) -> float:
    """DCR = fraction of trajectory that lies in a low-divergence regime.

    High DCR = trajectory collapsed quickly into agreement; low value for the
    whole trajectory after the collapse point.
    """
    if len(nodes) < 3:
        return 0.0
    s_vecs = [
        0.5 * (n.f_embedding + n.b_embedding)
        for n in nodes
        if n.f_embedding is not None and n.b_embedding is not None
    ]
    if len(s_vecs) < 3:
        return 0.0
    # Find first index t* where divergence drops below eps
    t_star = None
    for i in range(len(s_vecs) - 1):
        d = 1.0 - _embedding_cos(s_vecs[i], s_vecs[i + 1])
        if d < eps:
            t_star = i
            break
    if t_star is None:
        return 0.0
    return float((len(s_vecs) - t_star) / len(s_vecs))


def sycophancy_penalty(nodes: list[Node], edges: list[Edge], lam: float = 0.85) -> float:
    """P_syco = agreement without evidence.

    For each `agrees` edge between consecutive nodes, increment penalty if
    cosine(s_i, s_{i-1}) > lam (very similar) and the source node lacks
    distinguishing role tags (no specific evidence).
    """
    if not edges:
        return 0.0
    by_id = {n.id: n for n in nodes}
    score = 0.0
    cnt = 0
    for e in edges:
        if e.edge_type.value != "agrees":
            continue
        src = by_id.get(e.src_node_id)
        dst = by_id.get(e.dst_node_id)
        if src is None or dst is None:
            continue
        if src.f_embedding is None or src.b_embedding is None:
            continue
        s_src = 0.5 * (src.f_embedding + src.b_embedding)
        s_dst = 0.5 * (dst.f_embedding + dst.b_embedding)
        if _embedding_cos(s_src, s_dst) > lam and (src.role or "") in {"", "fact"}:
            score += 1.0
        cnt += 1
    return float(score / cnt) if cnt else 0.0


def correction_depth(nodes: list[Node], edges: list[Edge]) -> float:
    """Q_corr = (C_T − C_0) · Δ_pivot.

    Δ_pivot = 1 - cos(s_first, s_last); confidence delta proxied by quality
    score difference. Weight by presence of `corrects` or `pivots` edges.
    """
    if len(nodes) < 2:
        return 0.0
    first = nodes[0]
    last = nodes[-1]
    if (first.f_embedding is None or first.b_embedding is None or
            last.f_embedding is None or last.b_embedding is None):
        return 0.0
    s_first = 0.5 * (first.f_embedding + first.b_embedding)
    s_last = 0.5 * (last.f_embedding + last.b_embedding)
    pivot = 1.0 - _embedding_cos(s_first, s_last)
    confidence_delta = (last.quality_score - first.quality_score)
    has_pivot = any(
        e.edge_type.value in ("pivots", "corrects") for e in edges
    )
    boost = 1.5 if has_pivot else 1.0
    return float(boost * pivot * (0.5 + confidence_delta))


def debate_uplift(nodes: list[Node], edges: list[Edge]) -> float:
    """Q_eff = debate uplift, penalized by length.

    Proxy: edge density in the region (more debate = more cross-references)
    minus length penalty.
    """
    n = max(1, len(nodes))
    e = len(edges)
    density = e / n
    length_penalty = 0.05 * math.log(1 + n)
    return float(density - length_penalty)


def rarity_boost(nodes: list[Node], all_nodes_count: int) -> float:
    """Q_rare = 1 / (1 + f(z)) where f is the cluster frequency.

    Approximation: rarity ≈ 1 / (1 + |region| / total_nodes^0.5).
    Smaller regions in a large memory get higher Q_rare.
    """
    if all_nodes_count <= 0:
        return 0.0
    f = len(nodes) / math.sqrt(max(1, all_nodes_count))
    return float(1.0 / (1.0 + f))


def compute_pmed_components(
    nodes: list[Node], edges: list[Edge], total_nodes_in_memory: int
) -> PMEDComponents:
    """Compute all six PMED components for a region/trajectory."""
    return PMEDComponents(
        D_RPD=reasoning_path_divergence(nodes),
        DCR=disagreement_collapse_rate(nodes),
        P_syco=sycophancy_penalty(nodes, edges),
        Q_corr=correction_depth(nodes, edges),
        Q_eff=debate_uplift(nodes, edges),
        Q_rare=rarity_boost(nodes, total_nodes_in_memory),
    )


def pmed_priority(
    components: PMEDComponents,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    eta: float = 0.5,
    kappa: float = 0.5,
) -> float:
    """Priority signal for sleep-time consolidation.

    Higher score → keep / reinforce. Lower score → prune harder.
    """
    return components.total(alpha, beta, gamma, delta, eta, kappa)
