"""PAC-Bayes structural hallucination bound.

Implements MATH.md §3 — the second-order tandem-loss bound from
Masegosa-Lorenzen-Igel-Seldin (NeurIPS 2020) and the Chebyshev-Cantelli-Bennett
refinement (Wu-Masegosa-Lorenzen-Igel-Seldin NeurIPS 2021).

The bound: with probability ≥ 1-δ over the training corpus,
  L_MV(π_Γ) ≤ ((m̂ + r̂) / (1 - m̂ + r̂))²
where:
  m̂ = 2 · E_{h,h'~π_Γ²}[L̂(h) - L̂_T(h, h')]   (margin)
  r̂ = √(2(KL(π_Γ || π_uniform) + ln(2√n/δ)) / n)  (PAC-Bayes residual)
"""
from __future__ import annotations

import math


def chebyshev_cantelli_bound(
    L_hat: float, L_T_hat: float, kl: float, n: int, delta: float = 0.05
) -> float | None:
    """Eqn (3.2) of MATH.md — Chebyshev-Cantelli refinement.

    Args:
        L_hat: mean empirical loss E[L̂(h)] over the posterior π_Γ.
        L_T_hat: mean empirical tandem loss E[L̂_T(h, h')] over π_Γ × π_Γ.
        kl: KL(π_Γ || π_uniform).
        n: training sample size.
        delta: confidence parameter.

    Returns:
        Upper bound on L_MV(π_Γ), or None if vacuous (bound ≥ 1).
    """
    if n <= 0:
        return None
    margin = 2.0 * (L_hat - L_T_hat)
    if margin <= 0:
        return None  # tandem loss too high — no margin, bound vacuous
    residual = math.sqrt(2.0 * (kl + math.log(2.0 * math.sqrt(n) / delta)) / n)
    if margin - residual <= 0:
        return None  # vacuous
    if 1.0 - margin + residual <= 0:
        return None
    bound = ((margin + residual) / (1.0 - margin + residual)) ** 2
    return min(1.0, max(0.0, bound))


def pac_bayes_bound(
    L_hat: float, L_T_hat: float, kl: float, n: int, delta: float = 0.05
) -> float:
    """Eqn (3.1) of MATH.md — first-order tandem bound.

    Returns the upper bound, possibly ≥ 1 (vacuous).
    """
    if n <= 0:
        return 1.0
    return min(
        1.0,
        4.0 * (L_T_hat + (kl + math.log(2.0 * math.sqrt(n) / delta)) / n),
    )


def is_bound_vacuous(bound: float | None) -> bool:
    return bound is None or bound >= 1.0


def compute_bound_estimate(
    *,
    n_paths: int,
    avg_path_loss: float,
    avg_tandem_loss: float,
    n_training_samples: int,
    delta: float = 0.05,
) -> float | None:
    """High-level bound computation given retrieval statistics.

    Approximation:
      KL(π_Γ || π_uniform) ≈ ln(n_paths) when π_Γ is concentrated on one path
                          ≈ 0 when π_Γ is uniform
      In practice we use a heuristic KL = max(0, ln(n_paths) - H_pi_gamma)
      where H_pi_gamma is the empirical entropy.

    For v1, we use a conservative upper estimate KL ≈ ln(max(1, n_paths)).
    """
    if n_paths <= 0:
        return None
    kl = math.log(max(1, n_paths))
    return chebyshev_cantelli_bound(
        L_hat=avg_path_loss,
        L_T_hat=avg_tandem_loss,
        kl=kl,
        n=n_training_samples,
        delta=delta,
    )
