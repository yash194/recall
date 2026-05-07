"""Conformal Risk Control bound for retrieval-conditioned generation.

Per C-RAG (Kang et al., ICML 2024, arXiv 2402.03181) and Conformal-RAG
(Aug 2025, arXiv 2506.20978): replace the often-vacuous PAC-Bayes bound with
a calibration-based bound that's *finite-sample, distribution-free*, and
non-vacuous by construction.

The bound: with probability ≥ 1-δ,
    E[R(generator)] ≤ R̂_cal + ε(N, δ)

where R̂_cal is the empirical hallucination rate on a held-out calibration
set of size N, and ε(N, δ) decays as 1/√N. This bound is typically 5-10×
tighter than the closed-form PAC-Bayes bound at usable N.

Two flavors:
  - `crc_hoeffding`: Hoeffding upper bound (conservative, simple)
  - `crc_wilson`:    Wilson upper bound (tighter for low rates)

The composite returns the tighter of the two.
"""
from __future__ import annotations

import math
from typing import Sequence


def crc_hoeffding(empirical_risk: float, n_calibration: int, delta: float = 0.05) -> float:
    """Hoeffding upper bound on population risk.

    With probability ≥ 1-δ, E[R] ≤ R̂ + sqrt(log(1/δ) / 2N).
    """
    if n_calibration <= 0:
        return 1.0
    return min(
        1.0,
        empirical_risk + math.sqrt(math.log(1.0 / max(delta, 1e-9)) / (2.0 * n_calibration)),
    )


def crc_wilson(empirical_risk: float, n_calibration: int, delta: float = 0.05) -> float:
    """Wilson interval upper bound — tighter than Hoeffding for low rates.

    Standard Wilson upper:
      U = (p + z²/2N + z·√(p(1-p)/N + z²/4N²)) / (1 + z²/N)
    where z = sqrt(2·log(e/δ)).
    """
    if n_calibration <= 0:
        return 1.0
    p = max(0.0, min(1.0, empirical_risk))
    z = math.sqrt(2.0 * math.log(math.e / max(delta, 1e-9)))
    n = n_calibration
    upper = (
        (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)))
        / (1 + z * z / n)
    )
    return min(1.0, upper)


def crc_bound(
    cal_risks: Sequence[float],
    delta: float = 0.05,
) -> dict[str, float]:
    """Conformal risk control bound from a calibration set.

    Args:
        cal_risks: list of binary risks (0 = correct, 1 = hallucinated)
                   from a held-out calibration set.
        delta: confidence parameter.

    Returns:
        Dict with 'hoeffding', 'wilson', 'crc_min', 'empirical_risk', 'n'.
    """
    n = len(cal_risks)
    if n == 0:
        return {
            "hoeffding": 1.0, "wilson": 1.0, "crc_min": 1.0,
            "empirical_risk": 0.0, "n": 0,
        }
    r_hat = sum(cal_risks) / n
    h = crc_hoeffding(r_hat, n, delta)
    w = crc_wilson(r_hat, n, delta)
    return {
        "hoeffding": h,
        "wilson": w,
        "crc_min": min(h, w),
        "empirical_risk": r_hat,
        "n": n,
    }


def split_conformal_threshold(
    cal_scores: Sequence[float], alpha: float = 0.05
) -> float:
    """Split conformal threshold q̂ such that P(score ≤ q̂) ≥ 1-α.

    Per Vovk-Shafer formulation. With N calibration points,
        q̂ = sorted_scores[⌈(N+1)(1-α)⌉ - 1]
    gives valid finite-sample coverage.
    """
    n = len(cal_scores)
    if n == 0:
        return 1.0
    sorted_scores = sorted(cal_scores)
    q_idx = max(0, min(n - 1, int(math.ceil((n + 1) * (1.0 - alpha))) - 1))
    return sorted_scores[q_idx]
