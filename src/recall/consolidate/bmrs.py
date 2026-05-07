"""BMRS — Bayesian Model Reduction for Structured Pruning.

v0.7: corrected math. The previous closed-form formula had a sign error
(pruning condition was inverted) and used a fixed `s_squared=1.0` which made
EVERY non-zero-weight edge satisfy the prune condition with default config,
nuking 100% of edges in one consolidate pass.

The correct closed-form Bayes factor for "reduced (edge absent)" vs "full
(edge present)" under a Gaussian prior N(0, σ₀²) on the weight and a
Gaussian posterior N(μ, s²) is the standard Bayesian Model Reduction
result:

    log BF(reduced/full) = ½ log(σ₀²/s²) − ½ μ² · (σ₀² − s²) / (s² · σ₀²)

Reference: Friston & Penny (2011), *Post hoc Bayesian model selection*,
NeuroImage 56(4), p. 2089. Eq. (16) — Gaussian-conjugate special case.
Also used for structured pruning in Wright, Igel, Selvan (2024), *BMRS:
Bayesian Model Reduction for Structured Pruning*, NeurIPS spotlight, where
the per-edge posterior variance s² is estimated from the Laplace
approximation around the MAP weight (their Section 3.2).

Pruning rule (corrected): **prune if log BF > 0** (reduced model preferred).

Threshold behavior:
  - Strong concentrated edge (μ large, s² small): log BF very negative → KEEP
  - Posterior near zero (μ ≈ 0): log BF ≈ ½ log(σ₀²/s²) > 0 → PRUNE
  - Uninformed posterior (s² ≈ σ₀²): log BF ≈ 0 → ambiguous

Variance estimation: when no per-edge posterior variance is available
(e.g., a single observation per edge from cosine similarity), the cosine
measurement noise σ_n² ≈ 0.04² for BGE-small on similar text gives a
sensible default. The `estimate_edge_variance_from_gamma` heuristic is
retained as an option but is calibrated to a similar regime.
"""
from __future__ import annotations

import math


# Default cosine measurement noise for BGE-small; calibrated empirically
# from the variance of cos(s_i, s_j) over related text pairs (~0.04²).
_DEFAULT_COSINE_NOISE_VAR = 0.04 ** 2  # ≈ 1.6e-3


def bmrs_log_evidence_ratio(
    w_hat: float, s_squared: float, sigma_0_squared: float = 1.0
) -> float:
    """Closed-form log Bayes factor for reduced vs full model.

    BF(reduced/full) — the ratio of marginal likelihoods under
    "edge absent (w=0)" vs "edge present (w ~ N(0, σ₀²))".

    Friston & Penny 2011 Eq. (16), Gaussian-Gaussian conjugate special
    case:

        log BF = ½ log(σ₀² / s²) − ½ μ² (σ₀² − s²) / (s² σ₀²)

    Args:
        w_hat: posterior mean of the edge weight (e.g., observed cosine).
        s_squared: posterior variance of the edge weight; floored at 1e-6.
                   For a single noisy observation, set this to the
                   measurement noise variance.
        sigma_0_squared: prior variance over the weight under "edge
                   present"; default 1.0 (broad).

    Returns:
        log BF. Positive ⇒ reduced (absent) preferred ⇒ prune.
                Negative ⇒ full (present) preferred ⇒ keep.
    """
    s2 = max(s_squared, 1e-6)
    sigma_0 = max(sigma_0_squared, 1e-6)
    log_term = 0.5 * math.log(sigma_0 / s2)
    if abs(s2 - sigma_0) < 1e-12:
        return log_term
    mu_term = 0.5 * (w_hat * w_hat) * (sigma_0 - s2) / (s2 * sigma_0)
    return log_term - mu_term


def bmrs_should_prune(
    w_hat: float, s_squared: float, sigma_0_squared: float = 1.0
) -> bool:
    """True if reduced (edge-absent) model has higher marginal likelihood.

    v0.7 sign convention: log BF > 0 ⇒ prune.
    """
    return bmrs_log_evidence_ratio(w_hat, s_squared, sigma_0_squared) > 0.0


def cosine_edge_variance(
    weight: float, gamma_anti: float | None = None,
    base_noise_var: float = _DEFAULT_COSINE_NOISE_VAR,
) -> float:
    """Per-edge posterior variance estimate for cosine-weighted edges.

    For Recall's edge-induction pipeline a single observation per edge is
    available — the symmetric cosine cos(s_i, s_j). The posterior variance
    under a single noisy observation y with prior N(0, σ₀²) and noise
    σ_n² is:

        s² = σ₀² σ_n² / (σ₀² + σ_n²)

    Approximated here as σ_n² (since σ_n² << σ₀² in our regime) modulated
    by the directional signal strength. Edges with strong Γ_anti relative
    to magnitude are "more certain" (smaller s²), edges with no
    directional signal are slightly noisier.
    """
    s2 = base_noise_var
    if gamma_anti is not None and abs(weight) > 1e-6:
        directional_strength = abs(gamma_anti) / (abs(weight) + 1e-6)
        # Strong directionality ⇒ shrink variance (high confidence).
        # Weak directionality ⇒ inflate slightly.
        confidence_scale = 0.5 + 0.5 * min(1.0, directional_strength)
        s2 = s2 / max(0.1, confidence_scale)
    return float(max(1e-6, s2))


def estimate_edge_variance_from_gamma(
    gamma: float, gamma_anti: float | None
) -> float:
    """Legacy heuristic — kept for backward compat with existing audit.

    v0.7: prefer ``cosine_edge_variance`` for cosine-weighted edges. This
    function uses Γ symmetry/anti-symmetry ratios which were calibrated for
    HashEmbedder Γ values (range ~[-1, 1]) and over-prune on the smaller
    Γ values produced by neural embedders.
    """
    if gamma_anti is None:
        return _DEFAULT_COSINE_NOISE_VAR
    g = abs(gamma) + 1e-6
    a = abs(gamma_anti) + 1e-6
    asymmetry_ratio = a / g
    return float(max(0.05, min(2.0, 1.0 / (asymmetry_ratio + 0.1))))
