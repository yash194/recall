"""Tests for PAC-Bayes bound (MATH.md §3) and BMRS pruning (MATH.md §4)."""
from __future__ import annotations

import math

import pytest

from recall.bound.pac_bayes import (
    chebyshev_cantelli_bound,
    compute_bound_estimate,
    is_bound_vacuous,
    pac_bayes_bound,
)
from recall.consolidate.bmrs import (
    bmrs_log_evidence_ratio,
    bmrs_should_prune,
    estimate_edge_variance_from_gamma,
)


# --- PAC-Bayes ---


def test_pac_bayes_bound_decreases_with_n():
    """As sample size grows, the bound tightens."""
    b_small = pac_bayes_bound(L_hat=0.2, L_T_hat=0.05, kl=1.0, n=100, delta=0.05)
    b_large = pac_bayes_bound(L_hat=0.2, L_T_hat=0.05, kl=1.0, n=10000, delta=0.05)
    assert b_large < b_small


def test_pac_bayes_bound_increases_with_kl():
    """More divergent posterior → looser bound."""
    b_low_kl = pac_bayes_bound(L_hat=0.1, L_T_hat=0.05, kl=0.1, n=1000, delta=0.05)
    b_high_kl = pac_bayes_bound(L_hat=0.1, L_T_hat=0.05, kl=5.0, n=1000, delta=0.05)
    assert b_high_kl >= b_low_kl


def test_chebyshev_returns_none_when_vacuous():
    """When tandem ≥ mean loss, no margin → None."""
    bound = chebyshev_cantelli_bound(L_hat=0.5, L_T_hat=0.5, kl=1.0, n=100, delta=0.05)
    assert bound is None


def test_chebyshev_non_vacuous_with_diverse_paths():
    """With low tandem (diverse paths), bound is non-vacuous."""
    bound = chebyshev_cantelli_bound(L_hat=0.4, L_T_hat=0.05, kl=0.5, n=10000, delta=0.05)
    assert bound is not None
    assert 0 <= bound <= 1


def test_compute_bound_estimate_returns_value_or_none():
    """End-to-end bound estimation."""
    b = compute_bound_estimate(
        n_paths=8, avg_path_loss=0.4, avg_tandem_loss=0.05,
        n_training_samples=10_000, delta=0.05,
    )
    # Either non-vacuous in [0,1] or None
    assert b is None or 0 <= b <= 1


def test_is_bound_vacuous():
    assert is_bound_vacuous(None)
    assert is_bound_vacuous(1.0)
    assert is_bound_vacuous(1.5)
    assert not is_bound_vacuous(0.5)


# --- BMRS ---


def test_bmrs_keeps_edges_with_strong_weight():
    """An edge with large |w_hat| relative to its variance should be kept.

    v0.7: corrected formula. log BF(reduced/full) = ½ log(σ₀²/s²) −
    ½ μ²(σ₀²−s²)/(s²·σ₀²).  For w=0.7, s²=0.01, σ₀²=1.0: log BF ≈ 2.30 −
    24.25 ≈ −21.95 → keep (full model preferred).
    """
    log_ratio = bmrs_log_evidence_ratio(w_hat=0.7, s_squared=0.01, sigma_0_squared=1.0)
    assert log_ratio < 0  # full model preferred → keep
    assert not bmrs_should_prune(w_hat=0.7, s_squared=0.01, sigma_0_squared=1.0)


def test_bmrs_prunes_weak_signal():
    """Edge with tiny w_hat at realistic posterior variance: prune.

    v0.7: realistic s² = cosine measurement noise (0.01).
    For w=0.001, s²=0.01, σ₀²=1.0: log BF ≈ 2.303 → prune (reduced preferred).
    """
    assert bmrs_should_prune(w_hat=0.001, s_squared=0.01, sigma_0_squared=1.0)


def test_bmrs_evidence_ratio_signs():
    """Sanity-check the sign of the log Bayes factor for canonical cases."""
    # When posterior == prior (no data update), log BF = 0 — no evidence either way.
    r0 = bmrs_log_evidence_ratio(w_hat=0.0, s_squared=1.0, sigma_0_squared=1.0)
    assert abs(r0) < 1e-9

    # Strong concentrated posterior far from zero: log BF very negative → keep.
    r_strong = bmrs_log_evidence_ratio(w_hat=1.0, s_squared=0.01, sigma_0_squared=1.0)
    assert r_strong < -10

    # Concentrated posterior at zero: log BF positive → prune.
    r_zero = bmrs_log_evidence_ratio(w_hat=0.0, s_squared=0.01, sigma_0_squared=1.0)
    assert r_zero > 0  # ½ log(100) ≈ 2.3


def test_bmrs_variance_estimate_responds_to_anti_ratio():
    v_low_anti = estimate_edge_variance_from_gamma(gamma=0.5, gamma_anti=0.01)
    v_high_anti = estimate_edge_variance_from_gamma(gamma=0.5, gamma_anti=0.45)
    # high anti / gamma → low variance (more certain)
    assert v_high_anti < v_low_anti


def test_bmrs_variance_floor():
    v = estimate_edge_variance_from_gamma(gamma=1.0, gamma_anti=1.0)
    assert v >= 0.05
    assert v <= 2.0
