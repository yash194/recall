"""Tests for the Γ retrieval primitive (MATH.md §1)."""
from __future__ import annotations

import numpy as np
import pytest

from recall.geometry.gamma import (
    asymmetry_diagnostic,
    causal_component,
    gamma_anti,
    gamma_score,
    gamma_split,
    gamma_sym,
    semantic_component,
)


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _random_pair(rng: np.random.Generator, dim: int = 32) -> tuple[np.ndarray, np.ndarray]:
    f = _norm(rng.standard_normal(dim).astype(np.float32))
    b = _norm(rng.standard_normal(dim).astype(np.float32))
    return f, b


def test_gamma_score_returns_scalar():
    rng = np.random.default_rng(42)
    f_i, b_i = _random_pair(rng)
    f_j, b_j = _random_pair(rng)
    g = gamma_score(f_i, b_i, f_j, b_j)
    assert isinstance(g, float)


def test_components_recover_f_and_b():
    """s + c == f and s - c == b — by construction."""
    rng = np.random.default_rng(7)
    f, b = _random_pair(rng)
    s = semantic_component(f, b)
    c = causal_component(f, b)
    np.testing.assert_allclose(s + c, f, atol=1e-6)
    np.testing.assert_allclose(s - c, b, atol=1e-6)


def test_s_orthogonal_to_c_for_unit_vectors():
    """For L2-normalized f and b, s · c = 0 (parallelogram identity)."""
    rng = np.random.default_rng(13)
    for _ in range(50):
        f, b = _random_pair(rng)
        s = semantic_component(f, b)
        c = causal_component(f, b)
        # s · c = ¼ (||f||² - ||b||²) = 0 for unit vectors
        assert abs(float(np.dot(s, c))) < 1e-5


def test_gamma_self_anti_is_zero():
    """Γ(i → i) is purely symmetric: Γ_anti(i, i) = 0."""
    rng = np.random.default_rng(99)
    f, b = _random_pair(rng)
    sym, anti = gamma_split(f, b, f, b)
    assert abs(anti) < 1e-6
    # Sym = -c·c = -||c||² (always ≤ 0 for nonzero c)
    c = causal_component(f, b)
    np.testing.assert_allclose(sym, -float(np.dot(c, c)), atol=1e-6)


def test_asymmetry_theorem_1_1():
    """MATH.md Theorem 1.1: Γ(i,j) - Γ(j,i) = 2·(c_i·s_j - c_j·s_i) = 2·Γ_anti."""
    rng = np.random.default_rng(2025)
    for _ in range(100):
        f_i, b_i = _random_pair(rng)
        f_j, b_j = _random_pair(rng)
        diff = asymmetry_diagnostic(f_i, b_i, f_j, b_j)
        anti = gamma_anti(f_i, b_i, f_j, b_j)
        np.testing.assert_allclose(diff, 2 * anti, atol=1e-5)


def test_asymmetry_is_typically_nonzero():
    """For random pairs, Γ(i→j) ≠ Γ(j→i) almost surely."""
    rng = np.random.default_rng(1729)
    nonzero_count = 0
    n = 50
    for _ in range(n):
        f_i, b_i = _random_pair(rng)
        f_j, b_j = _random_pair(rng)
        if abs(asymmetry_diagnostic(f_i, b_i, f_j, b_j)) > 1e-3:
            nonzero_count += 1
    # Generically all should be nonzero (asymmetry theorem)
    assert nonzero_count >= n - 2  # tiny floating-point slack


def test_gamma_split_identities():
    """gamma_sym + gamma_anti reconstruct gamma_score for the (i,j) ordering only when
    averaged with the (j,i) score: Γ(i,j) = Γ_sym(i,j) + Γ_anti(i,j)."""
    rng = np.random.default_rng(31)
    for _ in range(20):
        f_i, b_i = _random_pair(rng)
        f_j, b_j = _random_pair(rng)
        sym = gamma_sym(f_i, b_i, f_j, b_j)
        anti = gamma_anti(f_i, b_i, f_j, b_j)
        # Γ(i→j) = (Γ(i→j) + Γ(j→i))/2 + (Γ(i→j) - Γ(j→i))/2 = sym + anti
        full = gamma_score(f_i, b_i, f_j, b_j)
        np.testing.assert_allclose(full, sym + anti, atol=1e-5)


def test_zero_vectors_dont_explode():
    """Edge case: zero embedding vectors should yield Γ = 0 without errors."""
    z = np.zeros(16, dtype=np.float32)
    g = gamma_score(z, z, z, z)
    assert g == 0.0
