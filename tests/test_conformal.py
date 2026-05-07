"""Tests for Conformal Risk Control bound."""
from __future__ import annotations

from recall.bound.conformal import (
    crc_bound,
    crc_hoeffding,
    crc_wilson,
    split_conformal_threshold,
)


def test_crc_hoeffding_decreases_with_n():
    b_small = crc_hoeffding(0.1, 100)
    b_large = crc_hoeffding(0.1, 10000)
    assert b_large < b_small


def test_crc_wilson_decreases_with_n():
    b_small = crc_wilson(0.1, 100)
    b_large = crc_wilson(0.1, 10000)
    assert b_large < b_small


def test_crc_bound_returns_dict():
    risks = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0] * 10  # 20% rate, n=100
    b = crc_bound(risks)
    assert "hoeffding" in b
    assert "wilson" in b
    assert "crc_min" in b
    assert b["empirical_risk"] == 0.2
    assert b["n"] == 100
    # Bound should be tighter than 1.0
    assert b["crc_min"] < 1.0


def test_crc_bound_empty_returns_one():
    b = crc_bound([])
    assert b["crc_min"] == 1.0


def test_crc_bound_non_vacuous_at_realistic_n():
    """At n=300, R̂=0.10, the conformal bound should be << 1.0."""
    risks = [0] * 270 + [1] * 30  # 10% empirical risk
    b = crc_bound(risks, delta=0.05)
    assert b["crc_min"] < 0.25
    # Compared to typical PAC-Bayes which would be > 1.0


def test_split_conformal_threshold_sorted():
    scores = [0.1, 0.3, 0.5, 0.7, 0.9]
    q = split_conformal_threshold(scores, alpha=0.2)
    # 80% coverage on n=5 → quantile ≈ 0.8 → returns ~0.7-0.9
    assert 0.7 <= q <= 0.9
