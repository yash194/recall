"""Γ — the directional retrieval primitive.

Implements:
    Γ(i → j) = f_i · b_j − s_i · s_j
    s = (f + b) / 2     (semantic, symmetric)
    c = (f - b) / 2     (causal, antisymmetric)

References MATH.md §1 for derivations.
"""
from __future__ import annotations

import numpy as np


def semantic_component(f: np.ndarray, b: np.ndarray) -> np.ndarray:
    """s = (f + b) / 2 — the symmetric / 'meaning' component."""
    return 0.5 * (f + b)


def causal_component(f: np.ndarray, b: np.ndarray) -> np.ndarray:
    """c = (f - b) / 2 — the antisymmetric / 'direction' component."""
    return 0.5 * (f - b)


def gamma_score(
    f_i: np.ndarray, b_i: np.ndarray, f_j: np.ndarray, b_j: np.ndarray
) -> float:
    """Compute Γ(i → j) = f_i · b_j − s_i · s_j.

    The forward + backward embeddings are treated as the dual-prompt views.
    Returns a scalar; positive ⇒ forward direction (i causes/precedes j).

    Per MATH.md §1.1, this can also be written:
        Γ(i → j) = c_i · s_j − s_i · c_j − c_i · c_j
    """
    s_i = semantic_component(f_i, b_i)
    s_j = semantic_component(f_j, b_j)
    return float(np.dot(f_i, b_j) - np.dot(s_i, s_j))


def gamma_split(
    f_i: np.ndarray, b_i: np.ndarray, f_j: np.ndarray, b_j: np.ndarray
) -> tuple[float, float]:
    """Decompose Γ into symmetric + antisymmetric components.

    Per MATH.md §1.3 / Theorem 1.2:
        Γ_sym(i,j)  = -c_i · c_j
        Γ_anti(i,j) = ⟨c_i, s_j⟩ - ⟨c_j, s_i⟩

    Returns (Γ_sym, Γ_anti).
    """
    s_i = semantic_component(f_i, b_i)
    s_j = semantic_component(f_j, b_j)
    c_i = causal_component(f_i, b_i)
    c_j = causal_component(f_j, b_j)
    sym = -float(np.dot(c_i, c_j))
    anti = float(np.dot(c_i, s_j) - np.dot(c_j, s_i))
    return sym, anti


def gamma_sym(
    f_i: np.ndarray, b_i: np.ndarray, f_j: np.ndarray, b_j: np.ndarray
) -> float:
    """Symmetric component only: −c_i · c_j (MATH.md §1.3, eqn 1.3)."""
    return gamma_split(f_i, b_i, f_j, b_j)[0]


def gamma_anti(
    f_i: np.ndarray, b_i: np.ndarray, f_j: np.ndarray, b_j: np.ndarray
) -> float:
    """Antisymmetric component: ⟨c_i, s_j⟩ - ⟨c_j, s_i⟩ (MATH.md §1.3, eqn 1.4).

    This is the directional signal — Γ_anti(i,j) = -Γ_anti(j,i) by construction.
    """
    return gamma_split(f_i, b_i, f_j, b_j)[1]


def asymmetry_diagnostic(
    f_i: np.ndarray, b_i: np.ndarray, f_j: np.ndarray, b_j: np.ndarray
) -> float:
    """Returns Γ(i → j) − Γ(j → i).

    By Theorem 1.1 (MATH.md), this equals 2·(⟨c_i, s_j⟩ − ⟨c_j, s_i⟩) = 2·Γ_anti(i, j).
    Useful for verifying the asymmetry theorem in tests.
    """
    return gamma_score(f_i, b_i, f_j, b_j) - gamma_score(f_j, b_j, f_i, b_i)
