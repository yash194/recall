"""Spectral causal amplification — Yash's own derivation (yash_math.md §6).

Key idea: even with LLM pre-filtering, the causal component c(t) may contain
noise dimensions where semantic variance dominates. Solve the generalized
eigenvalue problem to find the subspace where causal variance > semantic
variance, then project Γ into that subspace.

Generalized eigenvalue problem:

    Σ_c v = λ Σ_s^{reg} v

with Σ_s^{reg} = Σ_s + α·I  (regularization).

Eigenvalues λ_k = Var_causal(dim k) / Var_semantic(dim k).
Dimensions with λ > 1 form the "causal subspace."

In that projected space, Γ is computed as before, but the causal signal is
amplified.

This is the math Recall didn't have before — the original Γ implementation
worked in raw (f, b) space and was vulnerable to the catastrophic-cancellation
problem (yash_math.md §5.1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


@dataclass(slots=True)
class SpectralProjector:
    """Holds the causal-subspace projection matrix P (d × k).

    Use:
        proj = SpectralProjector.fit(f_b_pairs)
        c_proj = proj.project_c(c_vec)
        gamma_spec = proj.gamma_spec(f_i, b_i, f_j, b_j)
    """

    P: np.ndarray  # (d × k) projection onto causal subspace
    eigenvalues: np.ndarray  # (k,) eigenvalues, sorted descending
    threshold: float
    d: int

    @classmethod
    def fit(
        cls,
        pairs: list[tuple[np.ndarray, np.ndarray]],
        threshold: float = 1.0,
        ridge: float = 0.01,
        min_dim: int = 4,
    ) -> "SpectralProjector":
        """Fit the spectral projector from a calibration set of (f, b) pairs.

        Args:
            pairs: list of (f_t, b_t) embedding pairs from observed texts.
            threshold: keep dimensions with eigenvalue ≥ threshold.
                       1.0 = "causal variance ≥ semantic variance."
            ridge: relative ridge added to Σ_s for stability.
            min_dim: minimum subspace dimension.

        Returns:
            A fitted SpectralProjector.
        """
        if not pairs:
            raise ValueError("Need at least one (f, b) pair.")

        F = np.stack([p[0] for p in pairs])
        B = np.stack([p[1] for p in pairs])
        S = 0.5 * (F + B)
        C = 0.5 * (F - B)

        mu_s = S.mean(axis=0)
        mu_c = C.mean(axis=0)
        Sc = S - mu_s
        Cc = C - mu_c
        n = max(1, len(pairs) - 1)
        Sigma_s = (Sc.T @ Sc) / n
        Sigma_c = (Cc.T @ Cc) / n
        d = F.shape[1]

        # Regularize Σ_s
        alpha = ridge * float(np.trace(Sigma_s)) / max(1, d)
        Sigma_s_reg = Sigma_s + alpha * np.eye(d)

        # Generalized eigenvalue problem: Σ_c v = λ Σ_s^{reg} v
        # scipy.linalg.eigh handles the generalized problem when both matrices
        # are symmetric.
        try:
            evals, evecs = scipy.linalg.eigh(Sigma_c, Sigma_s_reg)
        except np.linalg.LinAlgError:
            # Fall back to identity if numerics fail
            return cls(P=np.eye(d, dtype=np.float32),
                       eigenvalues=np.ones(d, dtype=np.float32),
                       threshold=threshold, d=d)

        # Sort descending
        order = np.argsort(-evals)
        evals = evals[order]
        evecs = evecs[:, order]

        # Keep dims with eigenvalue ≥ threshold (or at least min_dim)
        keep = max(min_dim, int((evals >= threshold).sum()))
        keep = min(keep, d)
        P = evecs[:, :keep].astype(np.float32)
        evals_kept = evals[:keep].astype(np.float32)

        return cls(P=P, eigenvalues=evals_kept, threshold=threshold, d=d)

    def project_f(self, f: np.ndarray) -> np.ndarray:
        """f̃ = Pᵀ f"""
        return (self.P.T @ f).astype(np.float32)

    def project_b(self, b: np.ndarray) -> np.ndarray:
        return (self.P.T @ b).astype(np.float32)

    def project_s(self, s: np.ndarray) -> np.ndarray:
        return (self.P.T @ s).astype(np.float32)

    def project_c(self, c: np.ndarray) -> np.ndarray:
        return (self.P.T @ c).astype(np.float32)

    def gamma_spec(
        self, f_i: np.ndarray, b_i: np.ndarray, f_j: np.ndarray, b_j: np.ndarray
    ) -> float:
        """Spectral Γ — yash_math.md §6.4.

            Γ_spec(i→j) = c̃(i)ᵀs̃(j) − s̃(i)ᵀc̃(j) − c̃(i)ᵀc̃(j)
        """
        s_i = 0.5 * (f_i + b_i)
        c_i = 0.5 * (f_i - b_i)
        s_j = 0.5 * (f_j + b_j)
        c_j = 0.5 * (f_j - b_j)
        ci = self.P.T @ c_i
        cj = self.P.T @ c_j
        si = self.P.T @ s_i
        sj = self.P.T @ s_j
        return float(ci @ sj - si @ cj - ci @ cj)

    def gamma_spec_weighted(
        self, f_i: np.ndarray, b_i: np.ndarray, f_j: np.ndarray, b_j: np.ndarray
    ) -> float:
        """Eigenvalue-weighted Γ — yash_math.md §6.5.

        Weight each projected dim by its eigenvalue (more causal = more
        weight): W = diag(λ_1, ..., λ_k).

            Γ_w(i→j) = (W·c̃(i))ᵀs̃(j) − s̃(i)ᵀ(W·c̃(j)) − (W·c̃(i))ᵀc̃(j)
        """
        s_i = 0.5 * (f_i + b_i)
        c_i = 0.5 * (f_i - b_i)
        s_j = 0.5 * (f_j + b_j)
        c_j = 0.5 * (f_j - b_j)
        ci = self.P.T @ c_i
        cj = self.P.T @ c_j
        si = self.P.T @ s_i
        sj = self.P.T @ s_j
        W = self.eigenvalues
        Wci = W * ci
        Wcj = W * cj
        return float(Wci @ sj - si @ Wcj - Wci @ cj)

    def causal_to_semantic_ratio(self) -> float:
        """Diagnostic: average ‖c̃‖ / ‖s̃‖ in the projected subspace.

        Per yash_math.md §2.3, ρ > 0.3 is the threshold for Γ to work.
        """
        # In the projected subspace, the c-direction has variance ~ mean(λ_k)
        # and the s-direction has variance ~ 1.
        return float(np.sqrt(np.mean(self.eigenvalues)))


def fit_from_embedder(embedder, calibration_texts: list[str], **kwargs) -> SpectralProjector:
    """Convenience: build a SpectralProjector from an Embedder + texts."""
    pairs = []
    for t in calibration_texts:
        f, b = embedder.embed_dual(t)
        pairs.append((f, b))
    return SpectralProjector.fit(pairs, **kwargs)
