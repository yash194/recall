"""Geometric primitives — Γ score and decompositions."""
from recall.geometry.gamma import (
    gamma_score,
    gamma_split,
    gamma_anti,
    gamma_sym,
    semantic_component,
    causal_component,
)
from recall.geometry.spectral import SpectralProjector, fit_from_embedder
from recall.geometry.llm_dual_view import LLMPrefilteredEmbedder

__all__ = [
    "gamma_score",
    "gamma_split",
    "gamma_anti",
    "gamma_sym",
    "semantic_component",
    "causal_component",
    "SpectralProjector",
    "fit_from_embedder",
    "LLMPrefilteredEmbedder",
]
