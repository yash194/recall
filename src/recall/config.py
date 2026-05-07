"""Tenant-overridable thresholds and constants for Recall."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Config:
    # --- Quality gating ---
    THRESH_QUALITY: float = 0.4

    # Patterns observable in mem0 #4573 audit + production failure modes.
    # Match: junk strings hash near these → reject.
    QUALITY_NEGATIVE_TEMPLATES: tuple[str, ...] = (
        # Generic boilerplate
        "I don't have access",
        "let me clarify",
        "sure, i can help",
        "as an ai language model",
        "i'm sorry, i cannot",
        "could you provide more details",
        # System-prompt leaks
        "i am claude",
        "i am an ai assistant",
        "you are a helpful, harmless",
        "i cannot provide medical, legal",
        # Hallucinated-profile fingerprints (mem0 #4573 pattern)
        "user is a google engineer",
        "user prefers vim",
    )

    # --- Γ-edge induction ---
    # v0.7: factorization — symmetric cosine gates EXISTENCE, Γ_anti gates DIRECTION,
    # Γ_sym + text patterns gate TYPE. The legacy single-knob THRESH_GAMMA conflated
    # the three and was operationally degenerate with neural embedders (BGE produces
    # |Γ| ≈ 0.01–0.03, well below the 0.05 floor calibrated for HashEmbedder).
    THRESH_GAMMA: float = 0.05  # legacy; retained for tests that pin the old gate
    THRESH_MIN_COSINE_EDGE: float = 0.0  # safety floor on cos(s_i, s_j); 0 = top-K only
    MAX_EDGES_PER_NODE: int = 6  # cap edges induced per new node (bidirectional pairs count once)
    THRESH_DIRECTIONAL: float = 0.20  # min |Γ_anti| / (||c_i||·||c_j||) for a directed edge
    THRESH_WALK: float = -0.10  # min weight for walk traversal (allows mild negatives)
    BEAM_WIDTH: int = 8
    K_NEIGHBORS: int = 10  # top-k symmetric neighbors per write

    # --- PAC-Bayes bound (§3 of MATH.md) ---
    DELTA: float = 0.05  # confidence parameter

    # --- BMRS pruning (§4 of MATH.md) ---
    EDGE_PRIOR_VAR: float = 1.0
    EDGE_VAR_FLOOR: float = 1e-3

    # --- Subgraph extraction ---
    BUDGET_SUBGRAPH: float = 10.0

    # --- Embedder ---
    EMBED_DIM: int = 256  # default for HashEmbedder; BGE-M3 = 1024

    # --- Edge type vocabulary ---
    EDGE_TYPES: tuple[str, ...] = (
        "supports",
        "contradicts",
        "corrects",
        "agrees",
        "pivots",
        "temporal_next",
        "superseded",
    )

    # --- Forward / backward prompt prefixes ---
    FORWARD_PROMPT: str = "Forward describe — what comes next or follows from: "
    BACKWARD_PROMPT: str = "Backward describe — what came before or causes: "

    # --- Retrieval ---
    MAX_DEPTH: int = 4
    MAX_SUBGRAPH_NODES: int = 20

    # --- Provenance firewall ---
    RECALL_ARTIFACT_SOURCES: tuple[str, ...] = (
        "recall_artifact",
        "agent_self_recall",
        "memory_dump",
    )

    # --- v0.5 bulk-document ingestion ---
    # Sources that take the fast path: skip LLM splitter, skip per-write
    # conversation-buffer scan, skip Γ-edge induction. Edges can be
    # induced in a later offline consolidation pass.
    BULK_MODE_SOURCES: tuple[str, ...] = (
        "bulk_document",
        "document",
        "corpus",
        "ingest",
    )


DEFAULT = Config()
