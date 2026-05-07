"""Core data types for Recall.

All persisted state is one of: Drawer, Node, Edge, Motif, AuditEntry.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


class EdgeType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CORRECTS = "corrects"
    AGREES = "agrees"
    PIVOTS = "pivots"
    TEMPORAL_NEXT = "temporal_next"
    SUPERSEDED = "superseded"
    PENDING = "pending"  # set at write-time before classification

    @classmethod
    def is_negative(cls, t: "EdgeType") -> bool:
        """True for edge types whose weight is conventionally negated."""
        return t in (cls.CONTRADICTS, cls.SUPERSEDED)


@dataclass(slots=True)
class Drawer:
    """Verbatim immutable text fragment — the truth layer."""

    id: str
    tenant: str
    text: str
    source: str  # 'conversation' | 'document' | 'recall_artifact'
    scope: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)
    valid_from: datetime = field(default_factory=_now)
    valid_to: datetime | None = None
    transaction_time: datetime = field(default_factory=_now)

    def scope_json(self) -> str:
        return json.dumps(self.scope, sort_keys=True, default=str)


@dataclass(slots=True)
class Node:
    """Distilled thought-unit, points to drawer ranges, has dual embeddings.

    v0.6: `s_embedding` is an optional precomputed symmetric embedding
    (derived from raw text without forward/backward prompt prefixes).
    When present, retrieval uses it directly; when absent, retrieval falls
    back to the legacy (f+b)/2 derivation.
    """

    id: str = field(default_factory=_new_id)
    tenant: str = ""
    text: str = ""
    drawer_ids: list[str] = field(default_factory=list)
    f_embedding: np.ndarray | None = None  # forward (prompted) — Γ-edges
    b_embedding: np.ndarray | None = None  # backward (prompted) — Γ-edges
    s_embedding: np.ndarray | None = None  # v0.6: raw symmetric — retrieval
    role: str | None = None  # 'fact' | 'attempt' | 'decision' | 'pivot' | 'outcome' | 'correction'
    quality_score: float = 0.0
    quality_status: str = "pending"  # 'pending' | 'promoted' | 'rejected'
    scope: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    parent_node_id: str | None = None
    deprecated_at: datetime | None = None
    deprecated_reason: str | None = None
    created_at: datetime = field(default_factory=_now)
    transaction_time: datetime = field(default_factory=_now)

    def s(self) -> np.ndarray:
        """Symmetric component used by retrieval.

        v0.6: prefer the explicit `s_embedding` field (derived from raw
        text). Fall back to the legacy (f + b) / 2 if not set, for
        backward compatibility with stored nodes from v0.5 and earlier.
        """
        if self.s_embedding is not None:
            return self.s_embedding
        assert self.f_embedding is not None and self.b_embedding is not None
        return (self.f_embedding + self.b_embedding) / 2

    def c(self) -> np.ndarray:
        """Causal component c = (f - b) / 2 — used by Γ-edge induction."""
        assert self.f_embedding is not None and self.b_embedding is not None
        return (self.f_embedding - self.b_embedding) / 2

    def is_active(self) -> bool:
        return self.deprecated_at is None and self.quality_status == "promoted"

    def scope_json(self) -> str:
        return json.dumps(self.scope, sort_keys=True, default=str)


@dataclass(slots=True)
class Edge:
    """Typed, asymmetric, signed-weighted directed edge."""

    id: str = field(default_factory=_new_id)
    tenant: str = ""
    src_node_id: str = ""
    dst_node_id: str = ""
    edge_type: EdgeType = EdgeType.PENDING
    weight: float = 0.0  # signed; contradicts → negative
    gamma_score: float = 0.0  # raw Γ(src → dst)
    gamma_anti: float | None = None
    s_squared: float = 1.0  # BMRS variance estimate
    bmrs_log_ratio: float | None = None
    deprecated_at: datetime | None = None
    deprecated_reason: str | None = None
    created_at: datetime = field(default_factory=_now)
    last_validated_at: datetime | None = None

    def is_active(self) -> bool:
        return self.deprecated_at is None


@dataclass(slots=True)
class Motif:
    id: str = field(default_factory=_new_id)
    tenant: str = ""
    pattern: dict[str, Any] = field(default_factory=dict)
    instances: list[dict[str, Any]] = field(default_factory=list)
    occurrence_count: int = 0
    parameter_summary: str | None = None
    created_at: datetime = field(default_factory=_now)


@dataclass(slots=True)
class AuditEntry:
    seq: int | None = None
    tenant: str = ""
    timestamp: datetime = field(default_factory=_now)
    operation: str = ""
    actor: str = "system"
    target_type: str = ""
    target_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None


# --- Result types returned by the public API ---


@dataclass(slots=True)
class WriteResult:
    drawer_id: str | None
    nodes_written: list[str]
    nodes_rejected: list[tuple[str, str]]  # (node_id, reason)
    nodes_skipped_duplicate: int = 0
    edges_written: int = 0
    skipped_recall_loop: bool = False
    drawer_was_duplicate: bool = False


@dataclass(slots=True)
class Path:
    nodes: list[Node]
    edges: list[Edge]

    @property
    def cum_weight(self) -> float:
        return sum(e.weight for e in self.edges)

    def extend(self, n: Node, e: Edge) -> "Path":
        return Path(nodes=self.nodes + [n], edges=self.edges + [e])


@dataclass(slots=True)
class RetrievalResult:
    query: str
    seeds: list[Node]
    paths: list[Path]
    subgraph_nodes: list[Node]
    subgraph_edges: list[Edge]
    mode: str  # 'path' | 'symmetric' | 'hybrid'

    def node_ids(self) -> list[str]:
        return [n.id for n in self.subgraph_nodes]

    def edge_ids(self) -> list[str]:
        return [e.id for e in self.subgraph_edges]


@dataclass(slots=True)
class GenerationResult:
    query: str
    text: str
    retrieved: RetrievalResult
    flagged_claims: list[str] = field(default_factory=list)
    bound_value: float | None = None
    blocked: bool = False


@dataclass(slots=True)
class Trace:
    nodes: list[Node]
    edges: list[Edge]
    drawers: list[Drawer]
    bound_value: float | None
    generated_text: str
    flagged_claims: list[str]
    audit_entries: list[AuditEntry]


@dataclass(slots=True)
class ForgetResult:
    deprecated_node_id: str | None
    deprecated_edge_ids: list[str]
    error: str | None = None
