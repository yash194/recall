"""Write pipeline — Tier-1 ingestion.

Implements ARCHITECTURE.md §5.1 / observe(). Steps:

  1. Hash dedup on the verbatim drawer.
  2. Provenance firewall — reject if source is a recall artifact.
  3. Persist drawer.
  4. Node split (LLM with sentence-level fallback).
  5. For each node: dual embedding, quality gate, top-k neighbor lookup,
     Γ-edge induction.
  6. Persist nodes + edges; audit-log every operation.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from recall.audit.log import AuditLog
from recall.config import Config, DEFAULT
from recall.core.storage import Storage
from recall.embeddings import Embedder
from recall.geometry.gamma import gamma_anti, gamma_score, gamma_sym
from recall.llm import LLMClient
from recall.types import Drawer, Edge, EdgeType, Node, WriteResult
from recall.write.edge_classifier import classify_edge_type, signed_weight_for_type
from recall.write.quality import QualityClassifier
from recall.write.splitter import split_into_thoughts


def _drawer_id(text: str, scope: dict, source: str, tenant: str) -> str:
    """Stable drawer id = sha256 of (tenant, scope, source, text)."""
    import json

    blob = json.dumps(
        {"tenant": tenant, "scope": scope, "source": source, "text": text},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


class WritePipeline:
    """Encapsulates the Tier-1 write path."""

    def __init__(
        self,
        tenant: str,
        storage: Storage,
        embedder: Embedder,
        audit: AuditLog,
        llm: LLMClient | None = None,
        config: Config | None = None,
        quality_classifier: Any = None,
    ):
        self.tenant = tenant
        self.storage = storage
        self.embedder = embedder
        self.audit = audit
        self.llm = llm
        self.cfg = config or DEFAULT
        self.quality = quality_classifier or QualityClassifier(self.cfg)

    def observe(
        self,
        user_msg: str,
        agent_msg: str,
        scope: dict[str, Any] | None = None,
        source: str = "conversation",
        fast: bool | None = None,
    ) -> WriteResult:
        """Ingest one drawer.

        v0.5: If `fast=True` or `source` is in cfg.BULK_MODE_SOURCES, take
        the bulk-document path: sentence splitter only, skip the
        all-active-nodes conversation buffer, and skip Γ-edge induction.
        Edges can be induced in a later batch via `consolidate()`.
        """
        scope = scope or {}
        raw = (user_msg + " | AGENT: " + agent_msg).strip()
        if not raw:
            return WriteResult(drawer_id=None, nodes_written=[], nodes_rejected=[])

        # Determine fast path: explicit flag overrides source-based detection
        if fast is None:
            fast = source in self.cfg.BULK_MODE_SOURCES

        # Step 2 — Provenance firewall
        if source in self.cfg.RECALL_ARTIFACT_SOURCES:
            self.audit.append(
                "REJECT_RECALL_LOOP", "drawer", "<unwritten>", reason="provenance_firewall"
            )
            return WriteResult(
                drawer_id=None,
                nodes_written=[],
                nodes_rejected=[],
                skipped_recall_loop=True,
            )

        drawer = Drawer(
            id=_drawer_id(raw, scope, source, self.tenant),
            tenant=self.tenant,
            text=raw,
            source=source,
            scope=scope,
        )

        # Step 1 — drawer-level dedup
        if self.storage.has_drawer(drawer.id):
            self.audit.append(
                "SKIP_DUPLICATE_DRAWER", "drawer", drawer.id, reason="hash_dedup"
            )
            return WriteResult(
                drawer_id=drawer.id,
                nodes_written=[],
                nodes_rejected=[],
                drawer_was_duplicate=True,
            )

        # Step 3 — persist drawer
        self.storage.insert_drawer(drawer)
        self.audit.append("WRITE", "drawer", drawer.id, payload={"len": len(raw)})

        # Step 4 — node split. Fast path uses sentence splitter only (no LLM).
        spans = split_into_thoughts(raw, llm=None if fast else self.llm)
        if not spans:
            return WriteResult(drawer_id=drawer.id, nodes_written=[], nodes_rejected=[])

        # v0.6: batch-encode all span texts in one embedder call when the
        # embedder supports it (BGEEmbedder.embed_batch). Drops per-write
        # latency from O(spans · BGE_inference) to O(BGE_inference).
        span_texts = [s.text for s in spans]
        f_list: list = [None] * len(spans)
        b_list: list = [None] * len(spans)
        s_list: list = [None] * len(spans)
        if hasattr(self.embedder, "embed_batch"):
            try:
                f_list, b_list, s_list = self.embedder.embed_batch(span_texts)
            except Exception:
                # If batch encode fails for any reason, fall back to per-span
                f_list = [None] * len(spans)
                b_list = [None] * len(spans)
                s_list = [None] * len(spans)

        nodes_written: list[str] = []
        nodes_rejected: list[tuple[str, str]] = []
        nodes_skipped = 0
        edges_written = 0

        # Step 5 — for each candidate node
        for idx, span in enumerate(spans):
            node = Node(
                tenant=self.tenant,
                text=span.text,
                drawer_ids=[drawer.id],
                role=span.role,
                scope=scope,
            )
            # 5a — embeddings. Use batch results if available; otherwise
            # fall back to per-span calls (works for any Embedder impl).
            if f_list[idx] is not None and b_list[idx] is not None:
                f, b = f_list[idx], b_list[idx]
            else:
                f, b = self.embedder.embed_dual(span.text)
            # v0.6: store raw symmetric embedding for retrieval
            if s_list[idx] is not None:
                s_vec = s_list[idx]
            elif hasattr(self.embedder, "embed_symmetric"):
                s_vec = self.embedder.embed_symmetric(span.text)
            else:
                s_vec = (f + b) / 2.0
            node.f_embedding = f
            node.b_embedding = b
            node.s_embedding = s_vec

            # 5b — quality gate. Fast path skips the all_active_nodes scan.
            # Slow path uses a SQL-side LIMIT (v0.5) so we don't materialize
            # every node just to take the last 20.
            #
            # v0.7: when there are no prior conversation nodes, pass an EMPTY
            # buffer (not [raw]). The bio-fingerprint anchor-check needs to
            # see "no anchor" for genuinely first-time fabricated profile
            # claims; using [raw] as the conversation defeated the gate
            # because the bio's own entities trivially appear in [raw].
            if fast:
                recent_drawer_texts: list[str] = []
            else:
                try:
                    recent_drawer_texts = []
                    for n_recent in self.storage.all_active_nodes(
                        scope=scope, limit=20,
                    ):
                        if n_recent.id != node.id:
                            recent_drawer_texts.append(n_recent.text)
                except Exception:
                    recent_drawer_texts = []

            try:
                score, status = self.quality.classify(node, conversation=recent_drawer_texts)
            except TypeError:
                # Older signature (no conversation kw)
                score, status = self.quality.classify(node)
            node.quality_score = score
            if status == "rejected":
                node.quality_status = "rejected"
                self.storage.insert_node(node)  # persist for audit
                self.audit.append(
                    "REJECT", "node", node.id, reason="low_quality",
                    payload={"score": score, "text": span.text[:120]},
                )
                nodes_rejected.append((node.id, "low_quality"))
                continue

            # 5c — text-hash dedup at node level
            if self.storage.has_node_with_text_hash(_text_hash(span.text), scope):
                self.audit.append(
                    "SKIP_DUPLICATE_NODE", "node", node.id, reason="text_hash_dedup"
                )
                nodes_skipped += 1
                continue

            # 5d — Edge induction (v0.7 factorized).
            #
            # Existence  ← symmetric cosine cos(s_i, s_j)        [top-K cap]
            # Weight     ← cos(s_i, s_j), signed by edge type    [real semantic signal]
            # Direction  ← Γ_anti / (||c_i||·||c_j||)             [normalized causal probe]
            # Type       ← classify_edge_type(Γ_sym, Γ_anti, txt) [structural + lexical]
            #
            # Why this is right: Γ(i→j) = -c_i·c_j + (c_i·s_j - c_j·s_i) decomposes
            # into a *symmetric* component (-c·c, near zero whenever f≈b) and an
            # *antisymmetric* component (the directional probe). Gating edge
            # existence on |Γ| means edges only form when there is BOTH semantic
            # similarity AND directional structure — degenerate with modern
            # embedders where ||c|| is small. Decoupling these is principled.
            edges_for_this_node: list[Edge] = []
            if fast:
                neighbors: list = []
            else:
                neighbors = self.storage.topk_neighbors_for_gamma(
                    node, scope, k=self.cfg.K_NEIGHBORS
                )

            # Pre-compute node's symmetric vector + causal magnitude
            node_s_vec = node.s_embedding if node.s_embedding is not None else (
                0.5 * (node.f_embedding + node.b_embedding)
            )
            node_s_norm = float(np.linalg.norm(node_s_vec)) + 1e-12
            node_c_vec = 0.5 * (node.f_embedding - node.b_embedding)
            node_c_norm = float(np.linalg.norm(node_c_vec))

            edges_count = 0  # counts neighbor pairs (each may emit 1–2 directed edges)
            for n_neighbor in neighbors:
                if n_neighbor.f_embedding is None or n_neighbor.b_embedding is None:
                    continue

                # 1. Symmetric similarity — gates edge EXISTENCE
                neighbor_s_vec = (
                    n_neighbor.s_embedding if n_neighbor.s_embedding is not None
                    else 0.5 * (n_neighbor.f_embedding + n_neighbor.b_embedding)
                )
                neighbor_s_norm = float(np.linalg.norm(neighbor_s_vec)) + 1e-12
                cos_s = float(
                    np.dot(node_s_vec, neighbor_s_vec)
                    / (node_s_norm * neighbor_s_norm)
                )
                if cos_s < self.cfg.THRESH_MIN_COSINE_EDGE:
                    continue
                if edges_count >= self.cfg.MAX_EDGES_PER_NODE:
                    break

                # 2. Γ-decomposition — direction + type signals
                g_sym = gamma_sym(
                    node.f_embedding, node.b_embedding,
                    n_neighbor.f_embedding, n_neighbor.b_embedding,
                )
                g_anti = gamma_anti(
                    node.f_embedding, node.b_embedding,
                    n_neighbor.f_embedding, n_neighbor.b_embedding,
                )
                # Γ(i→j) = Γ_sym + Γ_anti and Γ(j→i) = Γ_sym − Γ_anti
                gamma_ij = g_sym + g_anti
                gamma_ji = g_sym - g_anti

                # 3. Direction strength — normalized so it's embedder-scale-invariant
                neighbor_c_vec = 0.5 * (
                    n_neighbor.f_embedding - n_neighbor.b_embedding
                )
                neighbor_c_norm = float(np.linalg.norm(neighbor_c_vec))
                c_norm_prod = node_c_norm * neighbor_c_norm + 1e-9
                direction_strength = (
                    abs(g_anti) / c_norm_prod if c_norm_prod > 1e-9 else 0.0
                )

                # 4. Direction decision — bidirectional unless directional signal exceeds floor
                if direction_strength < self.cfg.THRESH_DIRECTIONAL:
                    pairs = [
                        (node, n_neighbor, gamma_ij, g_anti),
                        (n_neighbor, node, gamma_ji, -g_anti),
                    ]
                elif g_anti > 0:
                    pairs = [(node, n_neighbor, gamma_ij, g_anti)]
                else:
                    pairs = [(n_neighbor, node, gamma_ji, -g_anti)]

                # 5. Materialize edges with cosine-based weight
                for src_node, dst_node, gamma_pair, anti_pair in pairs:
                    etype = classify_edge_type(src_node, dst_node, gamma_pair, anti_pair)
                    weight = signed_weight_for_type(etype, cos_s)
                    e = Edge(
                        tenant=self.tenant,
                        src_node_id=src_node.id,
                        dst_node_id=dst_node.id,
                        edge_type=etype,
                        weight=weight,
                        gamma_score=gamma_pair,
                        gamma_anti=anti_pair,
                        s_squared=1.0,
                    )
                    edges_for_this_node.append(e)

                edges_count += 1

            # 5e — promote node + persist edges
            node.quality_status = "promoted"
            self.storage.insert_node(node)
            self.audit.append(
                "PROMOTE", "node", node.id,
                payload={"score": score, "edges": len(edges_for_this_node)},
            )
            for e in edges_for_this_node:
                self.storage.insert_edge(e)
                self.audit.append(
                    "WRITE", "edge", e.id,
                    payload={"src": e.src_node_id, "dst": e.dst_node_id, "weight": e.weight},
                )
            edges_written += len(edges_for_this_node)
            nodes_written.append(node.id)

        return WriteResult(
            drawer_id=drawer.id,
            nodes_written=nodes_written,
            nodes_rejected=nodes_rejected,
            nodes_skipped_duplicate=nodes_skipped,
            edges_written=edges_written,
        )
