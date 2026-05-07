"""Sleep-time consolidator scheduler.

Per ARCHITECTURE.md §5.8: priority queue of dirty regions, runs:
  1. BMRS edge pruning (Wright-Igel-Selvan NeurIPS 2024)
  2. Mean-field GNN refinement (Selvan MedIA 2020)
  3. Motif extraction (Selvan Mosaic-of-Motifs 2026)
  4. Bound bookkeeping update

A "region" is a connected component of nodes sharing scope. Priority is:
    Δlocal_disagreement + Δrecency + ΔΓ-density (since last visit).

Triggered manually via `Memory.consolidate()`. v2 will run as a background
asyncio task with daily budgets per tenant.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from recall.audit.log import AuditLog
from recall.consolidate.bmrs import (
    bmrs_log_evidence_ratio,
    cosine_edge_variance,
    estimate_edge_variance_from_gamma,
)
from recall.consolidate.mean_field import mean_field_iterate
from recall.consolidate.motif import find_recurring_subgraphs
from recall.core.storage import Storage
from recall.types import Edge, Node


@dataclass(slots=True)
class ConsolidationStats:
    regions_processed: int = 0
    edges_pruned: int = 0
    edges_refined: int = 0
    motifs_found: int = 0
    nodes_merged: int = 0
    edges_induced: int = 0


def _build_regions(nodes: list[Node], edges: list[Edge]) -> list[set[str]]:
    """Connected components by undirected edge graph + scope."""
    adj: dict[str, set[str]] = defaultdict(set)
    for n in nodes:
        adj[n.id]  # ensure key exists
    for e in edges:
        if e.src_node_id in adj and e.dst_node_id in adj:
            adj[e.src_node_id].add(e.dst_node_id)
            adj[e.dst_node_id].add(e.src_node_id)

    seen: set[str] = set()
    regions: list[set[str]] = []
    for nid in adj:
        if nid in seen:
            continue
        # BFS
        stack = [nid]
        comp: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.add(cur)
            for nb in adj[cur]:
                if nb not in seen:
                    stack.append(nb)
        regions.append(comp)
    return regions


def _region_priority(
    region: set[str], nodes_by_id: dict[str, Node], edges: list[Edge]
) -> float:
    """Heuristic priority score: weighted by recency + edge density + disagreement."""
    region_edges = [e for e in edges if e.src_node_id in region and e.dst_node_id in region]
    if not region_edges:
        return 0.0

    # Recency: newer regions = higher priority
    most_recent = max((e.created_at for e in region_edges), default=datetime.now())
    age_seconds = (datetime.now() - most_recent.replace(tzinfo=None) if most_recent.tzinfo else datetime.now() - most_recent).total_seconds()
    recency = 1.0 / (1.0 + age_seconds / 3600.0)  # hour-decay

    # Density: edges per node
    density = len(region_edges) / max(1, len(region))

    # Disagreement: variance of edge weights (mixed signs → contested)
    weights = np.array([e.weight for e in region_edges], dtype=np.float32)
    disagreement = float(np.std(weights)) if len(weights) > 1 else 0.0

    return 0.4 * recency + 0.3 * density + 0.3 * disagreement


class Consolidator:
    """Sleep-time consolidator.

    Usage:
        c = Consolidator(storage, audit)
        stats = c.run(budget=10)  # process up to 10 regions
    """

    def __init__(self, storage: Storage, audit: AuditLog):
        self.storage = storage
        self.audit = audit

    def run(
        self,
        budget: int = 10,
        scope: dict[str, Any] | None = None,
        sigma_0_squared: float = 1.0,
        induce_edges: bool = False,
        induce_k: int = 6,
        induce_gamma_threshold: float | None = None,
    ) -> ConsolidationStats:
        """Run the consolidator.

        v0.5: pass `induce_edges=True` to induce Γ-edges for nodes that
        currently have no incident edges. This is the offline counterpart
        to bulk-document mode — corpora ingested via the fast path can earn
        their graph structure here instead of at write time.
        """
        stats = ConsolidationStats()

        nodes = self.storage.all_active_nodes(scope=scope)
        edges = self.storage.all_active_edges()
        if not nodes:
            return stats

        # v0.5 — batch edge induction for bulk-ingested corpora
        if induce_edges:
            self._induce_edges_for_isolated(
                nodes, edges, scope, induce_k, induce_gamma_threshold, stats,
            )
            # Refresh active edge list after induction
            edges = self.storage.all_active_edges()

        nodes_by_id = {n.id: n for n in nodes}
        regions = _build_regions(nodes, edges)
        regions_with_priority = [
            (_region_priority(r, nodes_by_id, edges), r) for r in regions
        ]
        regions_with_priority.sort(key=lambda x: -x[0])

        for _priority, region in regions_with_priority[:budget]:
            self._process_region(region, edges, sigma_0_squared, stats)
            stats.regions_processed += 1

        self.audit.append(
            "CONSOLIDATE", "global", "<bulk>",
            actor="consolidator",
            payload={
                "regions_processed": stats.regions_processed,
                "edges_pruned": stats.edges_pruned,
                "edges_refined": stats.edges_refined,
                "motifs_found": stats.motifs_found,
            },
        )
        return stats

    def _induce_edges_for_isolated(
        self,
        nodes: list[Node],
        existing_edges: list[Edge],
        scope: dict[str, Any] | None,
        k: int,
        gamma_threshold: float | None,
        stats: ConsolidationStats,
    ) -> None:
        """Induce Γ-edges for nodes with no incident edges.

        Used after bulk-document ingest to give the graph its structure.
        Uses the storage's vectorized topk_neighbors_for_gamma path, so cost
        is roughly O(N_isolated · matmul_over_N) per isolated node — at 10K
        nodes, ~1-2 sec on a laptop.
        """
        from recall.config import DEFAULT
        from recall.geometry.gamma import gamma_anti, gamma_score
        from recall.write.edge_classifier import classify_edge_type, signed_weight_for_type

        thresh = gamma_threshold if gamma_threshold is not None else DEFAULT.THRESH_GAMMA

        # Find nodes with no incident edges (isolated relative to existing edges)
        with_edges: set[str] = set()
        for e in existing_edges:
            with_edges.add(e.src_node_id)
            with_edges.add(e.dst_node_id)

        # Tenant comes from the storage (every node we touch shares it)
        tenant = self.storage.tenant if hasattr(self.storage, "tenant") else None

        for node in nodes:
            if node.id in with_edges:
                continue
            if node.f_embedding is None or node.b_embedding is None:
                continue
            # Find candidates by symmetric similarity
            neighbors = self.storage.topk_neighbors_for_gamma(
                node, scope or {}, k=k,
            )
            for nbr in neighbors:
                if nbr.id == node.id:
                    continue
                if nbr.f_embedding is None or nbr.b_embedding is None:
                    continue
                gij = gamma_score(
                    node.f_embedding, node.b_embedding,
                    nbr.f_embedding, nbr.b_embedding,
                )
                gji = gamma_score(
                    nbr.f_embedding, nbr.b_embedding,
                    node.f_embedding, node.b_embedding,
                )
                anti_ij = gamma_anti(
                    node.f_embedding, node.b_embedding,
                    nbr.f_embedding, nbr.b_embedding,
                )
                for src, dst, g in [(node, nbr, gij), (nbr, node, gji)]:
                    if abs(g) < thresh:
                        continue
                    edge_anti = anti_ij if src is node else -anti_ij
                    etype = classify_edge_type(src, dst, g, edge_anti)
                    weight = signed_weight_for_type(etype, g)
                    e = Edge(
                        tenant=tenant or src.tenant,
                        src_node_id=src.id,
                        dst_node_id=dst.id,
                        edge_type=etype,
                        weight=weight,
                        gamma_score=g,
                        gamma_anti=edge_anti,
                        s_squared=1.0,
                    )
                    self.storage.insert_edge(e)
                    self.audit.append(
                        "INDUCE_EDGE", "edge", e.id,
                        actor="consolidator",
                        payload={
                            "src": src.id, "dst": dst.id, "weight": weight,
                            "gamma": g,
                        },
                    )
                    stats.edges_induced += 1

    def _process_region(
        self,
        region: set[str],
        all_edges: list[Edge],
        sigma_0_squared: float,
        stats: ConsolidationStats,
    ) -> None:
        region_edges = [
            e for e in all_edges if e.src_node_id in region and e.dst_node_id in region
        ]
        if not region_edges:
            return

        # 0. Curvature-aware protection: bottleneck edges (Ollivier-Ricci κ < 0)
        # are essential for graph connectivity — protect them from BMRS pruning.
        try:
            from recall.graph.curvature import curvature_pruning_signal
            region_nodes = [
                n for n in self.storage.all_active_nodes() if n.id in region
            ]
            protected_edge_ids = set(
                curvature_pruning_signal(region_nodes, region_edges, threshold=0.7)
            )
        except Exception:
            protected_edge_ids = set()

        # 1. BMRS pruning (Wright-Igel-Selvan NeurIPS 2024 / Friston BMR 2011).
        # v0.7: corrected sign convention (prune when log BF > 0) and switched
        # the per-edge posterior variance to ``cosine_edge_variance`` —
        # calibrated for cosine-weighted edges. The legacy
        # ``estimate_edge_variance_from_gamma`` is retained only when the
        # edge has explicit s_squared metadata from a prior consolidate.
        for edge in region_edges:
            if edge.id in protected_edge_ids:
                # Curvature says this is a bottleneck — keep regardless of weight
                continue
            if edge.s_squared and edge.s_squared > 0 and edge.s_squared < 0.99:
                # An explicit s_squared was set previously by a refinement
                # pass — trust it.
                s_squared = edge.s_squared
            else:
                # Fresh edge: estimate from cosine-noise model.
                s_squared = cosine_edge_variance(
                    edge.weight, gamma_anti=edge.gamma_anti,
                )
            log_ratio = bmrs_log_evidence_ratio(
                w_hat=edge.weight, s_squared=s_squared, sigma_0_squared=sigma_0_squared
            )
            edge.bmrs_log_ratio = log_ratio
            # v0.7: prune when reduced model is preferred (log BF > 0).
            if log_ratio > 0:
                self.storage.deprecate_edge(edge.id, reason="bmrs_pruned")
                self.audit.append(
                    "PRUNE", "edge", edge.id,
                    actor="consolidator", reason="bmrs",
                    payload={"log_ratio": log_ratio, "weight": edge.weight},
                )
                stats.edges_pruned += 1

        # 2. Mean-field refinement on remaining (kept) edges.
        # v0.7: kept edges have log BF ≤ 0 (full model preferred).
        active_after_prune = [
            e for e in region_edges
            if e.bmrs_log_ratio is None or e.bmrs_log_ratio <= 0
        ]
        if active_after_prune:
            refined_count = mean_field_iterate(active_after_prune, T=5)
            stats.edges_refined += refined_count

        # 3. Motif extraction
        motifs = find_recurring_subgraphs(region, region_edges, min_occurrences=2)
        for m in motifs:
            self.audit.append(
                "MOTIF", "region", str(sorted(region))[:60],
                actor="consolidator",
                payload={"pattern": m["pattern"], "count": m["count"]},
            )
            stats.motifs_found += 1

        # 4. PMED scoring — region-level experience score
        try:
            from recall.consolidate.pmed_score import (
                compute_pmed_components, pmed_priority,
            )
            region_nodes = [
                n for n in self.storage.all_active_nodes() if n.id in region
            ]
            total_nodes = len(self.storage.all_active_nodes())
            comp = compute_pmed_components(region_nodes, region_edges, total_nodes)
            score = pmed_priority(comp)
            self.audit.append(
                "PMED_SCORE", "region", str(sorted(region))[:60],
                actor="consolidator",
                payload={
                    "score": score,
                    "D_RPD": comp.D_RPD, "DCR": comp.DCR, "P_syco": comp.P_syco,
                    "Q_corr": comp.Q_corr, "Q_eff": comp.Q_eff, "Q_rare": comp.Q_rare,
                },
            )
        except Exception:
            pass
