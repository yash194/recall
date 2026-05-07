"""The public Memory API — the three-line surface.

Example:
    from recall import Memory

    mem = Memory(tenant="my_app")
    mem.observe(user_msg, agent_msg, scope={"project": "P1"})
    answer = mem.bounded_generate(query, scope={"project": "P1"})
    trace = mem.trace(answer)
    mem.forget(node_id, reason="user said outdated")
"""
from __future__ import annotations

import time
from typing import Any, Literal

from recall.audit.log import AuditLog
from recall.bound.pac_bayes import compute_bound_estimate
from recall.bound.rag_bound import composite_hallucination_bound
from recall.bound.support import extract_claims, structurally_supported, support_score
from recall.config import Config, DEFAULT
from recall.consolidate.scheduler import ConsolidationStats, Consolidator
from recall.core.storage import SQLiteStorage, Storage
from recall.embeddings import Embedder, HashEmbedder, auto_embedder
from recall.llm import LLMClient, MockLLMClient
from recall.retrieval.intent import classify_intent
from recall.retrieval.linearize import linearize_subgraph
from recall.retrieval.pcsf import pcsf_extract
from recall.retrieval.pcst import pcst_extract, pcst_extract_networkx
from recall.retrieval.router import reciprocal_rank_fuse, route as graph_aware_route
from recall.retrieval.walk import gamma_walk
from recall.telemetry import Metrics
from recall.types import (
    ForgetResult,
    GenerationResult,
    Node,
    RetrievalResult,
    Trace,
    WriteResult,
)
from recall.write.pipeline import WritePipeline


class HallucinationBlocked(Exception):
    """Raised by bounded_generate(mode='strict') when generation contains
    structurally-unsupported claims."""

    def __init__(self, claims: list[str], retrieved: RetrievalResult, bound: float | None):
        self.claims = claims
        self.retrieved = retrieved
        self.bound = bound
        super().__init__(
            f"Hallucination bound violated: {len(claims)} unsupported claim(s). "
            f"PAC-Bayes bound: {bound}"
        )


_BOUNDED_PROMPT_TEMPLATE = """\
You are answering using ONLY the structured memory below. Do not introduce
information not supported by the context. If the context is insufficient, say so.

Context:
{context}

Query: {query}

Answer:"""


class Memory:
    """The public Recall API.

    Parameters:
        tenant: a unique memory boundary (user / org / agent identity).
        storage: 'sqlite://path/to.db' or ':memory:' (default).
        embedder: an Embedder; default HashEmbedder (deterministic, zero-dep).
        llm: an LLMClient; default MockLLMClient (deterministic, zero-dep).
        config: Config overrides for thresholds.
    """

    def __init__(
        self,
        tenant: str,
        storage: str | Storage = ":memory:",
        embedder: Embedder | None = None,
        llm: LLMClient | None = None,
        config: Config | None = None,
        support_method: str = "tfidf",
        retrieval_algo: str = "greedy",
        quality_classifier: Any = None,
        use_llm_quality: bool = False,
        use_llm_splitter: bool = False,
    ):
        self.tenant = tenant
        self.cfg = config or DEFAULT
        # Default to TF-IDF for real semantics; fall back to hash for tests
        self.embedder = embedder or auto_embedder(prefer="tfidf", dim=self.cfg.EMBED_DIM, config=self.cfg)
        self.llm = llm or MockLLMClient()
        self.support_method = support_method
        self.retrieval_algo = retrieval_algo

        # Quality classifier (template-based by default; LLM-based optional)
        if quality_classifier is None and use_llm_quality:
            from recall.llm_quality import LLMQualityClassifier
            quality_classifier = LLMQualityClassifier(self.llm, config=self.cfg)
        self._quality_classifier = quality_classifier

        if isinstance(storage, str):
            db_path = storage.replace("sqlite://", "") if storage.startswith("sqlite://") else storage
            self.storage = SQLiteStorage(
                tenant=tenant, db_path=db_path, embed_dim=self.embedder.dim
            )
        else:
            self.storage = storage

        self.audit = AuditLog(self.storage)
        self.metrics = Metrics()
        # v0.7: the LLM splitter is opt-in. When self.llm is a real network LLM
        # (OpenAI / TokenRouter / etc.) every observe() would otherwise pay one
        # round-trip to split single-sentence inputs that don't need splitting,
        # tanking write throughput from ~80/s to <1/s. The deterministic
        # sentence splitter handles 95% of observations correctly. Pass
        # `use_llm_splitter=True` to opt in for paragraph-scale ingest.
        splitter_llm = self.llm if use_llm_splitter else None
        self.pipeline = WritePipeline(
            tenant=tenant,
            storage=self.storage,
            embedder=self.embedder,
            audit=self.audit,
            llm=splitter_llm,
            config=self.cfg,
            quality_classifier=self._quality_classifier,
        )
        # Keep the generation LLM available for bounded_generate even when
        # the splitter doesn't use it.
        self._generation_llm = self.llm

    # --- WRITE ---

    def observe(
        self,
        user_msg: str,
        agent_msg: str = "",
        scope: dict[str, Any] | None = None,
        source: str = "conversation",
        fast: bool | None = None,
    ) -> WriteResult:
        """Store one conversation turn. Quality-gated, dedup'd, provenance-checked.

        v0.5: pass `fast=True` (or use a source in `BULK_MODE_SOURCES`) for
        bulk-document ingestion that skips Γ-edge induction. Edges can be
        induced in a later consolidation pass.
        """
        with self.metrics.time("observe"):
            r = self.pipeline.observe(
                user_msg, agent_msg, scope=scope, source=source, fast=fast
            )
        if r.skipped_recall_loop:
            self.metrics.increment("rejected_provenance")
        if r.drawer_was_duplicate:
            self.metrics.increment("skipped_drawer_dup")
        self.metrics.increment("nodes_rejected_quality", by=len(r.nodes_rejected))
        self.metrics.increment("nodes_promoted", by=len(r.nodes_written))
        return r

    def bulk_observe(
        self,
        texts: list[str],
        scope: dict[str, Any] | None = None,
        source: str = "bulk_document",
    ) -> list[WriteResult]:
        """Ingest a batch of texts via the bulk-document fast path (v0.5).

        Convenience wrapper over `observe(text, "", fast=True, source=...)`.
        Returns one WriteResult per text. Skip Γ-edge induction; run
        `consolidate(induce_edges=True)` later if you want graph structure.
        """
        return [
            self.observe(t, "", scope=scope, source=source, fast=True)
            for t in texts
        ]

    # --- READ ---

    def recall(  # noqa: D401  (verb form is intentional)
        self,
        query: str,
        scope: dict[str, Any] | None = None,
        mode: Literal["path", "symmetric", "hybrid", "auto", "multi_hop"] = "auto",
        k: int = 10,
        depth: int = 4,
    ) -> RetrievalResult:
        """Retrieve a connected subgraph for the query.

        mode='symmetric' falls back to vector RAG (cosine over node summaries).
        mode='path' performs Γ-walk seeded by cosine.
        mode='hybrid' fuses symmetric + path via RRF.
        mode='multi_hop' does HippoRAG/GraphRAG-style entity-expanded retrieval
            for compositional questions whose gold answer is not directly
            cosine-similar to the question text.
        """
        # v0.7: handle multi_hop mode upfront (no Γ-walk, just iterative
        # cosine + entity expansion).
        if mode == "multi_hop":
            with self.metrics.time("recall"):
                from recall.retrieval.multi_hop import multi_hop_recall
                if hasattr(self.embedder, "embed_symmetric"):
                    s_q = self.embedder.embed_symmetric(query)
                else:
                    f_q, b_q = self.embedder.embed_dual(query)
                    s_q = (f_q + b_q) / 2.0
                scope = scope or {}
                seeds = self.storage.topk_cosine(s_q, scope, k=k)
                expanded = multi_hop_recall(
                    self.storage,
                    self.embedder,
                    query,
                    scope=scope,
                    k_init=max(5, k),
                    k_per_entity=3,
                    k_final=max(k, 10),
                    hops=2,
                )
                return RetrievalResult(
                    query=query, seeds=seeds, paths=[],
                    subgraph_nodes=expanded, subgraph_edges=[], mode="multi_hop",
                )
        with self.metrics.time("recall"):
            scope = scope or {}
            # v0.6: use raw symmetric embedding for retrieval (no prompt
            # prefix). Falls back to (f+b)/2 if the embedder doesn't have
            # embed_symmetric (HashEmbedder, TfidfEmbedder default).
            if hasattr(self.embedder, "embed_symmetric"):
                s_q = self.embedder.embed_symmetric(query)
            else:
                f_q, b_q = self.embedder.embed_dual(query)
                s_q = (f_q + b_q) / 2.0

            # Symmetric seed (always cheap)
            seeds = self.storage.topk_cosine(s_q, scope, k=k)

            # mode='auto' — use the graph-aware router (Adaptive-RAG style)
            # to decide between symmetric / walk_short / walk_deep / hybrid
            # based on seed dispersion + query type.
            if mode == "auto":
                from recall.retrieval.router import _DIRECTIONAL_HINTS, _FACTUAL_HINTS
                q_lc = (query or "").lower()
                has_directional = any(h in q_lc for h in _DIRECTIONAL_HINTS)
                has_factual = any(h in q_lc for h in _FACTUAL_HINTS)
                # v0.5 cheap-path: clear factual cue + no directional cue
                # short-circuits to symmetric. v0.6: also short-circuit when
                # the graph has too few edges per node for walking (uses
                # cached n_active_edges() — O(1) when adjacency cache
                # exists). Keeps p50 retrieval latency O(matmul).
                if has_factual and not has_directional:
                    mode = "symmetric"
                elif (
                    hasattr(self.storage, "n_active_edges")
                    and self.storage.n_active_edges() == 0
                ):
                    # No edges — walks would be no-ops
                    mode = "symmetric"
                else:
                    all_nodes = self.storage.all_active_nodes(scope=scope or None)
                    all_edges_active = self.storage.all_active_edges()
                    seed_ids = [s.id for s in seeds]
                    routed = graph_aware_route(
                        query, all_nodes, all_edges_active, seed_ids,
                    )
                    if routed == "symmetric":
                        mode = "symmetric"
                    elif routed == "walk_deep":
                        mode = "path"
                        depth = max(depth, 4)
                    elif routed == "walk_short":
                        mode = "path"
                        depth = min(depth, 2)
                    else:  # hybrid
                        mode = "hybrid"

            if mode == "hybrid":
                # New hybrid: run symmetric AND path, fuse via RRF
                sym_result = RetrievalResult(
                    query=query, seeds=seeds, paths=[],
                    subgraph_nodes=seeds, subgraph_edges=[], mode="symmetric",
                )
                # Fall through to path-mode logic, then RRF-fuse
                mode = "path"
                _hybrid_seeds_for_fusion = seeds
            else:
                _hybrid_seeds_for_fusion = None

            if mode == "symmetric":
                return RetrievalResult(
                    query=query, seeds=seeds, paths=[],
                    subgraph_nodes=seeds, subgraph_edges=[], mode="symmetric",
                )

            # mode in {"path", "directional"} → Γ-walk + PCST extraction
            all_paths = []
            for seed in seeds:
                paths = gamma_walk(
                    self.storage, seed,
                    depth=depth,
                    weight_threshold=self.cfg.THRESH_WALK,
                    beam_width=self.cfg.BEAM_WIDTH,
                )
                all_paths.extend(paths)

            seed_id_list = [s.id for s in seeds]
            if self.retrieval_algo == "networkx":
                sub_nodes, sub_edges = pcst_extract_networkx(all_paths, budget=self.cfg.BUDGET_SUBGRAPH)
            elif self.retrieval_algo == "pcsf":
                sub_nodes, sub_edges = pcsf_extract(all_paths, budget=self.cfg.BUDGET_SUBGRAPH)
            elif self.retrieval_algo == "ppr":
                # Personalized PageRank on the signed typed-edge graph
                # (HippoRAG / Area 1 of the math sweep — load-bearing)
                from recall.graph.spectral import personalized_pagerank
                all_nodes = self.storage.all_active_nodes(scope=scope or None)
                all_edges_active = self.storage.all_active_edges()
                seed_ids = [s.id for s in seeds]
                ppr_scores = personalized_pagerank(
                    all_nodes, all_edges_active, seed_ids,
                    damping=0.85, n_iter=50,
                )
                # Take top-k by PPR score
                ranked = sorted(
                    all_nodes, key=lambda n: -ppr_scores.get(n.id, 0.0)
                )[: max(k, 5)]
                ranked_ids = {n.id for n in ranked}
                sub_nodes = ranked
                sub_edges = [
                    e for e in all_edges_active
                    if e.src_node_id in ranked_ids and e.dst_node_id in ranked_ids
                ]
            else:
                # v0.7: pass query seeds as required terminals so the high-cosine
                # match always survives PCST extraction.
                sub_nodes, sub_edges = pcst_extract(
                    all_paths,
                    budget=self.cfg.BUDGET_SUBGRAPH,
                    must_include=seed_id_list,
                )
                # If walk produced no usable subgraph (e.g., zero-edge graph at
                # cold-start), fall back to the cosine seeds directly. This is
                # the right thing semantically — we know which nodes match the
                # query best — and stops path-mode from returning empty results.
                if not sub_nodes:
                    sub_nodes = list(seeds)

            # If we entered via 'hybrid', fuse symmetric + path rankings
            if _hybrid_seeds_for_fusion is not None:
                path_ids = [n.id for n in (sub_nodes or seeds)]
                sym_ids = [n.id for n in _hybrid_seeds_for_fusion]
                fused_ids = reciprocal_rank_fuse([path_ids, sym_ids])
                # Build node list in fused order
                node_lookup = {n.id: n for n in (sub_nodes or seeds) + _hybrid_seeds_for_fusion}
                fused_nodes = [node_lookup[i] for i in fused_ids if i in node_lookup][:k]
                return RetrievalResult(
                    query=query,
                    seeds=seeds,
                    paths=all_paths,
                    subgraph_nodes=fused_nodes,
                    subgraph_edges=sub_edges,
                    mode="hybrid",
                )

            return RetrievalResult(
                query=query,
                seeds=seeds,
                paths=all_paths,
                subgraph_nodes=sub_nodes if sub_nodes else seeds,
                subgraph_edges=sub_edges,
                mode="path",
            )

    # --- READ + GENERATE ---

    def bounded_generate(
        self,
        query: str,
        scope: dict[str, Any] | None = None,
        bound: Literal["strict", "soft", "off"] = "strict",
        k: int = 10,
        depth: int = 4,
        mode: Literal["path", "symmetric", "hybrid", "auto"] = "hybrid",
    ) -> GenerationResult:
        """Retrieve and generate a response bounded by structural support.

        v0.7: default mode flipped from `"path"` to `"hybrid"`. Path-only
        retrieval can drop the high-prize seed during PCST extraction when
        the connected subgraph's optimum doesn't include it; hybrid fuses
        symmetric and path rankings via reciprocal-rank fusion so the seed
        always reaches the prompt. Empirically: 5/5 needle hit-rate vs 1/5
        with path-only on the v2 stress test.
        """
        timer = self.metrics.time("bounded_generate")
        timer.__enter__()
        retrieved = self.recall(query, scope=scope, mode=mode, k=k, depth=depth)

        # Build drawer lookup for linearization
        drawer_lookup: dict[str, str] = {}
        for n in retrieved.subgraph_nodes:
            for did in n.drawer_ids:
                d = self.storage.get_drawer(did)
                if d:
                    drawer_lookup[did] = d.text

        context = linearize_subgraph(
            retrieved.subgraph_nodes,
            retrieved.subgraph_edges,
            drawer_lookup=drawer_lookup,
        )

        # Generate
        prompt = _BOUNDED_PROMPT_TEMPLATE.format(context=context or "(no context)", query=query)
        raw = self.llm.complete(prompt, max_tokens=512)

        # Check structural support per claim
        flagged: list[str] = []
        for claim in extract_claims(raw):
            if not structurally_supported(
                claim,
                retrieved.subgraph_nodes,
                retrieved.subgraph_edges,
                self.storage,
                weight_threshold=self.cfg.THRESH_WALK,
                method=self.support_method,
            ):
                flagged.append(claim)

        # Compute bound estimate (telemetry)
        # Compute composite RAG + spectral bound (Zhang 2025 + arxiv 2508.19366)
        n_active = max(1, len(self.storage.all_active_nodes()))
        # Heuristic: distortion = fraction of unsupported claims
        distortion = (
            len(flagged) / max(1, len(extract_claims(raw))) if raw else 0.5
        )
        # Heuristic loss = fraction of flagged claims (proxy)
        avg_loss = distortion
        composite = composite_hallucination_bound(
            avg_path_loss=avg_loss,
            n_retrieved=max(1, len(retrieved.subgraph_nodes)),
            retrieval_distortion=distortion,
            n_edges_in_subgraph=len(retrieved.subgraph_edges),
            n_nodes_in_subgraph=len(retrieved.subgraph_nodes),
            embedding_dim=self.embedder.dim,
            delta=self.cfg.DELTA,
        )
        bound_value = composite["composite"]
        # Also keep PAC-Bayes telemetry on the side for ablation comparisons
        pac_bayes_estimate = compute_bound_estimate(
            n_paths=max(1, len(retrieved.paths)),
            avg_path_loss=avg_loss,
            avg_tandem_loss=0.25,
            n_training_samples=n_active,
            delta=self.cfg.DELTA,
        )

        result = GenerationResult(
            query=query,
            text=raw,
            retrieved=retrieved,
            flagged_claims=flagged,
            bound_value=bound_value,
            blocked=False,
        )

        if bound == "strict" and flagged:
            result.blocked = True
            self.audit.append(
                "GENERATION_BLOCKED",
                "query",
                query,
                payload={"flagged": flagged, "bound": bound_value},
            )
            timer.__exit__(None, None, None)
            self.metrics.increment("hallucinations_blocked")
            raise HallucinationBlocked(flagged, retrieved, bound_value)

        self.audit.append(
            "GENERATE",
            "query",
            query,
            payload={
                "retrieved_nodes": retrieved.node_ids(),
                "retrieved_edges": retrieved.edge_ids(),
                "flagged": flagged,
                "bound": bound_value,
                "raw_len": len(raw),
            },
        )
        timer.__exit__(None, None, None)
        return result

    # --- AUDIT ---

    def trace(self, generation: GenerationResult) -> Trace:
        """Return the full provenance trail for a generation."""
        retrieved = generation.retrieved
        drawers = []
        for n in retrieved.subgraph_nodes:
            for did in n.drawer_ids:
                d = self.storage.get_drawer(did)
                if d:
                    drawers.append(d)
        audit_entries = []
        for n in retrieved.subgraph_nodes:
            audit_entries.extend(self.audit.for_target(n.id))
        for e in retrieved.subgraph_edges:
            audit_entries.extend(self.audit.for_target(e.id))
        return Trace(
            nodes=retrieved.subgraph_nodes,
            edges=retrieved.subgraph_edges,
            drawers=drawers,
            bound_value=generation.bound_value,
            generated_text=generation.text,
            flagged_claims=generation.flagged_claims,
            audit_entries=audit_entries,
        )

    # --- FORGET ---

    def forget(
        self, node_id: str, reason: str = "user_request", actor: str = "user"
    ) -> ForgetResult:
        """Surgically deprecate a node and its incident edges, with audit log."""
        with self.metrics.time("forget"):
            node = self.storage.get_node(node_id)
            if node is None:
                return ForgetResult(
                    deprecated_node_id=None, deprecated_edge_ids=[], error="not_found"
                )

            deprecated_edges: list[str] = []
            for edge in self.storage.get_edges_from(node_id) + self.storage.get_edges_to(node_id):
                if edge.deprecated_at is None:
                    self.storage.deprecate_edge(edge.id, reason="cascading_node_forget")
                    deprecated_edges.append(edge.id)

            self.storage.deprecate_node(node_id, reason)
            self.audit.append(
                "FORGET", "node", node_id,
                actor=actor, reason=reason,
                payload={"cascaded_edges": deprecated_edges},
            )
            return ForgetResult(
                deprecated_node_id=node_id,
                deprecated_edge_ids=deprecated_edges,
            )

    # --- CONSOLIDATE (Tier-3 sleep-time) ---

    def consolidate(
        self,
        budget: int = 10,
        scope: dict[str, Any] | None = None,
        sigma_0_squared: float = 1.0,
        induce_edges: bool = False,
        induce_k: int = 6,
    ) -> ConsolidationStats:
        """Run sleep-time consolidation.

        Pipeline:
          1. (v0.5) Optional batch Γ-edge induction for isolated nodes —
             pass `induce_edges=True` after bulk-document ingest.
          2. BMRS edge pruning (Wright-Igel-Selvan NeurIPS 2024).
          3. Mean-field GNN refinement.
          4. Motif extraction.
        """
        consolidator = Consolidator(self.storage, self.audit)
        return consolidator.run(
            budget=budget,
            scope=scope,
            sigma_0_squared=sigma_0_squared,
            induce_edges=induce_edges,
            induce_k=induce_k,
        )

    # --- INTROSPECTION ---

    def stats(self) -> dict[str, int]:
        """Quick health snapshot."""
        nodes = self.storage.all_active_nodes()
        edges = self.storage.all_active_edges()
        return {
            "active_nodes": len(nodes),
            "active_edges": len(edges),
            "audit_entries": len(self.audit.since(self._epoch_start())),
        }

    def _epoch_start(self):
        from datetime import datetime, timezone

        return datetime(1970, 1, 1, tzinfo=timezone.utc)

    def close(self) -> None:
        self.storage.close()
