"""LettaMemoryBackend — drop-in archival-memory backend for Letta.

Letta's tiered memory exposes:
  - core_memory_append(content)
  - core_memory_replace(old, new)
  - archival_memory_insert(content)
  - archival_memory_search(query, top_k)

We wrap Recall to provide the archival_* operations with substantially better
behavior:
  - junk-gated writes
  - directional retrieval (Γ-walk)
  - bounded generation (separate API)
  - audit log of all memory operations

This is a *behavioral* replacement, not a binary one. Letta agents that point
their archival memory tool at this class get all of the above for free.

Usage:
    from recall.integrations import LettaMemoryBackend
    backend = LettaMemoryBackend(tenant="agent_42")
    backend.archival_memory_insert("User prefers Python over Java.")
    results = backend.archival_memory_search("language preference", top_k=5)
"""
from __future__ import annotations

from typing import Any

from recall import Memory


class LettaMemoryBackend:
    """Drop-in for Letta's archival_memory_* tool group.

    Returns plain dicts so it can be JSON-serialized to the tool boundary.
    """

    def __init__(
        self,
        tenant: str,
        storage: str = ":memory:",
        **memory_kwargs: Any,
    ):
        self.memory = Memory(tenant=tenant, storage=storage, **memory_kwargs)

    # --- Letta archival memory protocol ---

    def archival_memory_insert(self, content: str, scope: dict | None = None) -> dict:
        """Insert a memory; junk-gated, dedup'd, provenance-checked."""
        result = self.memory.observe("(letta_insert)", content, scope=scope or {},
                                     source="conversation")
        return {
            "ok": result.drawer_id is not None,
            "drawer_id": result.drawer_id,
            "nodes_written": result.nodes_written,
            "nodes_rejected": [{"id": nid, "reason": r} for nid, r in result.nodes_rejected],
            "skipped_duplicate": result.drawer_was_duplicate,
        }

    def archival_memory_search(self, query: str, top_k: int = 5,
                                scope: dict | None = None) -> list[dict]:
        """Path-based search. Returns top_k nodes with text + provenance."""
        result = self.memory.recall(query, scope=scope or {}, mode="path", k=top_k)
        out = []
        for n in result.subgraph_nodes:
            out.append({
                "id": n.id,
                "text": n.text,
                "role": n.role,
                "scope": n.scope,
            })
        return out[:top_k]

    def archival_memory_forget(self, memory_id: str, reason: str = "letta") -> dict:
        """Surgically forget a stored memory."""
        result = self.memory.forget(memory_id, reason=reason, actor="letta_agent")
        return {
            "ok": result.error is None,
            "deprecated_node_id": result.deprecated_node_id,
            "deprecated_edge_ids": result.deprecated_edge_ids,
            "error": result.error,
        }

    def archival_memory_count(self) -> int:
        return self.memory.stats()["active_nodes"]

    # --- Bonus: bounded generation as a tool ---

    def archival_memory_bounded_generate(
        self, query: str, scope: dict | None = None, bound: str = "soft"
    ) -> dict:
        """Generate an answer bounded by the memory subgraph."""
        gen = self.memory.bounded_generate(query, scope=scope or {}, bound=bound)
        return {
            "text": gen.text,
            "flagged_claims": gen.flagged_claims,
            "bound_value": gen.bound_value,
            "blocked": gen.blocked,
            "retrieved_node_count": len(gen.retrieved.subgraph_nodes),
        }
