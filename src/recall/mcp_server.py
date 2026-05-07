"""Recall as an MCP (Model Context Protocol) server.

Exposes Recall's memory operations as MCP tools so any MCP-aware client
(Claude Desktop, Cursor, Cline, Continue, Zed, Windsurf) gets typed-edge
memory automatically — without any code change in those tools.

Tools exposed:
  - add_memory(text, scope?)    — write a memory (gated, dedup'd)
  - search_memory(query, k?)    — return top-k connected nodes
  - traverse_edges(query)       — return reasoning paths (typed-edge graph)
  - bounded_answer(query)       — generate an answer, structurally bounded
  - forget(node_id, reason)     — surgical delete
  - audit(target?)              — provenance trail
  - graph_health()              — spectral / curvature / persistence summary
  - consolidate(budget?)        — run sleep-time consolidator now

Run as stdio MCP server:
    python -m recall.mcp_server

Or via uvx:
    uvx recall-mcp

Compatible with Anthropic's MCP SDK (python).
"""
from __future__ import annotations

import os
from typing import Any

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
except ImportError as e:
    raise ImportError(
        "MCP server requires `pip install mcp`. "
        "Install with: `pip install recall[mcp]` or `pip install mcp`."
    ) from e

from recall import Memory


# Global memory tenants keyed by tenant id (default "user")
_memories: dict[str, Memory] = {}


def _build_embedder():
    """Auto-detect the best available embedder.

    Priority:
      1. ``RECALL_EMBEDDER=hash|tfidf|bge`` — explicit override
      2. BGEEmbedder if ``sentence-transformers`` is importable AND
         ``RECALL_EMBEDDER!=tfidf``
      3. TfidfEmbedder
      4. HashEmbedder fallback
    """
    pref = os.environ.get("RECALL_EMBEDDER", "auto").lower()
    if pref == "hash":
        from recall.embeddings import HashEmbedder
        return HashEmbedder()
    if pref == "tfidf":
        from recall.embeddings import TfidfEmbedder
        return TfidfEmbedder()
    if pref in ("auto", "bge"):
        try:
            from recall.embeddings import BGEEmbedder
            model = os.environ.get("RECALL_BGE_MODEL", "BAAI/bge-small-en-v1.5")
            return BGEEmbedder(model_name=model)
        except ImportError:
            if pref == "bge":
                # User explicitly asked for BGE — surface the install hint
                raise
    # auto / fallback
    try:
        from recall.embeddings import TfidfEmbedder
        return TfidfEmbedder()
    except ImportError:
        from recall.embeddings import HashEmbedder
        return HashEmbedder()


def _build_llm():
    """Auto-detect the best available LLM client.

    Priority:
      1. ``OPENAI_API_KEY`` set → OpenAIClient (model from
         ``RECALL_OPENAI_MODEL``, default ``gpt-4o-mini``)
      2. ``ANTHROPIC_API_KEY`` set → Anthropic client (planned, falls back)
      3. MockLLMClient (deterministic stub — used by default)
    """
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from recall.llm import OpenAIClient
            base_url = os.environ.get("OPENAI_BASE_URL")  # e.g. TokenRouter
            model = os.environ.get("RECALL_OPENAI_MODEL", "gpt-4o-mini")
            kwargs = {"model": model}
            if base_url:
                kwargs["base_url"] = base_url
            return OpenAIClient(**kwargs)
        except ImportError:
            pass
    from recall.llm import MockLLMClient
    return MockLLMClient()


def _get_mem(tenant: str = "user") -> Memory:
    if tenant not in _memories:
        db_dir = os.environ.get("RECALL_DB_DIR", os.path.expanduser("~/.recall"))
        os.makedirs(db_dir, exist_ok=True)
        _memories[tenant] = Memory(
            tenant=tenant,
            storage=f"sqlite://{db_dir}/{tenant}.db",
            embedder=_build_embedder(),
            llm=_build_llm(),
        )
    return _memories[tenant]


# Build MCP server
server = Server("recall")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="add_memory",
            description=(
                "Add a memory to Recall's typed-edge graph. The memory is "
                "junk-gated, deduplicated, and connected by Γ-scored edges to "
                "related existing memories. Returns the new node ids."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The memory text to store."},
                    "agent_response": {
                        "type": "string",
                        "description": "Optional agent reply to also embed.",
                        "default": "",
                    },
                    "scope": {
                        "type": "object",
                        "description": "Scope tags (project, team, etc.).",
                        "additionalProperties": {"type": "string"},
                        "default": {},
                    },
                    "tenant": {"type": "string", "default": "user"},
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="search_memory",
            description=(
                "Search Recall's memory graph. Returns connected reasoning "
                "subgraph (nodes + typed edges) for the query."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "scope": {"type": "object", "default": {}},
                    "k": {"type": "integer", "default": 5},
                    "mode": {
                        "type": "string",
                        "enum": ["path", "symmetric", "hybrid"],
                        "default": "path",
                    },
                    "tenant": {"type": "string", "default": "user"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="bounded_answer",
            description=(
                "Generate an answer bounded by Recall's retrieved subgraph. "
                "Hallucination-bounded — claims that don't trace to a path "
                "are flagged."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "scope": {"type": "object", "default": {}},
                    "bound": {
                        "type": "string",
                        "enum": ["soft", "strict", "off"],
                        "default": "soft",
                    },
                    "tenant": {"type": "string", "default": "user"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="forget",
            description="Surgically delete a memory by node_id. Audit log preserved.",
            inputSchema={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "reason": {"type": "string", "default": "user_request"},
                    "tenant": {"type": "string", "default": "user"},
                },
                "required": ["node_id"],
            },
        ),
        Tool(
            name="audit",
            description="Return provenance trail of operations for a target id (or all).",
            inputSchema={
                "type": "object",
                "properties": {
                    "target_id": {"type": "string", "default": ""},
                    "limit": {"type": "integer", "default": 50},
                    "tenant": {"type": "string", "default": "user"},
                },
            },
        ),
        Tool(
            name="graph_health",
            description=(
                "Compute graph-theoretic health metrics: spectral gap (λ_2), "
                "Cheeger bounds, persistent-homology summary, curvature stats."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tenant": {"type": "string", "default": "user"},
                },
            },
        ),
        Tool(
            name="consolidate",
            description=(
                "Run sleep-time consolidation now: BMRS pruning + mean-field + "
                "motif extraction + curvature-aware protection."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "budget": {"type": "integer", "default": 10},
                    "tenant": {"type": "string", "default": "user"},
                },
            },
        ),
        Tool(
            name="stats",
            description="One-liner stats: active nodes, edges, audit entries.",
            inputSchema={
                "type": "object",
                "properties": {"tenant": {"type": "string", "default": "user"}},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    tenant = arguments.get("tenant", "user")
    mem = _get_mem(tenant)

    try:
        if name == "add_memory":
            r = mem.observe(
                arguments["text"],
                arguments.get("agent_response", ""),
                scope=arguments.get("scope", {}),
            )
            return [TextContent(
                type="text",
                text=(
                    f"drawer={r.drawer_id}\n"
                    f"nodes_written={r.nodes_written}\n"
                    f"rejected={r.nodes_rejected}\n"
                    f"edges={r.edges_written}\n"
                    f"duplicate={r.drawer_was_duplicate}"
                ),
            )]

        if name == "search_memory":
            res = mem.recall(
                arguments["query"],
                scope=arguments.get("scope", {}),
                mode=arguments.get("mode", "path"),
                k=int(arguments.get("k", 5)),
            )
            lines = [f"mode={res.mode}", f"nodes={len(res.subgraph_nodes)}",
                     f"edges={len(res.subgraph_edges)}", ""]
            for n in res.subgraph_nodes:
                etype = n.role or "fact"
                lines.append(f"[{etype}] {n.text}")
            return [TextContent(type="text", text="\n".join(lines))]

        if name == "bounded_answer":
            gen = mem.bounded_generate(
                arguments["query"],
                scope=arguments.get("scope", {}),
                bound=arguments.get("bound", "soft"),
            )
            lines = [
                gen.text,
                "",
                f"--- bound={gen.bound_value}, flagged={len(gen.flagged_claims)}, "
                f"retrieved_nodes={len(gen.retrieved.subgraph_nodes)} ---",
            ]
            return [TextContent(type="text", text="\n".join(lines))]

        if name == "forget":
            r = mem.forget(arguments["node_id"], reason=arguments.get("reason", "user"))
            return [TextContent(
                type="text",
                text=f"deprecated_node={r.deprecated_node_id} cascaded_edges={r.deprecated_edge_ids}",
            )]

        if name == "audit":
            target = arguments.get("target_id", "")
            limit = int(arguments.get("limit", 50))
            entries = mem.audit.for_target(target) if target else mem.audit.since(mem._epoch_start())
            lines = []
            for e in entries[-limit:]:
                ts = e.timestamp.isoformat() if e.timestamp else "?"
                lines.append(f"[{e.seq}] {ts} {e.operation:20s} {e.target_type}/{e.target_id} {e.reason or ''}")
            return [TextContent(type="text", text="\n".join(lines) or "(no audit entries)")]

        if name == "graph_health":
            from recall.graph import (
                graph_health as gh,
                persistent_homology_summary,
                curvature_summary,
            )
            from recall.graph.spectral import graph_health
            nodes = mem.storage.all_active_nodes()
            edges = mem.storage.all_active_edges()
            spec = graph_health(nodes, edges)
            topo = persistent_homology_summary(nodes)
            curv = curvature_summary(nodes, edges)
            lines = ["=== Spectral ==="]
            for k, v in spec.items():
                lines.append(f"  {k}: {v}")
            lines.append("=== Topology (persistence) ===")
            for k, v in topo.items():
                lines.append(f"  {k}: {v}")
            lines.append("=== Curvature (Ollivier-Ricci) ===")
            for k, v in curv.items():
                lines.append(f"  {k}: {v}")
            return [TextContent(type="text", text="\n".join(lines))]

        if name == "consolidate":
            stats = mem.consolidate(budget=int(arguments.get("budget", 10)))
            return [TextContent(
                type="text",
                text=(
                    f"regions={stats.regions_processed} "
                    f"pruned={stats.edges_pruned} "
                    f"refined={stats.edges_refined} "
                    f"motifs={stats.motifs_found}"
                ),
            )]

        if name == "stats":
            return [TextContent(type="text", text=str(mem.stats()))]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"ERROR: {type(e).__name__}: {e}")]


async def serve_stdio() -> None:
    """Entry point for stdio MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    import asyncio
    asyncio.run(serve_stdio())


if __name__ == "__main__":
    main()
