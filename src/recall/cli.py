"""Recall CLI.

Two namespaces:

  ``recall <subcmd>``       — low-level developer commands
                              (ingest / inspect / forget / audit).
  ``recall me <subcmd>``    — high-level personal commands
                              (add / ask / health / trace / consolidate).

The ``recall me`` namespace is what the README documents as the personal
memory CLI. v0.7 implements every subcommand the README advertises so the
documented surface and the actual surface match.

Default DB: ``~/.recall/personal.db`` (used by ``recall me``).
Default tenant for personal namespace: ``me``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from recall import Memory


# ---------- shared helpers ----------

def _default_personal_db() -> str:
    home = os.path.expanduser("~/.recall")
    Path(home).mkdir(parents=True, exist_ok=True)
    return os.path.join(home, "personal.db")


def _make_memory(args, *, default_tenant: str = "default") -> Memory:
    tenant = getattr(args, "tenant", None) or default_tenant
    db = getattr(args, "db", None) or "./recall.db"
    return Memory(tenant=tenant, storage=f"sqlite://{db}")


def _make_personal_memory(args) -> Memory:
    tenant = getattr(args, "tenant", None) or "me"
    db = getattr(args, "db", None) or _default_personal_db()
    return Memory(tenant=tenant, storage=f"sqlite://{db}")


# ---------- low-level subcommands ----------

def cmd_ingest(args) -> int:
    mem = _make_memory(args)
    if not args.input:
        text = sys.stdin.read()
        mem.observe("(stdin)", text, scope=json.loads(args.scope) if args.scope else {})
        print(mem.stats())
        return 0
    p = Path(args.input)
    if not p.exists():
        print(f"input not found: {p}", file=sys.stderr)
        return 2
    text = p.read_text()
    result = mem.observe(p.name, text, scope=json.loads(args.scope) if args.scope else {})
    print(json.dumps({
        "drawer_id": result.drawer_id,
        "nodes_written": result.nodes_written,
        "nodes_rejected": result.nodes_rejected,
        "edges_written": result.edges_written,
    }, indent=2))
    return 0


def cmd_inspect(args) -> int:
    mem = _make_memory(args)
    stats = mem.stats()
    print(json.dumps(stats, indent=2))
    if args.node:
        node = mem.storage.get_node(args.node)
        if node:
            print(f"\nNode {node.id}:")
            print(f"  text: {node.text[:200]}")
            print(f"  role: {node.role}")
            print(f"  quality: {node.quality_score:.3f} ({node.quality_status})")
            print(f"  active: {node.is_active()}")
        else:
            print(f"node not found: {args.node}", file=sys.stderr)
            return 2
    return 0


def cmd_forget(args) -> int:
    mem = _make_memory(args)
    result = mem.forget(args.node, reason=args.reason or "cli", actor=args.actor)
    print(json.dumps({
        "deprecated_node_id": result.deprecated_node_id,
        "deprecated_edge_ids": result.deprecated_edge_ids,
        "error": result.error,
    }, indent=2))
    return 0 if result.error is None else 2


def cmd_audit(args) -> int:
    mem = _make_memory(args)
    if args.export:
        print(mem.audit.export_jsonl())
        return 0
    if args.target:
        entries = mem.audit.for_target(args.target)
    else:
        entries = mem.audit.since(mem._epoch_start())
    for e in entries:
        ts = e.timestamp.isoformat() if e.timestamp else "?"
        print(f"[{e.seq:>4}] {ts}  {e.operation:>20}  {e.target_type}/{e.target_id}  {e.reason or ''}")
    return 0


# ---------- `recall me` subcommands ----------

def cmd_me_add(args) -> int:
    """Add a memory. Reads from positional args or stdin."""
    mem = _make_personal_memory(args)
    text = " ".join(args.text) if args.text else sys.stdin.read().strip()
    if not text:
        print("usage: recall me add 'memory text' [--scope '{\"key\":\"val\"}']", file=sys.stderr)
        return 2
    scope = json.loads(args.scope) if args.scope else {}
    r = mem.observe(text, "", scope=scope)
    if not r.nodes_written:
        if r.drawer_was_duplicate:
            print(json.dumps({"status": "duplicate", "drawer_id": r.drawer_id}, indent=2))
        else:
            print(json.dumps({"status": "rejected", "rejected": r.nodes_rejected}, indent=2))
        return 0
    print(json.dumps({
        "status": "added",
        "drawer_id": r.drawer_id,
        "nodes_written": r.nodes_written,
        "edges_written": r.edges_written,
    }, indent=2))
    return 0


def cmd_me_ask(args) -> int:
    """Ask a question — retrieve relevant memories or run bounded_generate.

    By default returns the top-k retrieved memories. With ``--bound`` it runs
    bounded_generate and returns the structurally-checked answer plus the
    hallucination bound and any flagged claims.
    """
    mem = _make_personal_memory(args)
    query = " ".join(args.query) if args.query else sys.stdin.read().strip()
    if not query:
        print("usage: recall me ask 'your question'", file=sys.stderr)
        return 2
    if args.bound:
        try:
            r = mem.bounded_generate(
                query, bound=args.bound, k=args.k, mode=args.mode,
            )
        except Exception as e:
            # HallucinationBlocked or downstream
            print(json.dumps({"status": "blocked", "error": str(e)[:300]}, indent=2))
            return 0
        out = {
            "query": query,
            "text": r.text,
            "bound_value": r.bound_value,
            "flagged_claims": r.flagged_claims,
            "blocked": r.blocked,
            "retrieved_node_ids": [n.id for n in r.retrieved.subgraph_nodes],
        }
        print(json.dumps(out, indent=2))
        return 0
    # Plain retrieval
    res = mem.recall(query, k=args.k, mode=args.mode)
    out = {
        "query": query,
        "mode": res.mode,
        "n_nodes": len(res.subgraph_nodes),
        "n_edges": len(res.subgraph_edges),
        "results": [
            {"id": n.id, "role": n.role, "text": n.text[:300]}
            for n in res.subgraph_nodes
        ],
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_me_health(args) -> int:
    """Spectral / topology / curvature diagnostics."""
    mem = _make_personal_memory(args)
    nodes = mem.storage.all_active_nodes()
    edges = mem.storage.all_active_edges()
    out: dict = {
        "tenant": mem.tenant,
        "nodes": len(nodes),
        "edges": len(edges),
        "edges_per_node": (len(edges) / max(len(nodes), 1)),
    }
    if not nodes:
        print(json.dumps(out, indent=2))
        return 0
    try:
        from recall.graph.spectral import graph_health
        out["spectral"] = graph_health(nodes, edges)
    except Exception as e:
        out["spectral_error"] = str(e)[:200]
    try:
        from recall.graph.topology import persistent_homology_summary
        out["topology"] = persistent_homology_summary(nodes)
    except Exception as e:
        out["topology_error"] = str(e)[:200]
    try:
        from recall.graph.curvature import curvature_summary
        out["curvature"] = curvature_summary(nodes, edges)
    except Exception as e:
        out["curvature_error"] = str(e)[:200]
    try:
        from recall.graph.sheaf import inconsistency_score
        out["sheaf"] = inconsistency_score(nodes, edges)
    except Exception as e:
        out["sheaf_error"] = str(e)[:200]
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_me_trace(args) -> int:
    """Print the audit log (or filtered slice).

    With ``--target NODE_ID`` only entries about that node.  With ``--limit``
    only the most recent N entries.  ``--export`` dumps as JSONL.
    """
    mem = _make_personal_memory(args)
    if args.export:
        print(mem.audit.export_jsonl())
        return 0
    if args.target:
        entries = mem.audit.for_target(args.target)
    else:
        entries = mem.audit.since(mem._epoch_start())
    if args.limit:
        entries = entries[-args.limit:]
    for e in entries:
        ts = e.timestamp.isoformat() if e.timestamp else "?"
        print(
            f"[{e.seq:>4}] {ts}  {e.operation:>20}  "
            f"{e.target_type}/{e.target_id[:12]}  {e.reason or ''}"
        )
    return 0


def cmd_me_consolidate(args) -> int:
    """Run sleep-time consolidation (BMRS pruning + motifs)."""
    mem = _make_personal_memory(args)
    sigma = args.sigma_0_squared
    stats = mem.consolidate(
        budget=args.budget,
        sigma_0_squared=sigma,
        induce_edges=args.induce_edges,
    )
    out = {
        "regions_processed": stats.regions_processed,
        "edges_pruned": stats.edges_pruned,
        "edges_refined": stats.edges_refined,
        "edges_induced": stats.edges_induced,
        "motifs_found": stats.motifs_found,
        "nodes_merged": stats.nodes_merged,
    }
    print(json.dumps(out, indent=2))
    return 0


# ---------- argument parser ----------

def _build_me_parser(parent: argparse._SubParsersAction) -> None:
    p_me = parent.add_parser("me", help="personal-memory subcommands")
    me_sub = p_me.add_subparsers(dest="me_cmd", required=True)

    p_add = me_sub.add_parser("add", help="add a memory")
    p_add.add_argument("text", nargs="*", help="memory text (default: stdin)")
    p_add.add_argument("--scope", help='JSON scope dict, e.g. \'{"project":"x"}\'')
    p_add.set_defaults(func=cmd_me_add)

    p_ask = me_sub.add_parser("ask", help="retrieve memories or run bounded_generate")
    p_ask.add_argument("query", nargs="*", help="question text (default: stdin)")
    p_ask.add_argument("--k", type=int, default=8, help="top-k to retrieve")
    p_ask.add_argument(
        "--mode",
        choices=["auto", "symmetric", "path", "hybrid", "multi_hop"],
        default="auto",
        help="retrieval mode",
    )
    p_ask.add_argument(
        "--bound", choices=["soft", "strict", "off"], default=None,
        help="if set, run bounded_generate with the given bound mode",
    )
    p_ask.set_defaults(func=cmd_me_ask)

    p_health = me_sub.add_parser("health", help="spectral / topology / curvature")
    p_health.set_defaults(func=cmd_me_health)

    p_trace = me_sub.add_parser("trace", help="audit log of memory operations")
    p_trace.add_argument("--target", help="filter by target id (drawer / node / edge)")
    p_trace.add_argument("--limit", type=int, default=0, help="last N entries (0 = all)")
    p_trace.add_argument("--export", action="store_true", help="dump as JSONL")
    p_trace.set_defaults(func=cmd_me_trace)

    p_cons = me_sub.add_parser("consolidate", help="run sleep-time consolidation")
    p_cons.add_argument("--budget", type=int, default=10)
    p_cons.add_argument(
        "--sigma-0-squared", type=float, default=1.0,
        dest="sigma_0_squared", help="BMRS prior variance (higher = prune more)",
    )
    p_cons.add_argument(
        "--induce-edges", action="store_true",
        help="also induce edges for isolated nodes (Γ-batch)",
    )
    p_cons.set_defaults(func=cmd_me_consolidate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="recall", description="Recall — memory layer for AI agents"
    )
    parser.add_argument("--tenant", default=None, help="memory tenant")
    parser.add_argument("--db", default=None, help="path to SQLite db")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest a text file or stdin")
    p_ingest.add_argument("input", nargs="?", help="path to text file (default: stdin)")
    p_ingest.add_argument("--scope", help="JSON scope dict")
    p_ingest.set_defaults(func=cmd_ingest)

    p_inspect = sub.add_parser("inspect", help="inspect memory state")
    p_inspect.add_argument("--node", help="show one node's details")
    p_inspect.set_defaults(func=cmd_inspect)

    p_forget = sub.add_parser("forget", help="forget a node by id")
    p_forget.add_argument("node", help="node id to forget")
    p_forget.add_argument("--reason", help="reason text")
    p_forget.add_argument("--actor", default="user", help="who is forgetting")
    p_forget.set_defaults(func=cmd_forget)

    p_audit = sub.add_parser("audit", help="view audit log")
    p_audit.add_argument("--target", help="filter by target id")
    p_audit.add_argument("--export", action="store_true", help="export as JSONL")
    p_audit.set_defaults(func=cmd_audit)

    _build_me_parser(sub)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
