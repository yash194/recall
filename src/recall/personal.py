"""Personal Recall — Recall as a personal knowledge graph CLI.

For the user who isn't building an AI app — they want a personal "second
brain" with reasoning paths, surgically-correctable memory, and bounded
hallucination. Replaces Obsidian/Logseq/Tana for users who want graph-based
retrieval.

CLI surface:

    recall me add "decided to use Postgres LISTEN/NOTIFY"
    recall me ask "what queue tech?"
    recall me trace
    recall me forget <node_id>
    recall me health
    recall me consolidate

Stores in ~/.recall/personal.db by default.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_DB = os.path.expanduser("~/.recall/personal.db")


def _get_memory(db_path: str | None = None):
    from recall import Memory
    db = db_path or DEFAULT_DB
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    return Memory(tenant="personal", storage=f"sqlite://{db}")


def cmd_add(args) -> int:
    mem = _get_memory(args.db)
    text = args.text
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if not text:
        print("No text provided. Pass as argument or via stdin.", file=sys.stderr)
        return 2
    scope = json.loads(args.scope) if args.scope else {}
    if args.tag:
        scope["tag"] = args.tag
    r = mem.observe("(me)", text, scope=scope, source="conversation")
    if r.drawer_was_duplicate:
        print(f"(duplicate) drawer={r.drawer_id[:12]}...")
        return 0
    if r.skipped_recall_loop:
        print("rejected (provenance gate)")
        return 0
    if not r.nodes_written:
        rejected = r.nodes_rejected[0][1] if r.nodes_rejected else "?"
        print(f"rejected ({rejected})")
        return 0
    for nid in r.nodes_written:
        print(f"+ {nid[:12]}...")
    print(f"({len(r.nodes_written)} new, {r.edges_written} edges)")
    return 0


def cmd_ask(args) -> int:
    mem = _get_memory(args.db)
    scope = json.loads(args.scope) if args.scope else {}
    if args.tag:
        scope["tag"] = args.tag

    if args.bounded:
        try:
            gen = mem.bounded_generate(args.query, scope=scope, bound="soft", k=args.k)
        except Exception as e:
            print(f"bounded_generate error: {e}", file=sys.stderr)
            return 2
        print(gen.text)
        if gen.flagged_claims:
            print(f"\n(flagged: {len(gen.flagged_claims)} unsupported claim(s))")
        if gen.bound_value is not None:
            print(f"(hallucination bound: {gen.bound_value:.3f})")
        return 0

    res = mem.recall(args.query, scope=scope, mode=args.mode, k=args.k)
    print(f"# {len(res.subgraph_nodes)} memories, {len(res.subgraph_edges)} connections")
    print()
    for n in res.subgraph_nodes:
        role = n.role or "fact"
        print(f"  [{role}] {n.text}")
        if args.verbose:
            print(f"        id={n.id[:12]}... quality={n.quality_score:.2f}")
    return 0


def cmd_trace(args) -> int:
    mem = _get_memory(args.db)
    if args.target:
        entries = mem.audit.for_target(args.target)
    else:
        entries = mem.audit.since(mem._epoch_start())
    for e in entries[-args.limit:]:
        ts = e.timestamp.isoformat() if e.timestamp else "?"
        print(f"[{e.seq:4d}] {ts}  {e.operation:>22s}  {e.target_type}/{e.target_id[:12]}  {e.reason or ''}")
    return 0


def cmd_forget(args) -> int:
    mem = _get_memory(args.db)
    r = mem.forget(args.node_id, reason=args.reason or "personal cli")
    if r.error:
        print(f"error: {r.error}", file=sys.stderr)
        return 2
    print(f"forgot {r.deprecated_node_id[:12]}... ({len(r.deprecated_edge_ids)} edges cascaded)")
    return 0


def cmd_health(args) -> int:
    mem = _get_memory(args.db)
    from recall.graph import (
        graph_health,
        persistent_homology_summary,
        curvature_summary,
    )
    nodes = mem.storage.all_active_nodes()
    edges = mem.storage.all_active_edges()
    spec = graph_health(nodes, edges)
    topo = persistent_homology_summary(nodes)
    curv = curvature_summary(nodes, edges)
    print("=== Memory health ===")
    print(f"  nodes: {spec.get('n_nodes', 0)}")
    print(f"  edges: {spec.get('n_edges', 0)}")
    print(f"  density: {spec.get('edge_density', 0):.2f}")
    print()
    print("=== Connectivity (spectral) ===")
    if "spectral_gap_lambda2" in spec:
        print(f"  λ_2 (spectral gap): {spec['spectral_gap_lambda2']:.4f}")
        print(f"  Cheeger lower bound: {spec['cheeger_lower_bound']:.4f}")
        print(f"  Cheeger upper bound: {spec['cheeger_upper_bound']:.4f}")
    print()
    print("=== Topology (persistent homology) ===")
    if "betti_0" in topo:
        print(f"  β_0 (components): {topo['betti_0']}")
        if topo.get("betti_1") is not None:
            print(f"  β_1 (loops): {topo['betti_1']}")
        print(f"  backend: {topo.get('backend', '?')}")
    print()
    print("=== Curvature (Ollivier-Ricci) ===")
    if "n_edges" in curv and curv["n_edges"]:
        print(f"  mean κ: {curv['mean_curvature']:.3f}")
        print(f"  bottleneck edges (κ<0): {curv['n_bottleneck_edges']}")
        print(f"  community edges (κ>0): {curv['n_community_edges']}")
    return 0


def cmd_consolidate(args) -> int:
    mem = _get_memory(args.db)
    stats = mem.consolidate(budget=args.budget)
    print(f"regions processed: {stats.regions_processed}")
    print(f"edges pruned (BMRS): {stats.edges_pruned}")
    print(f"edges refined (mean-field): {stats.edges_refined}")
    print(f"motifs found: {stats.motifs_found}")
    return 0


def cmd_ingest(args) -> int:
    """Bulk-ingest a file or directory."""
    mem = _get_memory(args.db)
    p = Path(args.path)
    if not p.exists():
        print(f"not found: {p}", file=sys.stderr)
        return 2
    files = []
    if p.is_dir():
        for ext in ("*.md", "*.txt", "*.org"):
            files.extend(p.rglob(ext))
    else:
        files = [p]
    total = 0
    for f in files:
        try:
            text = f.read_text()
        except Exception:
            continue
        # Split into paragraphs
        for para in text.split("\n\n"):
            para = para.strip()
            if len(para) < 20:
                continue
            r = mem.observe(str(f.name), para, scope={"file": str(f)},
                            source="document")
            total += len(r.nodes_written)
    print(f"ingested {total} memories from {len(files)} file(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="recall me",
        description="Personal knowledge graph with reasoning paths.",
    )
    p.add_argument("--db", default=DEFAULT_DB, help="path to personal db")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="add a memory")
    p_add.add_argument("text", nargs="?", help="memory text (or stdin)")
    p_add.add_argument("--scope", help="JSON scope dict")
    p_add.add_argument("--tag", help="convenience tag")
    p_add.set_defaults(func=cmd_add)

    p_ask = sub.add_parser("ask", help="search / query memory")
    p_ask.add_argument("query")
    p_ask.add_argument("--scope", help="JSON scope dict")
    p_ask.add_argument("--tag", help="filter by tag")
    p_ask.add_argument("--mode", choices=["path", "symmetric", "hybrid"], default="path")
    p_ask.add_argument("-k", type=int, default=10)
    p_ask.add_argument("--bounded", action="store_true",
                       help="generate a bounded answer (requires LLM)")
    p_ask.add_argument("-v", "--verbose", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_trace = sub.add_parser("trace", help="show audit log")
    p_trace.add_argument("--target", help="filter by target id")
    p_trace.add_argument("--limit", type=int, default=20)
    p_trace.set_defaults(func=cmd_trace)

    p_forget = sub.add_parser("forget", help="delete a memory")
    p_forget.add_argument("node_id")
    p_forget.add_argument("--reason")
    p_forget.set_defaults(func=cmd_forget)

    p_health = sub.add_parser("health", help="memory graph health")
    p_health.set_defaults(func=cmd_health)

    p_consolidate = sub.add_parser("consolidate", help="run sleep-time consolidation")
    p_consolidate.add_argument("--budget", type=int, default=20)
    p_consolidate.set_defaults(func=cmd_consolidate)

    p_ingest = sub.add_parser("ingest", help="bulk-ingest a file or directory")
    p_ingest.add_argument("path")
    p_ingest.set_defaults(func=cmd_ingest)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
