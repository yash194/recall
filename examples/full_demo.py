"""Full demo of Recall — showcases every component.

Run:
    PYTHONPATH=src python examples/full_demo.py
"""
from recall import Memory, TfidfEmbedder
from recall.integrations import LettaMemoryBackend


def demo():
    print("=" * 70)
    print("Recall — full feature demo")
    print("=" * 70)

    mem = Memory(tenant="demo_app", embedder=TfidfEmbedder(dim=384))

    print("\n--- 1. observe() — junk-gated, dedup'd, provenance-checked ---")
    # System prompt leak (rejected by provenance)
    r = mem.observe("(system)", "I am Claude, an AI assistant.", source="recall_artifact")
    print(f"  System leak attempt: skipped_recall_loop={r.skipped_recall_loop}")

    # Boilerplate (rejected by quality)
    r = mem.observe("user", "Sure, I can help with that.", source="conversation")
    print(f"  Boilerplate attempt: rejected={len(r.nodes_rejected)} promoted={len(r.nodes_written)}")

    # Genuine facts
    r1 = mem.observe(
        "We tried Postgres LISTEN/NOTIFY for the queue.",
        "AGENT: It lost messages under load.",
        scope={"project": "platform"},
    )
    r2 = mem.observe(
        "We switched to Redis Streams last Thursday.",
        "AGENT: Good - Redis Streams gives durability.",
        scope={"project": "platform"},
    )
    r3 = mem.observe(
        "Decision: use Redis Streams as our queue going forward.",
        "AGENT: Acknowledged.",
        scope={"project": "platform"},
    )
    print(f"  Genuine writes: {len(r1.nodes_written) + len(r2.nodes_written) + len(r3.nodes_written)} nodes promoted")

    # Duplicate (caught by drawer dedup)
    r_dup = mem.observe(
        "We switched to Redis Streams last Thursday.",
        "AGENT: Good - Redis Streams gives durability.",
        scope={"project": "platform"},
    )
    print(f"  Duplicate write: drawer_was_duplicate={r_dup.drawer_was_duplicate}")

    print(f"\n  Stats: {mem.stats()}")

    print("\n--- 2. recall() — connected reasoning subgraph ---")
    result = mem.recall(
        "What queue technology do we use, and why?",
        scope={"project": "platform"},
        mode="path",
    )
    print(f"  Mode: {result.mode}")
    print(f"  Retrieved {len(result.subgraph_nodes)} nodes, {len(result.subgraph_edges)} edges")
    print("  Nodes:")
    for n in result.subgraph_nodes:
        print(f"    [{n.role}] {n.text[:80]}")
    print("  Edges (with classified types):")
    for e in result.subgraph_edges[:5]:
        et = e.edge_type.value if hasattr(e.edge_type, "value") else str(e.edge_type)
        print(f"    --[{et}, w={e.weight:+.3f}]--")

    print("\n--- 3. bounded_generate() — hallucination-bounded answer ---")
    gen = mem.bounded_generate(
        "What queue technology do we use, and why?",
        scope={"project": "platform"},
        bound="soft",
    )
    print(f"  Generated {len(gen.text)} chars")
    print(f"  Flagged claims: {len(gen.flagged_claims)}")
    print(f"  PAC-Bayes bound: {gen.bound_value}")

    print("\n--- 4. trace() — full provenance ---")
    trace = mem.trace(gen)
    print(f"  Trace: {len(trace.nodes)} nodes, {len(trace.drawers)} drawers, "
          f"{len(trace.audit_entries)} audit entries")

    print("\n--- 5. consolidate() — sleep-time pruning + motif extraction ---")
    cstats = mem.consolidate(budget=10)
    print(f"  Regions processed: {cstats.regions_processed}")
    print(f"  Edges pruned (BMRS): {cstats.edges_pruned}")
    print(f"  Edges refined (mean-field): {cstats.edges_refined}")
    print(f"  Motifs found: {cstats.motifs_found}")

    print("\n--- 6. forget() — surgical, with audit ---")
    if result.subgraph_nodes:
        target = result.subgraph_nodes[0]
        f = mem.forget(target.id, reason="user requested")
        print(f"  Forgot node {f.deprecated_node_id}")
        print(f"  Cascaded {len(f.deprecated_edge_ids)} edges")

    print("\n--- 7. Letta integration (drop-in archival_memory) ---")
    letta_backend = LettaMemoryBackend(tenant="letta_demo")
    letta_backend.archival_memory_insert("User prefers Python over JavaScript for backend.")
    letta_backend.archival_memory_insert("User decided to deploy on AWS Fargate.")
    results = letta_backend.archival_memory_search("language preference", top_k=3)
    print(f"  Letta-style search returned {len(results)} memories")
    for r in results:
        print(f"    {r['text'][:80]}")

    print("\n--- 8. Metrics snapshot ---")
    snap = mem.metrics.snapshot()
    for op, info in snap.get("latency_p", {}).items():
        print(f"  {op:25s} p50={info['p50']*1000:.1f}ms  p99={info['p99']*1000:.1f}ms  count={info['count']}")
    for k, v in snap.get("counts", {}).items():
        print(f"  count.{k:25s} {v}")


if __name__ == "__main__":
    demo()
