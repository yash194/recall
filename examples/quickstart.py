"""Recall quickstart — three lines of memory.

Run:
    cd recall
    pip install -e .
    python examples/quickstart.py
"""
from recall import Memory


def main():
    mem = Memory(tenant="quickstart")

    # 1. Observe a few conversation turns
    mem.observe(
        "We tried Postgres LISTEN/NOTIFY for the queue but it kept losing messages under load.",
        "Confirmed — that's a known limitation.",
        scope={"project": "platform"},
    )
    mem.observe(
        "We switched to Redis Streams last Thursday and it's been stable.",
        "Good. Redis Streams gives durability and consumer groups.",
        scope={"project": "platform"},
    )
    mem.observe(
        "Decision: use Redis Streams as our queue going forward.",
        "Acknowledged.",
        scope={"project": "platform"},
    )

    print("Stats after observe:", mem.stats())

    # 2. Recall a path
    result = mem.recall(
        "What queue technology do we use, and why?",
        scope={"project": "platform"},
        mode="path",
    )
    print(f"\nRetrieved {len(result.subgraph_nodes)} nodes, {len(result.subgraph_edges)} edges.")
    for n in result.subgraph_nodes:
        print(f"  [{n.role or 'fact'}] {n.text}")

    # 3. Bounded generation
    gen = mem.bounded_generate(
        "What queue technology do we use, and why?",
        scope={"project": "platform"},
        bound="soft",
    )
    print(f"\nGenerated answer (mode=soft):")
    print(f"  {gen.text}")
    print(f"  flagged claims: {gen.flagged_claims}")
    print(f"  PAC-Bayes bound: {gen.bound_value}")

    # 4. Trace
    trace = mem.trace(gen)
    print(f"\nTrace: {len(trace.nodes)} nodes, {len(trace.drawers)} drawers,")
    print(f"  {len(trace.audit_entries)} audit entries")

    # 5. Forget
    if result.subgraph_nodes:
        target = result.subgraph_nodes[0]
        forget = mem.forget(target.id, reason="quickstart demo")
        print(f"\nForgot node {forget.deprecated_node_id}")
        print(f"  cascaded edges: {len(forget.deprecated_edge_ids)}")


if __name__ == "__main__":
    main()
