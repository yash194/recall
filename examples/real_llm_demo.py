"""Real-LLM end-to-end demo using TokenRouter / OpenRouter / OpenAI.

Set:
    export OPENAI_API_KEY=sk-...
    export OPENAI_BASE_URL=https://api.tokenrouter.com/v1   # or your endpoint

Then run:
    PYTHONPATH=src python examples/real_llm_demo.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recall import LLMQualityClassifier, Memory, RouterClient, TfidfEmbedder


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY (and OPENAI_BASE_URL if not openai.com).")
        return

    print("=" * 70)
    print("Recall — REAL LLM end-to-end demo")
    print("=" * 70)

    llm = RouterClient(model="openai/gpt-4o-mini")
    print(f"\nUsing LLM: openai/gpt-4o-mini via TokenRouter")

    mem = Memory(
        tenant="real_llm_demo",
        embedder=TfidfEmbedder(dim=384),
        llm=llm,
        use_llm_quality=True,
    )

    # 1. Junk-replay-style writes — see if LLM-driven quality classifier
    # rejects the bad ones
    print("\n--- Quality gate test (LLM-backed) ---")
    candidates = [
        ("user", "Sure, I can help with that.", "boilerplate"),
        ("user", "I am Claude, an AI assistant.", "system_leak"),
        ("user", "User John Doe is a Google engineer based in London.", "halluc_profile"),
        ("user", "We migrated our queue from Postgres LISTEN/NOTIFY to Redis Streams in March.", "genuine"),
        ("user", "The customer's preferred time zone is US/Pacific.", "genuine"),
        ("user", "All production deployments require approval from on-call.", "genuine"),
    ]
    for src, msg, label in candidates:
        r = mem.observe(src, msg, scope={"p": "demo"})
        promoted = len(r.nodes_written)
        rejected = len(r.nodes_rejected)
        verdict = "PROMOTED" if promoted else ("REJECTED" if rejected else "OTHER")
        print(f"  [{label:20s}] {verdict}  '{msg[:60]}'")

    print(f"\n  Final stats: {mem.stats()}")

    # 2. Path retrieval
    print("\n--- Path retrieval ---")
    result = mem.recall(
        "What did we change about our queue infrastructure?",
        scope={"p": "demo"},
        mode="path",
    )
    print(f"  Retrieved {len(result.subgraph_nodes)} nodes, {len(result.subgraph_edges)} edges")
    for n in result.subgraph_nodes[:5]:
        print(f"    [{n.role}] {n.text[:80]}")

    # 3. Bounded generation with REAL LLM
    print("\n--- bounded_generate (REAL LLM via TokenRouter) ---")
    gen = mem.bounded_generate(
        "What did we change about our queue infrastructure?",
        scope={"p": "demo"},
        bound="soft",
    )
    print(f"  Generated text:\n  {gen.text[:400]}")
    print(f"  Flagged claims: {len(gen.flagged_claims)}")
    print(f"  Composite bound: {gen.bound_value}")

    # 4. Audit trace
    print("\n--- Audit trace ---")
    trace = mem.trace(gen)
    print(f"  Trace: {len(trace.nodes)} nodes, {len(trace.drawers)} drawers, "
          f"{len(trace.audit_entries)} audit entries")

    # 5. Metrics
    print("\n--- Metrics ---")
    snap = mem.metrics.snapshot()
    for op, info in snap.get("latency_p", {}).items():
        print(f"  {op:25s} p50={info['p50']*1000:.1f}ms  count={info['count']}")
    for k, v in snap.get("counts", {}).items():
        if not k.endswith("_count"):
            print(f"  count.{k:25s} {v}")


if __name__ == "__main__":
    main()
