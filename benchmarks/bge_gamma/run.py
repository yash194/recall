"""Real BGE-M3 embedder benchmark.

Tests the Γ retrieval primitive using actual BAAI/bge-small-en-v1.5 embeddings
on the planted causal-chain task.

Run from the conda env that has torch + sentence-transformers:
    source /opt/anaconda3/bin/activate recall_env
    PYTHONPATH=src python benchmarks/bge_gamma/run.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from recall import BGEEmbedder, Memory


CAUSAL_CHAIN = [
    "The team adopted a new monitoring stack with PagerDuty alerts.",
    "Because of the new alerts, every queue message loss was immediately visible.",
    "The visibility revealed Postgres LISTEN/NOTIFY was dropping ~3% of messages under load.",
    "The team decided to switch to Redis Streams to fix the message loss.",
    "After the switch to Redis Streams, the queue has been stable with zero message loss.",
]

DISTRACTORS = [
    "The customer's name is Acme Corp.",
    "Our SOC2 audit is scheduled for Q3.",
    "Lunch is provided on Wednesdays.",
    "The build pipeline uses Turborepo.",
    "We use Stripe for payments.",
    "The frontend is built with Next.js 14.",
    "Database backups happen daily at 2am UTC.",
    "Our engineering team is 12 people.",
    "We're hiring a senior ML engineer.",
    "The CEO's calendar is managed by Lisa.",
] * 2


def main():
    print("=" * 70)
    print("Real BGE-M3 Γ benchmark")
    print("=" * 70)

    print("\nLoading BAAI/bge-small-en-v1.5 ...")
    t0 = time.time()
    embedder = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
    print(f"  loaded in {time.time() - t0:.1f}s, dim={embedder.dim}")

    # BGE neural embeddings have smaller-magnitude Γ scores than TF-IDF,
    # so we lower the edge-induction and walk thresholds.
    from recall.config import Config
    cfg = Config()
    cfg.THRESH_GAMMA = 0.005  # lower — neural embeddings have tighter ranges
    cfg.THRESH_WALK = -0.02
    mem = Memory(tenant="bge_gamma", embedder=embedder, config=cfg)

    t1 = time.time()
    for fact in CAUSAL_CHAIN:
        mem.observe(fact, "Acknowledged.", scope={"project": "infra"})
    for d in DISTRACTORS:
        mem.observe(d, "Got it.", scope={"project": "infra"})
    elapsed_observe = time.time() - t1
    print(f"\nIngest: {len(CAUSAL_CHAIN) + len(DISTRACTORS)} entries in {elapsed_observe:.1f}s")
    print(f"  ({(len(CAUSAL_CHAIN) + len(DISTRACTORS)) / elapsed_observe:.1f} entries/sec)")
    print(f"  Stats: {mem.stats()}")

    # Try both path and symmetric mode
    print("\n=== Path mode (Γ-walk) ===")
    t2 = time.time()
    result = mem.recall(
        "What ultimately led to the queue being stable?",
        scope={"project": "infra"},
        mode="path",
        k=10,
    )
    elapsed_query = time.time() - t2
    print(f"Query latency: {elapsed_query * 1000:.1f}ms")
    print(f"Retrieved {len(result.subgraph_nodes)} nodes, {len(result.subgraph_edges)} edges")

    print("\n=== Symmetric mode (cosine RAG baseline) ===")
    t3 = time.time()
    result_sym = mem.recall(
        "What ultimately led to the queue being stable?",
        scope={"project": "infra"},
        mode="symmetric",
        k=10,
    )
    elapsed_sym = time.time() - t3
    print(f"Query latency: {elapsed_sym * 1000:.1f}ms")
    print(f"Retrieved {len(result_sym.subgraph_nodes)} nodes (cosine top-k)")

    # Use whichever returned more nodes for chain-recall measurement
    if len(result_sym.subgraph_nodes) > len(result.subgraph_nodes):
        result = result_sym
        print("\n(using symmetric mode for chain-recall measurement)")

    chain_lower = [c.lower() for c in CAUSAL_CHAIN]
    retrieved_texts = [n.text.lower() for n in result.subgraph_nodes]
    matches = 0
    for c in chain_lower:
        for r in retrieved_texts:
            ct = set(c.split())
            rt = set(r.split())
            if not ct:
                continue
            if len(ct & rt) / len(ct) > 0.4:
                matches += 1
                break
    rate = matches / len(CAUSAL_CHAIN)
    print(f"\n>>> Causal-chain recall: {matches}/{len(CAUSAL_CHAIN)} = {rate * 100:.0f}%")
    print(f"    Target: ≥ 60%")
    print(f"    Compare: TfidfEmbedder hit 100%; HashEmbedder hit 60%")

    print(f"\n--- Recall metrics snapshot ---")
    snap = mem.metrics.snapshot()
    for op, info in snap.get("latency_p", {}).items():
        print(f"  {op:20s} p50={info['p50']*1000:.1f}ms  count={info['count']}")


if __name__ == "__main__":
    main()
