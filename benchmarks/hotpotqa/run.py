"""HotpotQA — real public multi-hop QA benchmark.

Tests Recall on retrieval quality: given a multi-hop question and the
distractor passages, does Γ-walk surface the supporting facts?

Metrics:
  - support_recall@k: % of supporting passages in top-k retrieved
  - mrr: mean reciprocal rank of first supporting passage
  - latency: p50, p99 retrieval latency

This uses the public HuggingFace `hotpot_qa` distractor set. Each example
contains a question, an answer, and ~10 passages of which 2 are the gold
support and ~8 are distractors.

Usage:
    python benchmarks/hotpotqa/run.py [--n 20]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from recall import Memory, TfidfEmbedder


def passage_id(title: str, idx: int) -> str:
    """Stable id for a (title, sentence_idx) tuple."""
    return f"{title}::{idx}"


def run_one(example, k: int = 5) -> dict:
    """Process one HotpotQA example.

    Returns dict with retrieval metrics for this example.
    """
    question = example["question"]
    answer = example["answer"]
    context_titles = example["context"]["title"]
    context_sentences = example["context"]["sentences"]

    # Each (title, sent_idx) pair is a distinct passage
    # Supporting facts is a list of [title, sent_idx]
    sf_titles = example["supporting_facts"]["title"]
    sf_idxs = example["supporting_facts"]["sent_id"]
    supporting = {(t, i) for t, i in zip(sf_titles, sf_idxs)}

    mem = Memory(tenant=f"hpq_{example['id']}", embedder=TfidfEmbedder(dim=384))

    # Ingest each sentence as a separate observation, scoped by passage_id
    all_pids = []
    for title, sentences in zip(context_titles, context_sentences):
        for sent_idx, sent in enumerate(sentences):
            pid = passage_id(title, sent_idx)
            all_pids.append((pid, title, sent_idx))
            mem.observe(sent, "", scope={"title": title, "sent": sent_idx},
                        source="document")

    t0 = time.time()
    result = mem.recall(question, mode="symmetric", k=k)
    latency = time.time() - t0

    # Map retrieved nodes back to (title, sent_idx)
    retrieved_pairs: list[tuple[str, int]] = []
    for n in result.subgraph_nodes:
        title = n.scope.get("title", "")
        sent_idx = n.scope.get("sent", -1)
        retrieved_pairs.append((title, sent_idx))

    # Recall@k: how many supporting facts in top-k retrieved
    in_topk = supporting & set(retrieved_pairs)
    recall = len(in_topk) / len(supporting) if supporting else 0.0

    # MRR: rank of first supporting fact
    rank = None
    for i, p in enumerate(retrieved_pairs):
        if p in supporting:
            rank = i + 1
            break
    mrr = 1.0 / rank if rank else 0.0

    return {
        "id": example["id"],
        "n_passages": len(all_pids),
        "n_supporting": len(supporting),
        "recall_at_k": recall,
        "mrr": mrr,
        "latency_sec": latency,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of examples")
    ap.add_argument("--k", type=int, default=5, help="top-k for retrieval")
    args = ap.parse_args()

    print("=" * 70)
    print(f"HotpotQA distractor — {args.n} examples, top-{args.k} retrieval")
    print("=" * 70)

    try:
        from datasets import load_dataset
    except ImportError:
        print("Requires `pip install datasets`. Aborting.")
        return

    ds = load_dataset("hotpot_qa", "distractor", split=f"validation[:{args.n}]")

    results = []
    for i, ex in enumerate(ds):
        try:
            r = run_one(ex, k=args.k)
        except Exception as e:
            print(f"  [{i+1}/{args.n}] FAILED: {e}")
            continue
        results.append(r)
        print(f"  [{i+1}/{args.n}] recall@{args.k}={r['recall_at_k']:.2f}  "
              f"mrr={r['mrr']:.2f}  latency={r['latency_sec']:.2f}s  "
              f"({r['n_passages']} passages, {r['n_supporting']} supporting)")

    if not results:
        print("\nNo successful runs.")
        return

    avg_recall = sum(r["recall_at_k"] for r in results) / len(results)
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    latencies = sorted(r["latency_sec"] for r in results)
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[-1]

    print(f"\n=== Summary ===")
    print(f"  avg recall@{args.k}:  {avg_recall:.3f}")
    print(f"  avg MRR:        {avg_mrr:.3f}")
    print(f"  latency p50:    {p50:.2f}s")
    print(f"  latency p99:    {p99:.2f}s")
    print(f"\n  (BM25/cosine baseline on HotpotQA recall@5 ≈ 0.55-0.65 in published RAG papers)")


if __name__ == "__main__":
    main()
