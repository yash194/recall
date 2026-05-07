"""HotpotQA with real BGE-small-en-v1.5 embedder.

Run from the conda env that has torch + sentence-transformers:
    source /opt/anaconda3/bin/activate recall_env
    PYTHONPATH=src python benchmarks/hotpotqa_bge/run.py --n 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def passage_id(title: str, idx: int) -> str:
    return f"{title}::{idx}"


def run_one(example, embedder, k: int = 5, algo: str = "greedy", mode: str = "symmetric") -> dict:
    from recall import Memory
    from recall.config import Config

    cfg = Config()
    cfg.THRESH_GAMMA = 0.005
    cfg.THRESH_WALK = -0.02

    question = example["question"]
    sf_titles = example["supporting_facts"]["title"]
    sf_idxs = example["supporting_facts"]["sent_id"]
    supporting = {(t, i) for t, i in zip(sf_titles, sf_idxs)}

    mem = Memory(tenant=f"hpq_bge_{example['id']}", embedder=embedder, config=cfg, retrieval_algo=algo)

    context_titles = example["context"]["title"]
    context_sentences = example["context"]["sentences"]
    for title, sentences in zip(context_titles, context_sentences):
        for sent_idx, sent in enumerate(sentences):
            mem.observe(sent, "", scope={"title": title, "sent": sent_idx},
                        source="document")

    t0 = time.time()
    result = mem.recall(question, mode=mode, k=k)
    latency = time.time() - t0

    retrieved_pairs = [(n.scope.get("title", ""), n.scope.get("sent", -1))
                       for n in result.subgraph_nodes]
    in_topk = supporting & set(retrieved_pairs)
    recall = len(in_topk) / len(supporting) if supporting else 0.0

    rank = None
    for i, p in enumerate(retrieved_pairs):
        if p in supporting:
            rank = i + 1
            break
    mrr = 1.0 / rank if rank else 0.0
    return {
        "id": example["id"],
        "n_passages": sum(len(s) for s in context_sentences),
        "n_supporting": len(supporting),
        "recall_at_k": recall,
        "mrr": mrr,
        "latency_sec": latency,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--algo", choices=["greedy", "networkx", "pcsf", "ppr"], default="greedy")
    ap.add_argument("--mode", choices=["symmetric", "path", "hybrid"], default="symmetric")
    args = ap.parse_args()

    print("=" * 70)
    print(f"HotpotQA distractor with BGE-small-en-v1.5 — n={args.n}, k={args.k}, "
          f"algo={args.algo}, mode={args.mode}")
    print("=" * 70)

    from recall import BGEEmbedder
    print("\nLoading BAAI/bge-small-en-v1.5 ...")
    embedder = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
    print(f"  loaded; dim={embedder.dim}\n")

    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split=f"validation[:{args.n}]")

    results = []
    for i, ex in enumerate(ds):
        try:
            r = run_one(ex, embedder, k=args.k, algo=args.algo, mode=args.mode)
        except Exception as e:
            print(f"  [{i+1}/{args.n}] FAILED: {str(e)[:120]}")
            continue
        results.append(r)
        print(f"  [{i+1}/{args.n}] recall@{args.k}={r['recall_at_k']:.2f}  "
              f"mrr={r['mrr']:.2f}  latency={r['latency_sec']*1000:.0f}ms")

    if not results:
        print("\nNo successful runs.")
        return

    avg_recall = sum(r["recall_at_k"] for r in results) / len(results)
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    latencies = sorted(r["latency_sec"] for r in results)
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[-1]

    print(f"\n=== Summary (real BGE) ===")
    print(f"  avg recall@{args.k}:  {avg_recall:.3f}")
    print(f"  avg MRR:        {avg_mrr:.3f}")
    print(f"  latency p50:    {p50*1000:.1f}ms")
    print(f"  latency p99:    {p99*1000:.1f}ms")
    print(f"\n  (BM25/cosine baselines on HotpotQA recall@5 ≈ 0.55-0.65 in published RAG papers)")
    print(f"  (TfidfEmbedder baseline: 0.578 at n=20)")


if __name__ == "__main__":
    main()
