"""MemoryAgentBench (ICLR 2026).

Four competencies tested:

  1. Accurate_Retrieval        — basic facts in long context
  2. Test_Time_Learning        — picking up new facts during a session
  3. Long_Range_Understanding  — multi-hop across distant memories
  4. Conflict_Resolution       — actually a multi-hop fact-consolidation
                                 split (factconsolidation_mh_6k); the
                                 name is misleading — questions are
                                 compositional 3-hop, not contradiction-
                                 override.

v0.7 honest finding: Recall's fine-grained sentence-split design is at a
*structural disadvantage* on this benchmark. Each example provides a
single 26K-char context with 100 numbered facts and no double-newlines.
BM25 / Cosine RAG ingest this as ONE chunk and trivially retrieve the
whole context (the gold answer is somewhere inside it, scored 1.0).
Recall's pipeline splits the chunk into ~50 sentence-coalesced nodes so
the gold answer ends up in 1 of 50 nodes — and when the question is a
3-hop composition whose answer-token isn't in the question, cosine
top-5 over fine-grained nodes is unreliable.

We do NOT claim Recall beats BM25/Cosine here. Recall's typed-edge
design helps with conversation-memory tasks where facts arrive
incrementally and need to be linked over time; it does not help with
document-QA where the right behavior is "keep the chunk together".

Dataset: ai-hyz/MemoryAgentBench on HuggingFace.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines_comparison"))

from baselines import BM25Index, CosineRAG, RecallAdapter


def split_context_into_memories(context: str, chunk_size: int = 800) -> list[str]:
    """Split context into chunks for memory ingestion."""
    chunks = []
    paragraphs = [p.strip() for p in context.split("\n\n") if len(p.strip()) > 30]
    cur = []
    cur_len = 0
    for p in paragraphs:
        cur.append(p)
        cur_len += len(p)
        if cur_len > chunk_size:
            chunks.append("\n\n".join(cur))
            cur = []
            cur_len = 0
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def normalize(s: str) -> str:
    import re
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def evaluate_one(example: dict, system, k: int = 5) -> dict:
    context = example.get("context", "")
    questions = example.get("questions", [])
    answers = example.get("answers", [])

    # Ingest context as memories
    chunks = split_context_into_memories(context)
    for i, chunk in enumerate(chunks):
        system.add(chunk, scope={"example_id": str(id(example)), "chunk": i})

    # Each example has multiple questions; we measure mean recall
    recalls = []
    for q, gold in zip(questions, answers):
        if not q or not gold:
            continue
        hits = system.search(q, k=k, scope_filter={"example_id": str(id(example))})
        if not hits:
            recalls.append(0.0)
            continue
        # Normalize gold
        gold_text = gold if isinstance(gold, str) else (gold[0] if gold else "")
        gold_n = normalize(gold_text)
        if not gold_n:
            continue
        # Substring match in any retrieved chunk
        hit_text = " || ".join(normalize(h.text) for h in hits)
        # Token-level: at least 50% of gold tokens in retrieved
        gold_toks = set(gold_n.split())
        hit_toks = set(hit_text.split())
        coverage = len(gold_toks & hit_toks) / max(len(gold_toks), 1)
        recalls.append(1.0 if coverage > 0.6 else (coverage if coverage > 0.3 else 0.0))

    if not recalls:
        return {"score": 0.0, "n_questions": 0, "n_chunks": len(chunks)}
    return {
        "score": sum(recalls) / len(recalls),
        "n_questions": len(recalls),
        "n_chunks": len(chunks),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="examples per split")
    ap.add_argument(
        "--splits", nargs="+",
        default=["Accurate_Retrieval", "Conflict_Resolution"],
        choices=["Accurate_Retrieval", "Test_Time_Learning",
                 "Long_Range_Understanding", "Conflict_Resolution"],
    )
    ap.add_argument(
        "--systems", nargs="+",
        default=["bm25", "cosine", "recall"],
        choices=["bm25", "cosine", "recall"],
    )
    args = ap.parse_args()

    print("=" * 78)
    print(f"MemoryAgentBench (ICLR 2026) — n={args.n} per split")
    print("=" * 78)

    from datasets import load_dataset

    results: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    name_map = {"bm25": "BM25", "cosine": "Cosine RAG (BGE)", "recall": "Recall (auto)"}

    for split in args.splits:
        print(f"\n--- {split} ---")
        try:
            ds = load_dataset("ai-hyz/MemoryAgentBench", split=split, streaming=True)
        except Exception as e:
            print(f"  failed to load split: {e}")
            continue

        examples = list(ds.take(args.n))
        print(f"  loaded {len(examples)} examples")

        for sys_name in args.systems:
            scores = []
            t0 = time.time()
            for i, ex in enumerate(examples):
                # Fresh system per example
                if sys_name == "bm25":
                    system = BM25Index()
                elif sys_name == "cosine":
                    system = CosineRAG()
                elif sys_name == "recall":
                    from recall import Memory, BGEEmbedder
                    from recall.config import Config
                    cfg = Config()
                    cfg.THRESH_GAMMA = 0.005
                    cfg.THRESH_WALK = -0.02
                    mem = Memory(
                        tenant=f"mab_{split}_{i}",
                        embedder=BGEEmbedder("BAAI/bge-small-en-v1.5"),
                        config=cfg,
                    )
                    system = RecallAdapter(mem)

                try:
                    r = evaluate_one(ex, system)
                    scores.append(r["score"])
                except Exception as e:
                    print(f"    [{sys_name}/{i}] failed: {str(e)[:120]}")
                    continue
            elapsed = time.time() - t0
            if scores:
                avg = sum(scores) / len(scores)
                results[split][sys_name] = scores
                print(f"  {name_map[sys_name]:<20} score = {avg:.3f}  ({len(scores)} ex, {elapsed:.1f}s)")

    # Final table
    print("\n" + "=" * 78)
    print("MemoryAgentBench summary")
    print("=" * 78)
    print(f"\n{'split':<32} {'BM25':>10} {'Cosine':>10} {'Recall':>10}")
    print("-" * 78)
    for split in args.splits:
        row_vals = [split]
        for sn in ["bm25", "cosine", "recall"]:
            scores = results[split].get(sn, [])
            row_vals.append(f"{sum(scores)/len(scores):.3f}" if scores else "---")
        print(f"{row_vals[0]:<32} {row_vals[1]:>10} {row_vals[2]:>10} {row_vals[3]:>10}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump({split: {sn: scores for sn, scores in by_sys.items()}
                   for split, by_sys in results.items()}, f, indent=2)
    print(f"\nSaved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
