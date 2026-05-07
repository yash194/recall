"""LongMemEval benchmark — the canonical 500-question long-conversation memory eval.

ICLR 2025. https://arxiv.org/abs/2410.10813
Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

Five abilities tested:
  - single-session-user
  - single-session-assistant
  - multi-session
  - knowledge-update
  - temporal-reasoning
  - abstention   (does the agent say "I don't know" when it doesn't?)

For each question, the agent has a "haystack" of N sessions and must answer
based on what's in those sessions. We measure retrieval quality by:
  - support_recall@k: of the gold supporting sessions, how many appear in top-k?
  - mrr: mean reciprocal rank of first supporting session

We compare:
  - CosineRAG (BGE)
  - BM25
  - Recall (auto-router mode)
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


DATA_PATH = "/tmp/longmemeval/longmemeval_s_cleaned.json"


def session_to_text(session: list[dict]) -> str:
    """Render a session (list of role/content turns) as a single block of text."""
    lines = []
    for turn in session:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def evaluate_one(example: dict, system, k: int = 5) -> dict:
    question = example["question"]
    qid = example["question_id"]
    qtype = example.get("question_type", "?")
    gold_session_ids = set(example.get("answer_session_ids") or [])

    # Ingest all haystack sessions
    # NB: per-example tenant is the isolation boundary — no scope filter
    # needed (and Recall's scope filter does exact JSON match, not subset).
    sessions = example["haystack_sessions"]
    session_ids = example["haystack_session_ids"]
    for sid, session in zip(session_ids, sessions):
        text = session_to_text(session)
        if len(text) > 8000:
            text = text[:8000]
        # Put session_id in scope so we can retrieve it back from node metadata
        system.add(text, scope={"session_id": sid}, meta={"session_id": sid})

    t0 = time.time()
    hits = system.search(question, k=k)  # no scope filter — tenant isolation suffices
    latency = time.time() - t0

    # Prefer session_id from meta (cosine/BM25 store it there); fall back to
    # scope (Recall stores tenant scope in scope, with non-session-id meta).
    retrieved_session_ids = []
    for h in hits:
        sid = None
        if h.meta:
            sid = h.meta.get("session_id")
        if sid is None and h.scope:
            sid = h.scope.get("session_id")
        retrieved_session_ids.append(sid)
    retrieved_set = set(retrieved_session_ids)

    # Metrics
    if gold_session_ids:
        in_topk = gold_session_ids & retrieved_set
        recall = len(in_topk) / len(gold_session_ids)
    else:
        # abstention questions have no gold support — skip retrieval metrics
        recall = float("nan")

    rank = None
    for i, sid in enumerate(retrieved_session_ids):
        if sid in gold_session_ids:
            rank = i + 1
            break
    mrr = 1.0 / rank if rank else 0.0

    return {
        "qid": qid,
        "qtype": qtype,
        "n_sessions": len(sessions),
        "n_gold": len(gold_session_ids),
        "recall_at_k": recall,
        "mrr": mrr,
        "latency_sec": latency,
    }


def stratified_sample(data: list[dict], n: int, seed: int = 2026) -> list[dict]:
    """Sample n examples spread across all question types (LongMemEval has 6).

    If `n` is < 6 takes one of each in order encountered. Otherwise distributes
    proportionally with a deterministic seed.
    """
    import random
    from collections import defaultdict
    by_type: dict[str, list[dict]] = defaultdict(list)
    for ex in data:
        by_type[ex.get("question_type", "?")].append(ex)
    rng = random.Random(seed)
    types = sorted(by_type.keys())
    out: list[dict] = []
    per_type = max(1, n // len(types))
    for t in types:
        bucket = by_type[t]
        rng.shuffle(bucket)
        out.extend(bucket[:per_type])
    # Fill remaining slots round-robin
    rng.shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of questions (max 500)")
    ap.add_argument("--k", type=int, default=5, help="top-k for retrieval")
    ap.add_argument(
        "--sample", choices=["head", "stratified"], default="head",
        help="head: first n examples (all single-session-user). "
             "stratified: balanced across all question types.",
    )
    ap.add_argument(
        "--systems",
        nargs="+",
        default=["bm25", "cosine", "recall"],
        choices=["bm25", "cosine", "recall"],
        help="systems to run",
    )
    args = ap.parse_args()

    print("=" * 78)
    print(f"LongMemEval (cleaned, ICLR 2025) — n={args.n}, k={args.k}, sample={args.sample}")
    print("=" * 78)

    if not Path(DATA_PATH).exists():
        print(f"\nDataset not found at {DATA_PATH}.")
        print("Run: python -c \"from huggingface_hub import hf_hub_download; "
              "hf_hub_download('xiaowu0162/longmemeval-cleaned', 'longmemeval_s_cleaned.json', "
              "repo_type='dataset', local_dir='/tmp/longmemeval')\"")
        return

    with open(DATA_PATH) as f:
        all_data = json.load(f)
    if args.sample == "stratified":
        data = stratified_sample(all_data, args.n)
    else:
        data = all_data[: args.n]

    print(f"\nLoaded {len(data)} examples (out of {len(all_data)})")
    print(f"Question types in slice: {sorted(set(d.get('question_type', '?') for d in data))}")

    results: dict[str, list[dict]] = {}

    # v0.5: cache the BGE model and CosineRAG model once per system,
    # so we don't pay the ~1s load cost per question.
    _shared_bge = None
    _shared_cosine_model = None

    for sys_name in args.systems:
        print(f"\n--- {sys_name} ---")
        system_results = []
        for i, ex in enumerate(data):
            try:
                # Fresh system per example (per LongMemEval methodology — each question is a separate task)
                if sys_name == "bm25":
                    system = BM25Index()
                elif sys_name == "cosine":
                    if _shared_cosine_model is None:
                        from sentence_transformers import SentenceTransformer
                        _shared_cosine_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
                    embed_fn = lambda t: _shared_cosine_model.encode(
                        [t], normalize_embeddings=True
                    )[0]
                    system = CosineRAG(embedder=embed_fn)
                elif sys_name == "recall":
                    from recall import Memory, BGEEmbedder
                    from recall.config import Config
                    cfg = Config()
                    cfg.THRESH_GAMMA = 0.005
                    cfg.THRESH_WALK = -0.02
                    if _shared_bge is None:
                        _shared_bge = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5")
                    mem = Memory(
                        tenant=f"lme_{ex['question_id']}",
                        embedder=_shared_bge,
                        config=cfg,
                    )
                    # LongMemEval treats each session as a unit. Bulk mode keeps
                    # the whole session as one node (matching cosine RAG's
                    # granularity), instead of splitting into per-sentence chunks
                    # which adds prefix-prompt noise on short text.
                    system = RecallAdapter(mem, mode="bulk")
                else:
                    continue

                r = evaluate_one(ex, system, k=args.k)
                system_results.append(r)
                if (i + 1) % 5 == 0 or i == 0:
                    avg_recall = sum(x["recall_at_k"] for x in system_results
                                     if x["recall_at_k"] == x["recall_at_k"]) / max(
                        1, sum(1 for x in system_results if x["recall_at_k"] == x["recall_at_k"]))
                    print(f"  [{i+1}/{len(data)}] running avg recall@{args.k} = {avg_recall:.3f}")
            except Exception as e:
                print(f"  [{i+1}] FAILED: {str(e)[:120]}")
                continue

        results[sys_name] = system_results

    # Summary
    print("\n" + "=" * 78)
    print(f"SUMMARY (LongMemEval, n={args.n}, k={args.k})")
    print("=" * 78)

    name_map = {"bm25": "BM25", "cosine": "CosineRAG (BGE)", "recall": "Recall (auto)"}

    print(f"\n{'system':<20} {'recall@'+str(args.k):>10} {'MRR':>8} "
          f"{'p50 lat (ms)':>14} {'p99 lat (ms)':>14}")
    print("-" * 78)

    summary = {}
    for sys_name, rs in results.items():
        valid = [r for r in rs if r["recall_at_k"] == r["recall_at_k"]]  # filter NaN (abstention)
        if not valid:
            print(f"{name_map[sys_name]:<20} (no valid results)")
            continue
        avg_recall = sum(r["recall_at_k"] for r in valid) / len(valid)
        avg_mrr = sum(r["mrr"] for r in valid) / len(valid)
        latencies = sorted(r["latency_sec"] for r in valid)
        p50 = latencies[len(latencies) // 2] * 1000
        p99 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))] * 1000
        summary[sys_name] = {
            "n_evaluated": len(valid),
            "recall_at_k": avg_recall,
            "mrr": avg_mrr,
            "latency_p50_ms": p50,
            "latency_p99_ms": p99,
        }
        print(f"{name_map[sys_name]:<20} {avg_recall:>10.3f} {avg_mrr:>8.3f} "
              f"{p50:>14.1f} {p99:>14.1f}")

    print("\nPer-question-type breakdown (recall@k):")
    print("-" * 78)
    qtype_breakdown: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sys_name, rs in results.items():
        for r in rs:
            if r["recall_at_k"] == r["recall_at_k"]:
                qtype_breakdown[r["qtype"]][sys_name].append(r["recall_at_k"])

    for qtype, by_sys in sorted(qtype_breakdown.items()):
        line_parts = [f"{qtype:<28}"]
        for sys_name in results:
            scores = by_sys.get(sys_name, [])
            if scores:
                line_parts.append(f"{name_map[sys_name][:10]}={sum(scores)/len(scores):.3f}")
            else:
                line_parts.append(f"{name_map[sys_name][:10]}=---")
        print(" | ".join(line_parts))

    # Save JSON (include sample type so head vs stratified don't collide)
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    suffix = f"_{args.sample}" if args.sample != "head" else ""
    out_path = out_dir / f"results_n{args.n}_k{args.k}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "raw": results}, f, indent=2, default=str)
    print(f"\nSaved raw results: {out_path}")


if __name__ == "__main__":
    main()
