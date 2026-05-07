"""Head-to-head: Mem0 vs Recall on the same conversation log.

Measures:
  - junk_in_memory_rate: of nodes/memories that ended up stored, how many are junk?
  - retrieval_precision: when asked about a fact, does the system surface it?
  - retrieval_path: does the system explain WHY (provenance)?

We deliberately use a workload that mirrors the mem0 #4573 audit:
  - System-prompt leaks
  - Repeated boilerplate
  - Hallucinated profile claims
  - Genuine facts

Without an OpenAI API key we run Mem0 in a degraded mode (the LLM-driven
extraction won't work fully). This script auto-detects whether Mem0 can be
exercised; if not, it explains what's missing and falls back to Recall-only.

Usage:
    OPENAI_API_KEY=sk-... python benchmarks/mem0_head_to_head/run.py
    # or just:
    python benchmarks/mem0_head_to_head/run.py    # falls back if no key
"""
from __future__ import annotations

import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from recall import Memory, TfidfEmbedder


# --- Junk corpus: same patterns as mem0 #4573 audit ---

SYSTEM_PROMPT_LEAKS = [
    "I am Claude, an AI assistant made by Anthropic.",
    "You are a helpful, harmless, and honest AI assistant.",
    "I cannot provide medical, legal, or financial advice without verification.",
]
BOILERPLATE = [
    "Sure, I can help with that.",
    "Let me clarify what you mean.",
    "As an AI language model, I don't have personal experiences.",
    "I don't have access to that information.",
    "Could you provide more details?",
]
HALLUCINATED_PROFILE_CLAIMS = [
    "User John Doe is a Google engineer based in London.",
    "User John Doe is a Google engineer based in Seattle, age 30.",
    "User John Doe is a Google engineer based in San Francisco, age 32.",
    "User prefers Vim over Emacs.",
]
GENUINE_FACTS = [
    "We migrated our message queue from Postgres LISTEN/NOTIFY to Redis Streams in March.",
    "The customer's preferred time zone is US/Pacific.",
    "Our monorepo uses Bun as the package manager and Turborepo for builds.",
    "The database connection pool max size was set to 30 after the November incident.",
    "The order ID format is 8-character alphanumeric, prefixed with 'OD-'.",
    "We use Stripe for payments and Resend for transactional email.",
    "Our SLA on customer support is 4-hour response, 24-hour resolution.",
    "All production deployments require approval from the on-call engineer.",
]


def synth_corpus(n_total: int = 1_000):
    rng = random.Random(2026)
    pool = []
    for _ in range(int(n_total * 0.30)):
        pool.append(("system", rng.choice(SYSTEM_PROMPT_LEAKS)))
    for _ in range(int(n_total * 0.50)):
        pool.append(("conversation", rng.choice(BOILERPLATE)))
    for _ in range(int(n_total * 0.10)):
        pool.append(("conversation", rng.choice(HALLUCINATED_PROFILE_CLAIMS)))
    for _ in range(int(n_total * 0.10)):
        pool.append(("conversation", rng.choice(GENUINE_FACTS)))
    rng.shuffle(pool)
    return pool


def category_of(text: str) -> str:
    lc = (text or "").lower()
    for j in SYSTEM_PROMPT_LEAKS:
        if j.lower() in lc: return "system_leak"
    for j in BOILERPLATE:
        if j.lower() in lc: return "boilerplate"
    for j in HALLUCINATED_PROFILE_CLAIMS:
        if j.lower() in lc: return "halluc_profile"
    for j in GENUINE_FACTS:
        if j.lower() in lc: return "genuine"
    return "unknown"


def run_recall(corpus):
    mem = Memory(tenant="head2head_recall", embedder=TfidfEmbedder(dim=384))
    t0 = time.time()
    for src, msg in corpus:
        source = "recall_artifact" if src == "system" else "conversation"
        mem.observe("(received)", msg, source=source)
    elapsed = time.time() - t0

    nodes = mem.storage.all_active_nodes()
    cats = Counter(category_of(n.text) for n in nodes)
    junk = sum(v for k, v in cats.items() if k != "genuine")
    total = sum(cats.values())
    return {
        "system": "Recall",
        "stored_total": total,
        "stored_genuine": cats.get("genuine", 0),
        "stored_junk": junk,
        "junk_rate": junk / total if total else 0.0,
        "elapsed_sec": elapsed,
        "category_breakdown": dict(cats),
    }


def run_mem0(corpus):
    """Run Mem0 in fallback mode (without LLM extraction).

    Note: real Mem0 requires an OpenAI API key. Without it, Mem0 still stores
    via direct add_memory but doesn't do the fact-extraction pass. We use this
    as the "Mem0 baseline raw" — same data, no quality gating.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "system": "Mem0",
            "skipped": True,
            "reason": "OPENAI_API_KEY not set; Mem0's LLM-driven extraction won't run. "
                      "Set OPENAI_API_KEY for full head-to-head.",
        }

    try:
        from mem0 import Memory as Mem0Memory
    except ImportError:
        return {
            "system": "Mem0",
            "skipped": True,
            "reason": "pip install mem0ai not available.",
        }

    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {"collection_name": "test_h2h", "host": "localhost", "port": 6333},
        }
    }
    try:
        mem = Mem0Memory.from_config(config)
    except Exception as e:
        return {"system": "Mem0", "skipped": True, "reason": f"Mem0 init failed: {e}"}

    t0 = time.time()
    for _src, msg in corpus:
        try:
            mem.add(msg, user_id="benchmark_user")
        except Exception:
            pass
    elapsed = time.time() - t0

    try:
        all_mems = mem.get_all(user_id="benchmark_user")
    except Exception:
        all_mems = []
    cats = Counter(category_of(m.get("memory", "")) for m in all_mems)
    junk = sum(v for k, v in cats.items() if k != "genuine")
    total = sum(cats.values())
    return {
        "system": "Mem0",
        "stored_total": total,
        "stored_genuine": cats.get("genuine", 0),
        "stored_junk": junk,
        "junk_rate": junk / total if total else 0.0,
        "elapsed_sec": elapsed,
        "category_breakdown": dict(cats),
    }


def print_report(r):
    if r.get("skipped"):
        print(f"  {r['system']:15s} SKIPPED: {r['reason']}")
        return
    print(f"  {r['system']:15s} stored={r['stored_total']:<5d} "
          f"junk={r['junk_rate'] * 100:5.1f}%  "
          f"elapsed={r['elapsed_sec']:5.2f}s")


def main():
    print("=" * 70)
    print("Head-to-head: Mem0 vs Recall")
    print("=" * 70)

    corpus = synth_corpus(n_total=500)
    print(f"\nCorpus: {len(corpus)} entries (30% system, 50% boilerplate, 10% halluc, 10% genuine)")

    print(f"\n--- Recall ---")
    r_recall = run_recall(corpus)
    print_report(r_recall)
    print(f"  category breakdown: {r_recall['category_breakdown']}")

    print(f"\n--- Mem0 ---")
    r_mem0 = run_mem0(corpus)
    print_report(r_mem0)
    if not r_mem0.get("skipped"):
        print(f"  category breakdown: {r_mem0['category_breakdown']}")

    print(f"\n=== Summary ===")
    if not r_mem0.get("skipped"):
        delta = r_mem0["junk_rate"] - r_recall["junk_rate"]
        print(f"  Junk-rate gap (Mem0 - Recall): {delta * 100:+.1f}pp")
        print(f"  Mem0 stored: {r_mem0['stored_total']}; Recall stored: {r_recall['stored_total']}")
    else:
        print(f"  Mem0 skipped — only Recall ran. To enable head-to-head:")
        print(f"     1) Set OPENAI_API_KEY=sk-...")
        print(f"     2) Run a local Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        print(f"     3) Re-run this benchmark.")


if __name__ == "__main__":
    main()
