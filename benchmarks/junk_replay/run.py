"""mem0-#4573 junk-replay benchmark.

Replays the kind of corpus that produced 97.8% junk in mem0's audit.

Two modes:
  - default: template-based quality classifier (no API key needed)
  - --llm: LLM-driven quality classifier via OPENAI_API_KEY (much stricter)

Both modes score: junk_in_memory_rate. Mem0 audit baseline: 97.8%.
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from recall import Memory


# --- Junk categories ---

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
    "User prefers Vim over Emacs.",  # the famous mem0 fabrication
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


def synth_corpus(n_total: int = 2_000) -> list[tuple[str, str, str]]:
    """Generate a synthetic corpus matching the mem0 #4573 failure profile.

    Returns list of (user_msg, agent_msg, ground_truth_category) tuples.
    """
    rng = random.Random(2026)
    pool: list[tuple[str, str, str]] = []
    for _ in range(int(n_total * 0.30)):
        pool.append(("(system)", rng.choice(SYSTEM_PROMPT_LEAKS), "system_leak"))
    for _ in range(int(n_total * 0.50)):
        pool.append(("user said something", rng.choice(BOILERPLATE), "boilerplate"))
    for _ in range(int(n_total * 0.10)):
        pool.append(("conversation", rng.choice(HALLUCINATED_PROFILE_CLAIMS), "halluc_profile"))
    for _ in range(int(n_total * 0.10)):
        pool.append(("user message", rng.choice(GENUINE_FACTS), "genuine"))
    rng.shuffle(pool)
    return pool


def category_of(text: str) -> str:
    """Reverse-classify a stored node back to its junk category for analysis."""
    lc = text.lower()
    for j in SYSTEM_PROMPT_LEAKS:
        if j.lower() in lc:
            return "system_leak"
    for j in BOILERPLATE:
        if j.lower() in lc:
            return "boilerplate"
    for j in HALLUCINATED_PROFILE_CLAIMS:
        if j.lower() in lc:
            return "halluc_profile"
    for j in GENUINE_FACTS:
        if j.lower() in lc:
            return "genuine"
    return "unknown"


def main():
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true",
                    help="Use LLM quality classifier (requires OPENAI_API_KEY)")
    ap.add_argument("--n", type=int, default=300,
                    help="Corpus size (smaller for LLM mode due to API cost)")
    args = ap.parse_args()

    print("=" * 70)
    print(f"mem0-#4573 junk-replay benchmark — n={args.n}, llm={args.llm}")
    print("=" * 70)

    corpus = synth_corpus(n_total=args.n)
    if args.llm:
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: --llm requires OPENAI_API_KEY in env.")
            return
        from recall import RouterClient
        llm = RouterClient(model=os.environ.get("RECALL_QUALITY_MODEL", "openai/gpt-4o-mini"))
        mem = Memory(tenant="junk_replay_llm", llm=llm, use_llm_quality=True)
        print(f"Using LLM-driven quality gate via {llm.__class__.__name__}\n")
    else:
        mem = Memory(tenant="junk_replay")
        print("Using template-based quality gate\n")

    counts = Counter()
    by_category_in: Counter = Counter()
    for user_msg, agent_msg, category in corpus:
        by_category_in[category] += 1
        # System-prompt leaks come from a synthetic 'recall_artifact' source so
        # the provenance firewall rejects them. (In production, the LLM SDK
        # marks system-prompt echoes with that source.)
        source = "recall_artifact" if user_msg == "(system)" else "conversation"
        result = mem.observe(user_msg, agent_msg, source=source)
        if result.skipped_recall_loop:
            counts["rejected_provenance"] += 1
        elif result.drawer_was_duplicate:
            counts["skipped_drawer_dup"] += 1
        else:
            counts["nodes_rejected_quality"] += sum(
                1 for (_, reason) in result.nodes_rejected if "low_quality" in reason
            )
            counts["nodes_skipped_text_dup"] += result.nodes_skipped_duplicate
            counts["nodes_promoted"] += len(result.nodes_written)

    total = len(corpus)

    # Determine which categories landed in memory
    promoted_nodes = mem.storage.all_active_nodes()
    by_category_promoted = Counter(category_of(n.text) for n in promoted_nodes)

    # Junk-in-memory rate
    junk_in_memory = sum(v for k, v in by_category_promoted.items() if k != "genuine")
    total_in_memory = sum(by_category_promoted.values())
    junk_rate = (junk_in_memory / total_in_memory) if total_in_memory else 0.0

    # Print
    print(f"\n--- Input corpus (by category, with duplicates) ---")
    for cat, n in by_category_in.most_common():
        print(f"  {cat:20s} {n:>5d}")
    print(f"  {'TOTAL':20s} {total:>5d}")

    print(f"\n--- Pipeline outcomes ---")
    print(f"  rejected_provenance       {counts['rejected_provenance']:>5d}  (recall_artifact source)")
    print(f"  skipped_drawer_dup        {counts['skipped_drawer_dup']:>5d}  (exact-text repeats)")
    print(f"  nodes_rejected_quality    {counts['nodes_rejected_quality']:>5d}  (boilerplate templates)")
    print(f"  nodes_skipped_text_dup    {counts['nodes_skipped_text_dup']:>5d}  (post-split repeats)")
    print(f"  nodes_promoted            {counts['nodes_promoted']:>5d}  (actually stored)")

    print(f"\n--- What ended up IN MEMORY (the user-visible state) ---")
    for cat, n in by_category_promoted.most_common():
        marker = " "
        if cat == "genuine":
            marker = "+"  # this is what we want
        elif cat in ("system_leak", "boilerplate", "halluc_profile"):
            marker = "-"  # this is junk
        print(f"  {marker} {cat:20s} {n:>3d}")
    print(f"    {'TOTAL':20s} {total_in_memory:>3d}")

    print(f"\n>>> Junk-in-memory rate: {junk_rate * 100:.1f}%")
    print(f"    (mem0 #4573 measured: 97.8% junk-in-memory)")
    print(f"    (Recall target:       <  5.0%)")

    if junk_rate < 0.05:
        print(f"    PASS: Recall met target.")
    elif junk_rate < 0.20:
        print(f"    PARTIAL: substantial improvement over Mem0 baseline; tune patterns to hit <5%.")
    else:
        print(f"    FAIL: tune quality patterns and/or improve dedup.")


if __name__ == "__main__":
    main()
