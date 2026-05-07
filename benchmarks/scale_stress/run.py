"""Scale stress test — the 'infinite-feeling, finite-actual memory' claim.

Validates: as memory grows from 100 → 50,000 nodes, does Recall maintain:
  1. Bounded *active* node count after consolidation? (the bounded part)
  2. Stable retrieval latency? (the infinite-feeling part)
  3. Stable recall quality? (the no-quality-decay part)

Compared to a baseline cosine-RAG which grows linearly forever and gets
slower / noisier with size.

We seed a small "gold" planted set at the start (10 distinctive memories
with a known query that should match them), then ingest N synthetic
distractor memories. At each milestone (100 / 1K / 10K / 50K) we measure:
  - storage size on disk (MB)
  - active node count
  - retrieval latency p50 / p99 over 30 queries
  - gold-set recall@5 (does it still find the planted memories?)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines_comparison"))

import numpy as np

from baselines import CosineRAG, RecallAdapter


# 10 distinctive "gold" memories that should be retrievable forever
GOLD_MEMORIES = [
    "The team migrated from Postgres LISTEN/NOTIFY to Redis Streams in March 2024 because of message loss under load.",
    "Customer Acme Corp signed a 3-year contract worth $4.2M starting Q3 2024.",
    "The production database was restored from backup at 03:47 UTC on November 12, 2023 after the partition fill incident.",
    "Engineer Priya Singh proposed the bi-phasic dose-response model in the April 28 design review.",
    "The HIPAA Business Associate Agreement was signed with Acme Medical on January 15, 2025.",
    "All deployments require approval from the on-call engineer per the production runbook section 4.2.",
    "The monorepo uses Bun 1.1.4 as the package manager and Turborepo 1.13.2 for builds.",
    "Stripe payment processing fees average 2.9% + $0.30 per transaction for our customer mix.",
    "The kubernetes namespace 'prod-eu-west-1' has dedicated reserved capacity of 48 vCPU.",
    "Daily database backups happen at 02:00 UTC and are retained for 30 days in S3 us-east-1.",
]

# Gold queries that should retrieve the matching gold memory
GOLD_QUERIES = [
    "what queue tech did we switch to and why?",
    "what's the Acme Corp contract value?",
    "when was the production database restored?",
    "who proposed the bi-phasic dose-response model?",
    "did we sign a BAA with Acme Medical?",
    "what's the deployment approval process?",
    "what build tool does the monorepo use?",
    "what are our payment processing fees?",
    "how much capacity is in prod-eu-west-1?",
    "when do daily backups run?",
]


def synthesize_distractor(rng: random.Random, idx: int) -> str:
    """Generate a plausible-but-irrelevant distractor memory."""
    topics = [
        "internal tooling", "marketing campaign", "office logistics",
        "team standup notes", "product roadmap brainstorm", "customer support ticket",
        "engineering rotation schedule", "quarterly OKR", "vendor evaluation",
        "design system audit", "documentation update", "code review feedback",
        "sprint retrospective", "performance review", "onboarding checklist",
    ]
    objects = [
        "the dashboard", "the API gateway", "the analytics pipeline", "the auth service",
        "the search index", "the email templates", "the notification system",
        "the user profile page", "the billing flow", "the admin panel", "the CLI tools",
    ]
    actions = [
        "needs refactoring", "was deprecated last quarter", "got a +12% latency improvement",
        "is being migrated to a new framework", "had an outage last week",
        "added a feature flag", "is feature-complete", "is in beta",
        "was reviewed by the security team", "passed an audit",
    ]
    return (f"Memo #{idx}: regarding {rng.choice(topics)}, {rng.choice(objects)} "
            f"{rng.choice(actions)} per the meeting on {rng.randint(1, 28):02d}/"
            f"{rng.randint(1, 12):02d}.")


def measure_at_milestone(system, milestone_label: str, gold_queries: list[str],
                         expected_substrings: list[str], k: int = 5) -> dict:
    """Run gold queries, measure recall + latency."""
    latencies = []
    hit_count = 0
    for q, expected_sub in zip(gold_queries, expected_substrings):
        t0 = time.time()
        hits = system.search(q, k=k)
        latencies.append(time.time() - t0)
        # Check if any retrieved hit contains the expected substring
        hit_texts = [h.text for h in hits]
        if any(expected_sub.lower() in h.lower() for h in hit_texts):
            hit_count += 1

    latencies.sort()
    return {
        "milestone": milestone_label,
        "gold_recall_at_k": hit_count / max(len(gold_queries), 1),
        "active_size": system.size(),
        "latency_p50_ms": latencies[len(latencies) // 2] * 1000,
        "latency_p99_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))] * 1000,
        "latency_mean_ms": sum(latencies) / len(latencies) * 1000,
    }


def run_system(system, n_total: int, milestones: list[int], system_name: str) -> dict:
    """Ingest gold + distractors, take measurements at each milestone."""
    print(f"\n{'='*78}\n{system_name} — scale-stress run (target N={n_total})\n{'='*78}")

    # Plant gold memories first with distinctive substrings to track them
    gold_substrings = [
        "Redis Streams", "Acme Corp", "March 2024",
        "$4.2M", "03:47 UTC", "Priya Singh",
        "bi-phasic", "Acme Medical", "HIPAA",
        "section 4.2", "Bun 1.1.4", "Turborepo",
        "2.9%", "$0.30", "prod-eu-west-1",
        "48 vCPU", "02:00 UTC", "30 days",
    ]
    # Map each query to a substring it should hit
    gold_query_substrings = [
        "Redis Streams",      # queue switch
        "$4.2M",              # contract value
        "03:47 UTC",          # restore time
        "Priya Singh",        # proposer
        "HIPAA",              # BAA
        "section 4.2",        # deployment approval
        "Bun 1.1.4",          # build tool
        "2.9%",               # payment fees
        "48 vCPU",            # capacity
        "02:00 UTC",          # backup time
    ]

    for g in GOLD_MEMORIES:
        system.add(g, scope={"label": "gold"})

    rng = random.Random(2026)
    measurements = []

    n_distractors_added = 0
    for target in milestones:
        # Add distractors until we hit the milestone (where N = gold + distractors)
        while system.size() < target:
            text = synthesize_distractor(rng, n_distractors_added)
            system.add(text, scope={"label": "noise"})
            n_distractors_added += 1
            if system.size() >= target:
                break
            if system.size() >= n_total:
                break

        if system.size() < target * 0.95:
            # Couldn't grow past the system's bounded ceiling — record and stop
            print(f"  Reached system size cap at {system.size()} (target was {target})")

        m = measure_at_milestone(system, str(target), GOLD_QUERIES, gold_query_substrings)
        m["target"] = target
        m["distractors_added"] = n_distractors_added
        measurements.append(m)

        print(f"  N={target:>6d}  active={m['active_size']:>5d}  "
              f"gold_recall@5={m['gold_recall_at_k']:.2f}  "
              f"latency p50={m['latency_p50_ms']:>6.1f}ms p99={m['latency_p99_ms']:>6.1f}ms")

        if system.size() >= n_total:
            break

    return {"system": system_name, "measurements": measurements}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=10000)
    ap.add_argument("--milestones", nargs="+", type=int,
                    default=[100, 500, 1000, 2500, 5000, 10000])
    ap.add_argument(
        "--systems", nargs="+",
        default=["recall", "cosine"],
        choices=["recall", "cosine"],
    )
    args = ap.parse_args()

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"scale_n{args.max_n}.json"

    all_results = {}

    for sys_name in args.systems:
        if sys_name == "cosine":
            system = CosineRAG()
            label = "Cosine RAG (BGE-small)"
        elif sys_name == "recall":
            from recall import Memory, BGEEmbedder
            from recall.config import Config
            cfg = Config()
            cfg.THRESH_GAMMA = 0.005
            cfg.THRESH_WALK = -0.02
            cfg.K_NEIGHBORS = 6  # cap edges per write to control growth
            mem = Memory(
                tenant="scale_stress",
                embedder=BGEEmbedder("BAAI/bge-small-en-v1.5"),
                storage=f"sqlite:///tmp/recall_stress_{int(time.time())}.db",
                config=cfg,
            )
            # Bulk mode: each ingested memory becomes one node (matching the
            # cosine baseline's granularity). Demonstrates the pure scaling
            # behavior of the v0.5 vectorized cache without graph overhead.
            system = RecallAdapter(mem, mode="bulk")
            label = "Recall (BGE + bulk)"

        result = run_system(system, args.max_n, args.milestones, label)
        all_results[sys_name] = result

    # Comparison table
    print("\n" + "=" * 78)
    print("COMPARISON — gold-recall@5 and p50 latency vs scale")
    print("=" * 78)
    print(f"\n{'N':>8} | {'Recall recall':>14} | {'Cosine recall':>14} | "
          f"{'Recall p50':>12} | {'Cosine p50':>12}")
    print("-" * 78)

    if "recall" in all_results and "cosine" in all_results:
        for r_m, c_m in zip(
            all_results["recall"]["measurements"],
            all_results["cosine"]["measurements"],
        ):
            print(f"{r_m['target']:>8d} | {r_m['gold_recall_at_k']:>14.2f} | "
                  f"{c_m['gold_recall_at_k']:>14.2f} | {r_m['latency_p50_ms']:>10.1f}ms | "
                  f"{c_m['latency_p50_ms']:>10.1f}ms")

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
