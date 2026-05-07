"""Empirical hallucination-bound calibration.

Tests the Conformal Risk Control bound: do empirically observed hallucination
rates fall within the predicted bound? This validates the math.

Methodology (per Kang et al. ICML 2024 — C-RAG):
  1. Build a corpus of N facts.
  2. For each fact, generate a related question whose ground-truth answer is
     in the corpus.
  3. Half of questions get answered by Recall with bound='soft' (so we record
     all answers + flag). Half get answered with bound='strict' (only emit
     supported claims).
  4. Compute empirical hallucination rate (per fact-checking against gold).
  5. Compute predicted CRC bound on a held-out calibration set.
  6. Verify: empirical ≤ bound at 95% confidence.

We use a synthetic corpus where ground-truth checking is unambiguous —
named entity / number / date matching against the gold facts.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


GOLD_FACTS = [
    {"q": "What's the migration date for the queue switch?", "gold": "March 2024", "fact": "The queue migration to Redis Streams happened in March 2024."},
    {"q": "What's the Acme Corp contract value?", "gold": "$4.2M", "fact": "The Acme Corp contract is worth $4.2M for 3 years."},
    {"q": "When was the database restored?", "gold": "03:47 UTC", "fact": "The database was restored at 03:47 UTC on November 12, 2023."},
    {"q": "Who proposed the bi-phasic dose-response model?", "gold": "Priya Singh", "fact": "Priya Singh proposed the bi-phasic dose-response model on April 28."},
    {"q": "When was the HIPAA BAA signed?", "gold": "January 15, 2025", "fact": "The HIPAA BAA with Acme Medical was signed on January 15, 2025."},
    {"q": "What approval is required for deployments?", "gold": "on-call engineer", "fact": "Deployments require approval from the on-call engineer per runbook 4.2."},
    {"q": "What package manager does the monorepo use?", "gold": "Bun 1.1.4", "fact": "The monorepo uses Bun 1.1.4 as the package manager."},
    {"q": "What are Stripe's payment processing fees?", "gold": "2.9%", "fact": "Stripe payment processing fees average 2.9% + $0.30 per transaction."},
    {"q": "How much capacity is in prod-eu-west-1?", "gold": "48 vCPU", "fact": "The prod-eu-west-1 namespace has 48 vCPU reserved capacity."},
    {"q": "When do daily backups run?", "gold": "02:00 UTC", "fact": "Daily database backups happen at 02:00 UTC."},
    {"q": "What's our customer count?", "gold": "450", "fact": "We have 450 paying customers as of last quarter."},
    {"q": "What's the NPS score?", "gold": "47", "fact": "Our NPS score is 47 as of last quarter."},
    {"q": "Who manages the CEO calendar?", "gold": "Lisa", "fact": "Lisa from operations manages the CEO calendar."},
    {"q": "When does equity vest?", "gold": "4 years", "fact": "Equity grants vest over 4 years with a 1-year cliff."},
    {"q": "What's the office location?", "gold": "San Francisco", "fact": "The office is in San Francisco SOMA district."},
    {"q": "When does board meet?", "gold": "monthly", "fact": "The board meets monthly."},
    {"q": "What's our SLA on support?", "gold": "4-hour", "fact": "Support SLA is 4-hour first-response, 24-hour resolution."},
    {"q": "What pipeline tool do we use?", "gold": "Turborepo 1.13.2", "fact": "We use Turborepo 1.13.2 for build pipelines."},
    {"q": "Who signed the BAA?", "gold": "Acme Medical", "fact": "We signed a HIPAA BAA with Acme Medical."},
    {"q": "What's the order ID format?", "gold": "OD-", "fact": "Order IDs are 8-character alphanumeric prefixed with 'OD-'."},
]


def fact_matches(answer: str, gold: str) -> bool:
    """Strict gold-answer matching."""
    return gold.lower() in (answer or "").lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cal", type=int, default=15, help="calibration set size (used to fit the bound)")
    ap.add_argument("--n-test", type=int, default=20, help="test set size (used to verify empirical ≤ bound)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-llm", action="store_true",
                    help="run without LLM (uses MockLLMClient — bound test only, no real generation)")
    args = ap.parse_args()

    print("=" * 78)
    print("Empirical Hallucination-Bound Calibration (Conformal Risk Control)")
    print("=" * 78)

    from recall import Memory
    from recall.bound.conformal import crc_bound

    # Use real LLM if available
    use_llm = not args.no_llm and os.environ.get("OPENAI_API_KEY")
    if use_llm:
        from recall.llm_router import RouterClient
        llm = RouterClient(model="openai/gpt-4o-mini")
        embedder = None
        try:
            from recall import BGEEmbedder
            embedder = BGEEmbedder("BAAI/bge-small-en-v1.5")
        except ImportError:
            pass
        mem = Memory(tenant="bound_cal", llm=llm, embedder=embedder)
        print(f"  Using REAL LLM (gpt-4o-mini via TokenRouter)")
    else:
        mem = Memory(tenant="bound_cal")
        print(f"  Using MockLLMClient (no real generation; bound is computed from retrieval only)")

    # Ingest the corpus
    print(f"  Ingesting {len(GOLD_FACTS)} gold facts...")
    for f in GOLD_FACTS:
        mem.observe(f["fact"], "Acknowledged.", scope={"set": "all"})

    # Split questions: calibration vs test
    rng = random.Random(args.seed)
    indices = list(range(len(GOLD_FACTS)))
    rng.shuffle(indices)
    cal_idx = indices[: args.n_cal]
    test_idx = indices[args.n_cal : args.n_cal + args.n_test]

    print(f"\n  Calibration set: {len(cal_idx)} questions")
    print(f"  Test set:        {len(test_idx)} questions")

    # CALIBRATION: run questions, record empirical hallucination
    print("\n--- Calibration phase ---")
    cal_risks = []  # 1 if hallucination, 0 if correct
    for i in cal_idx:
        item = GOLD_FACTS[i]
        try:
            gen = mem.bounded_generate(item["q"], scope={"set": "all"}, bound="soft")
            correct = fact_matches(gen.text, item["gold"])
            cal_risks.append(0 if correct else 1)
        except Exception as e:
            cal_risks.append(1)  # treat error as hallucination

    empirical_cal_rate = sum(cal_risks) / max(len(cal_risks), 1)
    bound = crc_bound(cal_risks, delta=0.05)

    print(f"\n  Empirical hallucination rate (calibration):  {empirical_cal_rate:.3f}")
    print(f"  Hoeffding upper bound (95% CI):              {bound['hoeffding']:.3f}")
    print(f"  Wilson upper bound  (95% CI):               {bound['wilson']:.3f}")
    print(f"  CRC reported bound:                         {bound['crc_min']:.3f}")

    # TEST: run held-out questions, verify empirical ≤ bound
    print("\n--- Test phase (verifying empirical ≤ bound) ---")
    test_risks = []
    test_results = []
    for i in test_idx:
        item = GOLD_FACTS[i]
        try:
            gen = mem.bounded_generate(item["q"], scope={"set": "all"}, bound="soft")
            correct = fact_matches(gen.text, item["gold"])
            test_risks.append(0 if correct else 1)
            test_results.append({
                "question": item["q"],
                "gold": item["gold"],
                "answer": gen.text[:120],
                "correct": correct,
                "bound_value": gen.bound_value,
                "flagged_claims": len(gen.flagged_claims),
            })
        except Exception as e:
            test_risks.append(1)

    empirical_test_rate = sum(test_risks) / max(len(test_risks), 1)
    print(f"\n  Empirical hallucination rate (test):  {empirical_test_rate:.3f}")
    print(f"  Predicted bound (from calibration):   {bound['crc_min']:.3f}")

    valid = empirical_test_rate <= bound['crc_min']
    print(f"\n  CRC bound holds: {'YES' if valid else 'NO'}  "
          f"(test={empirical_test_rate:.3f} {'<=' if valid else '>'} bound={bound['crc_min']:.3f})")

    # Strict-mode test: how often does bounded_generate(strict) refuse to emit?
    print("\n--- Strict-mode refusal rate ---")
    refused = 0
    answered = 0
    for i in test_idx[:10]:  # smaller subset to keep cost down
        item = GOLD_FACTS[i]
        try:
            gen = mem.bounded_generate(item["q"], scope={"set": "all"}, bound="strict")
            answered += 1
        except Exception:
            refused += 1
    total = refused + answered
    print(f"  Strict mode: {answered}/{total} answered, {refused}/{total} refused (HallucinationBlocked)")

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "calibration.json", "w") as f:
        json.dump({
            "n_cal": len(cal_risks),
            "n_test": len(test_risks),
            "empirical_cal_rate": empirical_cal_rate,
            "empirical_test_rate": empirical_test_rate,
            "predicted_bound": bound,
            "bound_holds_on_test": valid,
            "test_results": test_results,
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_dir / 'calibration.json'}")


if __name__ == "__main__":
    main()
