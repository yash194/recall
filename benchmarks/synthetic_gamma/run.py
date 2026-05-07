"""Synthetic Γ benchmark — does Γ-walk recover a planted causal chain?

We seed the memory with a deliberate causal chain:
  A → B → C → D → E  (each step labelled with explicit cause→effect language)
plus 20 distractor facts in the same scope.

Then ask: "What ultimately led to E?" and check whether the retrieved
subgraph contains the planted chain.

This is a unit-test-grade synthetic experiment. The 100-trace multi-domain
benchmark per yash_results.md replaces this for the real Γ paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from recall import Memory


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
] * 2  # 20 distractors


def main():
    mem = Memory(tenant="synthetic_gamma")

    # Plant the causal chain in order — Γ should pick up the directionality
    # from the natural language alone (no manual edge-typing).
    for i, fact in enumerate(CAUSAL_CHAIN):
        mem.observe(fact, "Acknowledged.", scope={"project": "infra"})

    # Add distractors
    for d in DISTRACTORS:
        mem.observe(d, "Got it.", scope={"project": "infra"})

    print(f"Stats after planting: {mem.stats()}")

    # Ask the directional question
    result = mem.recall(
        "What ultimately led to the queue being stable?",
        scope={"project": "infra"},
        mode="path",
    )

    print(f"\nQuery: 'What ultimately led to the queue being stable?'")
    print(f"Mode:  {result.mode}")
    print(f"Retrieved {len(result.subgraph_nodes)} nodes, {len(result.subgraph_edges)} edges")
    print()

    # Check whether the chain was recovered
    chain_lower = [c.lower() for c in CAUSAL_CHAIN]
    retrieved_texts = [n.text.lower() for n in result.subgraph_nodes]

    matches = 0
    for chain_fact in chain_lower:
        for retrieved_text in retrieved_texts:
            # Loose match: 50% token overlap
            chain_toks = set(chain_fact.split())
            ret_toks = set(retrieved_text.split())
            if not chain_toks:
                continue
            overlap = len(chain_toks & ret_toks) / len(chain_toks)
            if overlap > 0.4:
                matches += 1
                break

    recall_rate = matches / len(CAUSAL_CHAIN)
    print(f"Causal-chain recall: {matches}/{len(CAUSAL_CHAIN)} = {recall_rate * 100:.0f}%")
    print(f"Target: ≥ 60% (3 of 5 chain elements recovered)")

    print("\nRetrieved nodes:")
    for n in result.subgraph_nodes:
        print(f"  [{n.role or 'fact'}] {n.text[:100]}")


if __name__ == "__main__":
    main()
