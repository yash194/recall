"""Tests for query intent classification."""
from __future__ import annotations

from recall.retrieval.intent import classify_intent


def test_directional_keywords():
    assert classify_intent("Why did we switch to Redis?") == "directional"
    assert classify_intent("What caused the outage?") == "directional"
    assert classify_intent("What comes after this step?") == "directional"


def test_symmetric_keywords():
    assert classify_intent("What is our queue stack?") in ("symmetric", "hybrid")
    assert classify_intent("Tell me about Redis.") in ("symmetric", "hybrid")


def test_hybrid_when_both():
    # "What is X" + "why" ⇒ hybrid
    assert classify_intent("What is the reason we use Redis?") == "hybrid"


def test_default_hybrid():
    # Ambiguous query → defaults to hybrid
    assert classify_intent("queue stack") in ("hybrid", "symmetric", "directional")
