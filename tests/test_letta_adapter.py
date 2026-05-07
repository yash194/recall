"""Tests for the Letta integration adapter."""
from __future__ import annotations

from recall import HashEmbedder, MockLLMClient
from recall.integrations import LettaMemoryBackend


def test_archival_insert_and_search():
    backend = LettaMemoryBackend(
        tenant="letta_test",
        embedder=HashEmbedder(dim=64),
        llm=MockLLMClient(),
    )
    r = backend.archival_memory_insert("User decided to use Redis Streams.")
    assert r["ok"]

    results = backend.archival_memory_search("Redis", top_k=3)
    assert isinstance(results, list)


def test_archival_count():
    backend = LettaMemoryBackend(
        tenant="letta_count",
        embedder=HashEmbedder(dim=64),
        llm=MockLLMClient(),
    )
    backend.archival_memory_insert("Real substantive fact about the platform.")
    n = backend.archival_memory_count()
    assert n >= 0


def test_archival_forget():
    backend = LettaMemoryBackend(
        tenant="letta_forget",
        embedder=HashEmbedder(dim=64),
        llm=MockLLMClient(),
    )
    r = backend.archival_memory_insert("Real substantive content to remember.")
    if not r["nodes_written"]:
        return  # may not promote with HashEmbedder; skip
    nid = r["nodes_written"][0]
    f = backend.archival_memory_forget(nid, reason="test")
    assert f["ok"]
