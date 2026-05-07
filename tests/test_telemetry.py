"""Tests for telemetry / Metrics."""
from __future__ import annotations

import time

from recall import HashEmbedder, Memory, MockLLMClient
from recall.telemetry import Metrics


def test_metrics_increment():
    m = Metrics()
    m.increment("foo")
    m.increment("foo", by=4)
    assert m.snapshot()["counts"]["foo"] == 5


def test_metrics_observe_latency():
    m = Metrics()
    m.observe_latency("op", 0.001)
    m.observe_latency("op", 0.005)
    snap = m.snapshot()
    assert "op" in snap["latency_p"]
    assert snap["latency_p"]["op"]["count"] == 2


def test_metrics_time_context():
    m = Metrics()
    with m.time("slow_op"):
        time.sleep(0.001)
    snap = m.snapshot()
    assert "slow_op" in snap["latency_p"]
    assert snap["counts"]["slow_op_count"] == 1


def test_memory_records_metrics():
    mem = Memory(tenant="metrics_test", embedder=HashEmbedder(dim=64), llm=MockLLMClient())
    mem.observe("Some fact.", "Yes.", scope={})
    mem.recall("query", scope={}, mode="symmetric")
    snap = mem.metrics.snapshot()
    assert "observe" in snap["latency_p"]
    assert "recall" in snap["latency_p"]
