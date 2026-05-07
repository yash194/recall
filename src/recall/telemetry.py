"""Telemetry and metrics for Recall operations.

Light-weight in-process metrics. v2 will export to Prometheus / OpenTelemetry.

Tracks:
  - call counts per operation (observe / recall / bounded_generate / forget / consolidate)
  - latency histograms (p50 / p95 / p99)
  - rejection counts per gate (provenance / quality / dedup)
  - error counts

Usage (built into Memory automatically):
    print(mem.metrics.snapshot())
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from threading import Lock


class Metrics:
    """Process-local metrics collector."""

    def __init__(self):
        self._lock = Lock()
        self._counts: dict[str, int] = defaultdict(int)
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._errors: dict[str, int] = defaultdict(int)

    def increment(self, name: str, by: int = 1) -> None:
        with self._lock:
            self._counts[name] += by

    def observe_latency(self, name: str, seconds: float) -> None:
        with self._lock:
            self._latencies[name].append(seconds)
            if len(self._latencies[name]) > 10_000:
                self._latencies[name] = self._latencies[name][-5_000:]

    def error(self, name: str) -> None:
        with self._lock:
            self._errors[name] += 1

    @contextmanager
    def time(self, name: str):
        t0 = time.time()
        try:
            yield
        finally:
            self.observe_latency(name, time.time() - t0)
            self.increment(f"{name}_count")

    def snapshot(self) -> dict:
        with self._lock:
            out: dict = {"counts": dict(self._counts), "errors": dict(self._errors), "latency_p": {}}
            for name, vals in self._latencies.items():
                if not vals:
                    continue
                s = sorted(vals)
                n = len(s)
                out["latency_p"][name] = {
                    "p50": s[n // 2],
                    "p95": s[min(n - 1, int(n * 0.95))],
                    "p99": s[min(n - 1, int(n * 0.99))],
                    "count": n,
                    "mean": sum(s) / n,
                }
            return out


# Global default metrics (per-process). Per-tenant metrics live on Memory.
GLOBAL_METRICS = Metrics()
