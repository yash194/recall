# PRINCIPLES — Design and implementation discipline

> The rules every change has to pass before it lands.
> A short list, kept short on purpose.

---

## 1. Math first, code second

Every primitive in `src/recall/` corresponds to a section in
`docs/MATH.md`. If a behavior is novel, it gets a math section before
it gets an implementation. If a math section becomes stale (the code
diverges from the spec), the **math doc is updated, not the code's
behavior bent to match the doc** — but the doc reflects what the code
actually does, so code and spec stay in sync.

The four functions in `MATH.md §10` (`gamma`, `gamma_split`,
`hallucination_bound`, `bmrs_log_ratio`) are the mathematical heart of
Recall. They are not changed without a corresponding theorem update.

---

## 2. Every claim is testable, every test is hostile

Each theorem in `MATH.md` has a section `tests/test_*.py` that
mechanically verifies a non-trivial consequence. Each benchmark in
`benchmarks/` has a publicly-documented kill criterion (see
`MATH.md §9`).

A change that "works" but doesn't surface in the test suite or the
benchmark suite is a change we can't ship. If a test passes that
shouldn't, we add an adversarial input that breaks it.

We do **not** use the LLM that generates as the LLM that grades. All
benchmark scoring is exact-match or token-overlap.

---

## 3. Honest benchmarks, including the failures

`benchmarks/REPORT.md` tracks the historical v0.4 → v0.5 → v0.6
trajectory **including the v0.4 failure modes** (scope-filter bug,
scale-stress stall, MemoryAgentBench timeout). The benchmarks that
exposed those failures stayed in the repo; the JSON outputs that
documented them stayed checked in (`results_n15_k5.json` is still
present, even though the n=30 results superseded it).

When a benchmark says Recall lost (BM25 wins on single-session-user),
that loss is **reported in the headline table**, not buried.

When a number is mid-flight (e.g. scale stress n=10K not yet rerun),
it's marked "queued for vN" in the document, not omitted.

Self-reported numbers from third parties are **not** comparable to our
measurements. We never quote a Mem0 / Zep / Hindsight number as
authoritative; we either reproduce them in this repo or label them
"self-reported" in the comparison table.

---

## 4. Public data only

Every dataset Recall benchmarks against is on Hugging Face or another
public archive (LongMemEval-cleaned, MemoryAgentBench, HotpotQA). The
download command is in the script. No proprietary corpora,
no shadow benchmarks. Anyone with internet and a laptop can run
`./benchmarks/run_all.sh` and get the same numbers (modulo
floating-point and CPU-thermal noise).

If a third-party paper claims numbers we can't reproduce, we say so.

---

## 5. Scope-isolated tenants, audit everything

Every `Memory(tenant=...)` is a hard isolation boundary. Cross-tenant
reads are not possible from the public API. Every state change
(`observe`, `forget`, `consolidate`) writes one append-only row to
`audit_log` with actor, timestamp, target, payload, reason.

`mem.forget(node_id, reason, actor)` cascades through incident edges
*and* logs the cascade. There is no silent deletion. The forgetting is
auditable, explainable, and reversible (the audit log records what was
forgotten so it could be rebuilt from upstream state if needed).

---

## 6. Bounded active set under continuous ingest

The substrate is required to *not* grow without bound under continuous
write pressure. BMRS pruning (`MATH §4`) deactivates weak edges in
sleep-time consolidation; the active node count is what retrieval pays
for, not the historical write count. Storage retains every audit
entry, every drawer, every node — but retrieval only sees the active
subset. This is the "bounded-feeling, finite-actual" promise.

If a benchmark shows the active set growing unboundedly, that's a bug,
not a feature.

---

## 7. No premature abstraction; no half-finished implementations

Quoted from CLAUDE.md guidance:

> Don't add features, refactor, or introduce abstractions beyond what
> the task requires. A bug fix doesn't need surrounding cleanup; a
> one-shot operation doesn't need a helper. Don't design for
> hypothetical future requirements. Three similar lines is better than
> a premature abstraction. No half-finished implementations either.

Every `# TODO` in the codebase has a tracked task or a `vN.M roadmap`
note in `CHANGELOG.md`. There are no abstract base classes for hypothetical
future implementations; if v0.7 needs FAISS, we'll add the FAISS
backend behind the `Storage` protocol when v0.7 starts, not now.

---

## 8. Small public surface, stable across versions

The public `Memory` API has five methods: `observe`, `recall`,
`bounded_generate`, `trace`, `forget`. Plus three v0.5+ helpers:
`bulk_observe`, `consolidate(induce_edges=...)`, the `RecallAdapter`
mode parameter. Backward-compatible additions only.

Internal protocols (`Storage`, `Embedder`, `LLMClient`) can change
between versions; the public API of `Memory` does not break.

---

## 9. Fail closed on hallucination, fail loud on inconsistency

`bounded_generate(query, bound="strict")` raises
`HallucinationBlocked` if any extracted claim is not structurally
supported by the retrieved subgraph. The default is `strict`. There is
no silent fallback to "best guess."

`mem.consolidate()` returns the H¹ frustration score on the active
subgraph. A frustration > 0 is reported in telemetry; the consolidator
flags the frustrated cycle in the audit log. Inconsistency does not go
unrecorded.

---

## 10. Hardware: a laptop should be enough

The default install runs entirely on CPU — BGE-small-en-v1.5 encodes
at ~30 ms / call on a MacBook Apple Silicon. The full benchmark sweep
finishes in under an hour. There is no GPU dependency, no required
external service, no API key needed for the default config (an
LLM key is needed only for the CRC bound calibration benchmark and for
`OpenAIClient`). Developers should be able to clone, install, and run
all 154 tests and the benchmark suite without provisioning anything.

If a contribution makes the local benchmark suite require a GPU or
require an external service, it doesn't land in the open-source
default. Optional extras (BGE-M3, Voyage, Qdrant) live behind
`pip install -e ".[extras]"` and are documented as such.

---

## 11. Versioning: three numbers, semantic

- `vMAJOR` — breaking changes to the public `Memory` API.
- `vMINOR` — new capabilities, backward-compatible (e.g. v0.5 added
  `fast=True`; v0.6 added `embed_symmetric`). Schema migrations are
  automatic and tested.
- `vPATCH` — bug fixes, perf improvements, no behavioral change.

The `CHANGELOG.md` has a section per `vMINOR`. The current major is
`v0.x` because the public API is still pre-1.0; we'll cut `v1.0` when
the substrate is production-ready at >50K nodes per tenant
(post-FAISS swap).

---

## 12. The list of things we deliberately said no to

These are open questions, deferred on purpose:

- **No multi-modal nodes (image/audio drawers).** v3+. Out of scope
  for the typed-edge text substrate.
- **No federated graph sync.** Privacy + audit semantics are
  incompatible with cross-tenant edge propagation. Never going to ship.
- **No real-time multi-agent write contention.** v2 problem (would
  require Postgres-style locks; SQLite WAL is fine for embedded).
- **No KuzuDB / Neo4j adapter.** SQLite + the `_EmbeddingCache` is
  enough through ~50K nodes. Beyond that, FAISS HNSW behind the same
  Storage protocol.
- **No automatic LLM-judge for benchmarks.** Exact-match scoring or
  reproducible token-overlap. Period.

Each of these is reconsidered on its own merits if the use case
demands it. None is a "soft no."

---

*Principles document — Recall v0.6 — 2026-05-07. Short on purpose.
Each rule has saved us from a bad change at least once.*
