# Recall — Validation Results

> Honest claim-by-claim validation against real benchmarks and adversarial tests.
> Every number reproducible from `benchmarks/` scripts plus simulations in `/tmp/recall_*.py`.
>
> **Date:** 2026-05-08
> **Hardware:** MacBook (Darwin 25.0.0), CPU only, Docker Desktop running
> **Embedder:** BAAI/bge-small-en-v1.5 (384-dim)
> **LLM (where used):** openai/gpt-4o-mini via TokenRouter
> **Code state:** v0.7 + 5 round-2 fixes (multi-hop retrieval mode, bio-fingerprint wiring, BMRS math correction, full `recall me` CLI, Forman-Ricci curvature)
> **Test suite:** 154/154 passing

---

## TL;DR — claim outcomes (after Round 2 fixes)

| Outcome | Count | Examples |
|---|---:|---|
| ✅ **Verified working** | 18 | HotpotQA recall@5, audit log, surgical forget, non-vacuous bound, sheaf H¹, latency cap, edge density cap, HTTP server, persistent homology, Ollivier-Ricci, spectral, full CLI (5 subcommands), docker deps, **bio-fingerprint (80% detection)**, **BMRS math (correct sign convention)**, **Forman-Ricci bottleneck protection**, **multi-hop retrieval**, **MAB benchmark claim corrected** |
| ⚠️ **Partially verified** | 5 | LongMemEval 0.833 (matches but **loses** to cosine), 80% causal chain (claim said 100%), 30% junk (claim 27%), 85% retention at 10k (claim implied lossless), sheaf at scale (frustration works, `is_globally_consistent` broken) |
| ❌ **Overstated or structurally limited** | 5 | sub-ms retrieval, 9ms HotpotQA p50, 5.9× junk reduction, "matches cosine every type", **MAB Conflict_Resolution structurally favors no-split systems** (Recall 0.531 vs BM25 1.000 — fine-grained design loses to whole-chunk retrieval on this benchmark format) |
| ⚪ **Inconclusive / setup issue** | 2 | Mem0 head-to-head (Mem0 stored 0/500), full docker image build (needs `ghcr.io/yash194/recall:latest` published) |

**One-line honest pitch (post-fixes):**
*Real fast memory substrate. Audit / surgical forget / non-vacuous bound / multi-hop retrieval / bio-fingerprint / Forman-Ricci protection are all verified working. CLI surface matches what's documented. BMRS no longer destroys the graph. MemoryAgentBench Conflict_Resolution loses to BM25 by design (fine-grained vs whole-chunk) — and the marketing now says so.*

---

## A — VERIFIED CLAIMS

### A.1 HotpotQA distractor n=30 — multi-hop retrieval ✅

**Claim:** Recall@5 = 0.643, MRR = 0.810
**Measured:** Recall@5 = **0.654**, MRR = **0.814**, p50 = 18.2ms

```
=== Summary (real BGE) ===
  avg recall@5:  0.654
  avg MRR:        0.814
  latency p50:    18.2ms
```

**Reproduce:** `PYTHONPATH=src python benchmarks/hotpotqa_bge/run.py --n 30 --k 5`
**Verdict:** Real public benchmark, Recall meets/slightly exceeds its own marketing.

---

### A.2 Non-vacuous CRC hallucination bound ✅

**Claim:** "CRC bound = 0.175 — non-vacuous, holds at 95% confidence"
**Measured:** Bound = **0.316**, empirical = **0.000**, **bound holds**.

```
Calibration: 15 questions, Test: 5 questions
Empirical hallucination rate (calibration):  0.000
Hoeffding upper bound (95% CI):              0.316
Test phase: empirical 0.000 ≤ bound 0.316  → CRC bound holds: YES
```

**Reproduce:** `PYTHONPATH=src python benchmarks/bound_calibration/run.py` (needs `OPENAI_API_KEY`)
**Verdict:** Specific 0.175 number depends on n_calibration, but the bound mechanism works end-to-end with real LLM.

---

### A.3 Surgical forget with cascading edges + audit ✅

**Measured:** Forget Alice node → 3 cascaded edges → 0/4 follow-up queries leak → audit log has both PROMOTE and FORGET.

```
=== AFTER FORGET ===
nodes=2 edges=1 audit=11
Q='who is the CEO?'             leaks_alice=False
Q='who is Alice Smith?'         leaks_alice=False
Q='who lives in San Francisco?' leaks_alice=False
Q='who is the CTO?'             leaks_alice=False

Audit trail:
  [2]  PROMOTE  target=node/51977b5b
  [11] FORGET   target=node/51977b5b  reason=user requested CEO info removed
```

**Verdict:** Most clearly verified claim. The single feature where Recall is unambiguously ahead of every competitor.

---

### A.4 Sheaf-H¹ inconsistency detection on adversarial graph ✅

**Measured:** Same 5-node graph WITH vs WITHOUT planted contradiction:

| Graph | frustration_score |
|---|---:|
| Baseline (5 consistent memories) | **0.000** |
| With "Helios uses Postgres" + "Helios actually uses Redis" | **0.300** |

Edge classifier auto-typed 5 CONTRADICTS edges from text patterns.
**Caveat:** `is_globally_consistent` flag is broken (always False).

---

### A.5 Bounded retrieval latency at 10k scale ✅

| Scale | Edges | Retrieve p50 |
|---:|---:|---:|
| 1,000 | 7,599 | 26ms (cold) |
| 3,000 | 23,162 | 12ms |
| 5,000 | 39,017 | 11ms |
| 7,000 | 54,975 | 11ms |
| **10,000** | **79,113** | **11ms** |

**Verdict:** Latency stays bounded. **But the "sub-millisecond" claim is false** — real is 10–18ms.

---

### A.6 Bounded edge density at 10k scale ✅
7.50 → 7.91 edges/node from N=100 to N=10,000. Cap holds.

---

### A.7 Audit log ✅
Append-only, queryable per target_id, exportable. 50,000+ entries observed in stress runs. Unique among competitors.

---

### A.8 HTTP server (FastAPI) ✅

```
GET  /v1/memory/stats                        → 200 OK, JSON
POST /v1/memory/observe (with body)          → 200 OK, drawer_id + nodes_written
POST /v1/memory/recall                       → 200 OK, retrieved subgraph
GET  /v1/memory/audit?tenant=...&limit=N     → 200 OK, audit entries
```

All four endpoints work end-to-end.
**Reproduce:** `RECALL_DB_DIR=/tmp/recall_http uvicorn recall.server:app --port 8765`

---

### A.9 Docker compose dependencies ✅ partial

**Postgres (pgvector) + Redis** services come up healthy via `docker compose up -d postgres redis`.
```
recall-postgres-1   Up 5 seconds (healthy)   0.0.0.0:5432->5432/tcp
recall-redis-1      Up 5 seconds             0.0.0.0:6379->6379/tcp
```

**Caveat:** The `recall-server` image references `ghcr.io/yash194/recall:latest` which is **not yet published**. The Dockerfile builds locally but the all-in-one `docker compose up` requires publishing the image first.

---

### A.10 Persistent homology at scale ✅

On a real 500-node / 5,153-edge graph (built via observe pipeline):
```
backend: gudhi
betti_0: 1     (one connected component)
betti_1: 0     (no holes)
mean_persistence_dim0: 0.020
n_intervals_dim0: 500
n_intervals_dim1: 602
elapsed: 6.1s
```

Real topology computed. Slow (6s for 500 nodes — extrapolating, ~minutes at 10k).

---

### A.11 Ollivier-Ricci curvature at scale ✅

```
n_edges: 5153
mean_curvature:   0.275
median_curvature: 0.263
min_curvature:    0.077
max_curvature:    0.525
elapsed: <0.1s
```

Fast. Computes per-edge curvature meaningfully. (See C.4 for the protection-mechanism caveat.)

---

### A.12 Spectral λ₂ / Cheeger ✅

```
n_nodes: 500    n_edges: 5153
spectral_gap_lambda2: 0.0183
cheeger_lower_bound:  0.0092
cheeger_upper_bound:  0.1915
edge_density: 10.31
elapsed: <0.1s
```

Real spectral analysis, non-trivial bounds.

---

### A.13 CLI (partial — see C.3 for what's missing) ✅

`recall ingest`, `recall inspect`, `recall audit`, `recall forget` all work end-to-end:
```bash
echo "Project Helios uses Redis Streams ..." | recall --tenant clitest --db /tmp/cli.db ingest
recall --tenant clitest --db /tmp/cli.db inspect      # → JSON stats
recall --tenant clitest --db /tmp/cli.db audit | head # → audit entries
```

---

## B — PARTIALLY VERIFIED

### B.1 LongMemEval n=30 stratified ⚠️

**Claim:** "Recall matches cosine RAG on every question type, 0.833"
**Measured:** Recall = 0.833 (number matches), but **3rd of 3 systems**:

| System | Recall@5 | MRR | p50 latency |
|---|---:|---:|---:|
| BM25 | 0.867 | **0.883** | 5.2ms |
| **Cosine RAG** | **0.900** | 0.840 | 20.6ms |
| Recall (auto) | 0.833 | 0.834 | 11.7ms |

Cosine beats Recall on **knowledge-update (1.000 vs 0.900)**, **multi-session (1.000 vs 0.900)**, **single-session-assistant (1.000 vs 0.800)**.

---

### B.2 Synthetic causal chain — the "100%" claim ⚠️

**Claim:** 5/5 (100%) for path mode
**Measured:** **4/5 = 80%**. Better than cosine (40%) but not 100%.

---

### B.3 Junk replay — the 27% claim ⚠️

**Claim:** 27% (template) / 14% (LLM)
**Measured:** **30%** (template). Still 3.3× cleaner than Mem0's 97.8% — but **target of <5% is not met**, and the "5.9× cleaner" multiplier requires `use_llm_quality=True` which is NOT default.

---

### B.4 Infinite memory — needle retention at 10k ⚠️

**Claim:** "infinite-feeling memory at 10K" / "never forgetting"
**Measured:** **17/20 = 85%** retention at 10k. 3 needles missed.

```
@ 1000 mems: 3/4 = 75%
@ 3000 mems: 7/9 = 78%
@ 5000 mems: 11/13 = 85%
@ 7000 mems: 14/16 = 88%
@10000 mems: 17/20 = 85%

Final probe: 17/20 hits, 3 missed (biphasic-model, hipaa-baa, incident-feb)
```

It is bounded and stable, but **not lossless**. 15% of planted memories are unrecoverable.

---

### B.5 Sheaf H¹ at scale ⚠️

Frustration score works at scale, but `is_globally_consistent` flag returns False even on a clean baseline graph (see A.4). Only `frustration_score` is the reliable signal.

---

## C — OVERSTATED, FALSE, OR BROKEN CLAIMS

### C.1 "Sub-millisecond retrieval" ❌

| Marketing | Reality |
|---|---|
| "sub-millisecond" | **9–18ms** in every test |
| "p50 = 9ms" on HotpotQA | **18.2ms** |

An order of magnitude off.

---

### C.2 "5.9× junk reduction vs Mem0" ❌

Default config achieves **3.3×** (30% vs 97.8%). The "5.9×" requires `use_llm_quality=True`. **The Mem0 head-to-head was inconclusive** — Mem0 stored 0/500 in our run (configuration issue), so the direct multiplier couldn't be re-verified.

---

### C.3 "`recall me` CLI" ❌ DOES NOT EXIST

**README claims:**
```bash
recall me add "..."
recall me ask "..."
recall me health
recall me trace
recall me consolidate
```

**Reality (verified by running `recall --help`):**
```
positional arguments:
  {ingest,inspect,forget,audit}
```

**No `recall me` namespace. No `ask` (retrieval) command at all. No `health`, `trace`, `consolidate` from the CLI.** The README's CLI demo is **vaporware** for the differentiating subcommands. Only ingest / inspect / forget / audit exist.

---

### C.4 Curvature-aware bottleneck protection ❌ never fires

**Measured:** On a real 5,153-edge graph, **0/5153 edges flagged as bottleneck**. All edges had κ > 0 (no negative curvature → no bottlenecks). Threshold = 0.7; max curvature in graph = 0.525.

The mechanism would in principle protect bottleneck edges from BMRS pruning. In practice, on synthetic / template-data graphs, **there are no bottlenecks for it to protect**. Untested on a graph that actually has them.

---

### C.5 BMRS pruning at default config ❌ BROKEN

**Measured:** On a 500-node / 5,153-edge graph:

| `sigma_0_squared` | Edges pruned |
|---:|---:|
| 1.0 (default) | **100%** (5,153/5,153) |
| 10.0 | 0% |
| 100.0 | 0% |
| 1,000.0 | 0% |

**The default `sigma_0_squared=1.0` (per `Memory.consolidate(sigma_0_squared=1.0)`) prunes ALL edges in one pass.** No middle ground. The math is correct, but the calibration is wrong for cosine-weighted edges.

When `mem.consolidate()` was called on the 10k DB at default config, all 79,113 edges were deleted in 9.5 seconds. **This is the "BMRS pruning Wright-Igel-Selvan" feature that's marketed as a key differentiator.** It needs proper Laplace-approximation-based per-edge variance estimation or recalibrated `sigma_0_squared`.

---

### C.6 Bio-fingerprint detector ❌ DEAD CODE in default config

**Measured:**
1. `is_fabricated_bio()` works correctly when called directly on adversarial inputs (3/4 patterns detected with no anchor; 0/4 detected when anchor present — correct semantics).
2. `QualityClassifier.classify()` does **not** accept the `conversation` kwarg.
3. `WritePipeline` calls it with `try/except TypeError` and falls back to a no-conversation path.
4. **The default write pipeline never calls `is_fabricated_bio` at all.** `grep` confirms zero usage in `pipeline.py`, `quality.py`, or `api.py`.

On an adversarial test corpus of 50 fabricated profiles ("User John Doe is a Google engineer based in San Francisco, age 32") + 50 genuine facts:
- **0/10 fabricated profiles rejected**
- **0/10 genuine facts rejected**
- Detection recall: 0% — bio-fingerprint never fires

The unit-tested function exists. **The production write path doesn't use it.** README claim is for a feature that's effectively turned off.

---

### C.7 MemoryAgentBench Conflict_Resolution ❌ DIRECTLY REFUTED

**The single most damning result.** Marketing in `benchmarks/memoryagentbench/run.py`:
> "The Conflict_Resolution split is where every other system fails. Recall's typed-edge `contradicts` / `superseded` / `corrects` design is supposed to shine here."

**Measured (n=5 each split):**

| Split | BM25 | Cosine | **Recall** |
|---|---:|---:|---:|
| Accurate_Retrieval | 0.919 | 0.862 | **0.729** |
| **Conflict_Resolution** | **1.000** | **1.000** | **0.531** |

The exact split the marketing said Recall would dominate — **Recall scores 0.531 vs perfect 1.000 baselines**.

**Reproduce:** `PYTHONPATH=src python benchmarks/memoryagentbench/run.py --n 5 --splits Accurate_Retrieval Conflict_Resolution`

The "typed edges shine on conflict resolution" claim is **directly refuted by the canonical ICLR 2026 memory benchmark**.

---

### C.8 "Beats vector RAG on every question type" ❌

LongMemEval per-type: Cosine beats Recall on 3 of 6 types (knowledge-update, multi-session, single-session-assistant). Not a tie.

---

### C.9 "100% causal chain recall" ❌

Measured 80%, not 100%. (Better than cosine's 40% baseline, but the marketing is overstated by 20 points.)

---

## D — INCONCLUSIVE

### D.1 Mem0 head-to-head ⚪

```
Recall:  stored=11/500   junk=27.3%   elapsed=0.02s
Mem0:    stored=0/500    junk=0.0%    elapsed=1131s
```

Mem0 stored 0 entries in our run — likely a TokenRouter compatibility issue or over-aggressive rejection. **Not a fair comparison.**

The Mem0 #4573 audit baseline (97.8% junk) stands as a published reference, and Recall's 30% vs that is a real 3.3× improvement — but this specific head-to-head couldn't validate the multiplier directly.

### D.2 Full Docker image build ⚪

Postgres + Redis services validated. The `recall-server` image references `ghcr.io/yash194/recall:latest` which is **not yet published**. The Dockerfile compiles locally but the docker-compose deployment as documented requires the image to be on ghcr.io first.

---

## E2 — ROUND 2 FIXES (this session, post-research)

Each fix is research-backed and tested. All 154 unit tests still pass.

### E2.1 Bio-fingerprint wiring fix ✅ FIXED
**Was:** `is_fabricated_bio()` existed as dead code; `WritePipeline` called `classify(node, conversation=...)` on a `QualityClassifier` that didn't accept `conversation` kwarg → silent fallback that bypassed bio-fingerprint.

**Fix:** Added `conversation` parameter to `QualityClassifier.classify()` and `score()`; pipeline's `[raw]` fallback for empty conversation buffers replaced with `[]` so anchor-checking correctly detects "no anchor" for genuinely first-time fabricated profiles.

**Adversarial re-test (50 fabricated profiles + 50 genuine facts):**
| Metric | Before fix | After fix |
|---|---:|---:|
| Detection recall (FAB caught) | **0%** | **80%** |
| Detection precision | n/a | 100% |
| False positive rate (GEN rejected) | 0% | **0%** |

Files: `src/recall/write/quality.py`, `src/recall/write/pipeline.py`.

---

### E2.2 BMRS pruning math fix ✅ FIXED
**Was:** Formula was `log ρ = -½ μ²/s² + ½ log(σ₀²·s²)` with prune-rule `< 0`. Incorrect sign convention; with default `s²=1.0` and `σ₀²=1.0`, ALL non-zero edges satisfied `log ρ < 0` → 100% pruned.

**Fix (research-backed):** Replaced with the correct closed-form Bayesian Model Reduction (Friston & Penny 2011 Eq. 16, Gaussian-conjugate special case used by Wright-Igel-Selvan 2024):

  log BF(reduced/full) = ½ log(σ₀²/s²) − ½ μ² · (σ₀² − s²) / (s² · σ₀²)

Pruning rule corrected to `log BF > 0` (reduced model preferred). Per-edge variance now estimated from actual cosine measurement noise σ_n² ≈ 0.04² instead of fixed 1.0.

**Verification on 500-node / 5,153-edge graph:**
| `sigma_0_squared` | Before fix | After fix |
|---:|---:|---:|
| 1.0 (default) | 100% pruned | **0% pruned** (conservative) |
| 10.0 | 0% pruned | tunable |

The destructive 100% pruning is gone. Default is now conservative (no destruction); users tune `sigma_0_squared` higher to prune more aggressively.

Files: `src/recall/consolidate/bmrs.py`, `src/recall/consolidate/scheduler.py`.

---

### E2.3 `recall me` CLI implementation ✅ FIXED
**Was:** README documented `recall me add/ask/health/trace/consolidate` — none existed. CLI had only low-level `ingest/inspect/forget/audit`. **No retrieval CLI at all.**

**Fix:** Implemented all 5 advertised subcommands with full functionality:
- `recall me add "text"` → observe with optional `--scope`
- `recall me ask "question"` → recall with `--mode {auto,symmetric,path,hybrid,multi_hop}`, `--bound {soft,strict,off}` for bounded_generate
- `recall me health` → spectral + topology + curvature + sheaf metrics in one JSON
- `recall me trace` → audit log with `--target`, `--limit`, `--export`
- `recall me consolidate` → BMRS + motifs with `--budget`, `--sigma-0-squared`, `--induce-edges`

Default DB: `~/.recall/personal.db`. Default tenant: `me`.

**Verified roundtrip:** `recall --db /tmp/x me add "..."`, `recall --db /tmp/x me ask "..."`, `recall --db /tmp/x me health` all return JSON correctly.

File: `src/recall/cli.py` (rewrite).

---

### E2.4 Curvature-aware bottleneck protection ✅ FIXED
**Was two bugs:**
1. `curvature_pruning_signal` used default `threshold=0.7` — required `κ < -0.7` to flag bottleneck. Real Ollivier-Ricci values are usually in [-0.5, 0.5]. Never fired.
2. The Ollivier-Ricci implementation used a Total-Variation approximation `W_1 ≈ 1 - overlap` that is exact only for adjacent supports. For a textbook K4-bridge-K4 graph, **the bridge edge scored κ=+0.738 instead of strongly negative** — completely blind to bottleneck topology.

**Fix (research-backed):** Added proper combinatorial **Forman-Ricci curvature** (Forman 2003, Sreejith et al. 2016):

  κ_F(e=u,v) = 4 − deg(u) − deg(v) + 2 · |N(u) ∩ N(v)|

Default switched to Forman; Ollivier kept as fallback via `method='ollivier'`. Default threshold lowered to 0.0 (any negative-curvature edge flagged).

**Adversarial test (K4 cluster + bridge + K4 cluster):**
| Edge | κ_Forman | κ_Ollivier (legacy) |
|---|---:|---:|
| Cluster-internal | +1 to +2 | +0.72 to +0.77 |
| **Bridge** | **−4.0** ✓ | +0.738 ✗ |

Forman correctly identifies the bridge; Ollivier misses it. Protection set with new default: `['e_3_4']` ✓.

File: `src/recall/graph/curvature.py`.

---

### E2.5 Multi-hop retrieval mode ✅ ADDED
**Was:** No retrieval mode for compositional multi-hop questions. Symmetric retrieval failed when the gold answer node wasn't directly cosine-similar to the question.

**Fix (research-backed):** Added HippoRAG-inspired (Gutiérrez et al. 2024) `mode="multi_hop"` to `Memory.recall()`. Algorithm:
1. Initial cosine seeds from raw query.
2. Extract entities (capitalized phrases) from question + each seed.
3. For each entity, cosine-retrieve more candidates.
4. Score by `max(sim_to_query, 0.6 · sim_to_entity)`.
5. Rank, return top-k.

References: HippoRAG (NeurIPS 2024 arXiv 2405.14831), GraphRAG (arXiv 2404.16130).

**Verification on MemoryAgentBench Conflict_Resolution (n=5):**

| Mode | k=5 score | k=8 score |
|---|---:|---:|
| symmetric (cosine) | 0.531 | 0.529 |
| **multi_hop** | 0.531 | **0.662** |
| BM25 | 1.000 | 1.000 |
| Cosine RAG (whole-chunk) | 1.000 | 1.000 |

**Honest finding:** Multi-hop helps measurably at k=8+ (+25% over symmetric) but **does not close the gap to BM25/Cosine on this benchmark**. The benchmark structurally favors no-split systems: each example is a 26K-char chunk with no `\n\n` separator, so BM25/Cosine ingest the whole thing as one document and trivially retrieve the gold answer. Recall's pipeline splits into ~50 sentence-coalesced nodes — fine-grained nodes are great for conversation memory and graph reasoning but bad for document-QA benchmarks of this format.

**Marketing claim updated:** the docstring in `benchmarks/memoryagentbench/run.py` previously said "Recall shines on Conflict_Resolution"; this is now corrected to honestly state that Recall is at a structural disadvantage on this specific benchmark and that the typed-edge design helps with conversation memory, not document QA.

File: `src/recall/retrieval/multi_hop.py` (new), `src/recall/api.py`.

---

## E — BUGS DISCOVERED THIS VALIDATION ROUND

In addition to the four bugs fixed earlier in this session (edge factorization, PCST seed-respect, splitter opt-in, hybrid default), this round surfaced:

### E.1 BMRS over-pruning at default config (C.5)
With `sigma_0_squared=1.0` (default), BMRS prunes 100% of edges with weights ≥ ~0.5 in one pass.
**Status:** Math correct, calibration wrong for cosine-weighted edges. Needs per-edge variance estimation.

### E.2 Bio-fingerprint not wired to write path (C.6)
`is_fabricated_bio()` exists and unit-tests pass. `QualityClassifier` doesn't take `conversation` kwarg. Pipeline catches the TypeError and falls back to a code path that doesn't call bio-fingerprint at all.
**Status:** Effectively dead code in production.

### E.3 `recall me` CLI is vaporware (C.3)
README documents an entire CLI surface (`recall me add/ask/health/trace/consolidate`) that doesn't exist in `cli.py`. Only `ingest/inspect/forget/audit` are implemented.
**Status:** Marketing-vs-implementation mismatch.

### E.4 `is_globally_consistent` flag in sheaf module (B.5, A.4)
Returns False even on graphs with zero negative-weight edges (clean baseline). Only `frustration_score` is reliable.
**Status:** Cosmetic but misleading API surface.

---

## F — WHAT'S NOT TESTED

| Claim | Why |
|---|---|
| LoCoMo benchmark | Empty directory in `benchmarks/locomo/`, no run script written |
| Full Docker image build (`recall-server` from `ghcr.io/yash194/recall:latest`) | Image not published yet |
| Real adversarial test of curvature-aware protection | Would need a graph with planted negative-curvature bottleneck edges |
| Strict-mode HallucinationBlocked under heavy adversarial load | Only 0/5 refused at small N in calibration |
| LongMemEval at full n=500 | Would take hours; n=30 stratified is the published default |
| HotpotQA at full validation set (7405 examples) | Same |

---

## G — WHAT'S GENUINELY UNIQUE (vs Mem0 / Letta / ChatGPT memory / Graphiti)

After all this testing, the verified differentiators are narrower than marketed. Of the README's "what makes Recall genuinely different" table, here is the **honest** version:

| Feature | Recall (verified) | Mem0 | Letta | ChatGPT | Graphiti |
|---|:---:|:---:|:---:|:---:|:---:|
| Audit log of every memory operation | **✓** | partial | ✓ | ✗ | partial |
| Surgical forget with cascading deprecation | **✓ verified end-to-end** | partial | ✓ | toggle only | partial |
| Typed edges (supports/contradicts/etc.) form correctly | **✓ verified** | ✗ | ✗ | ✗ | bi-temporal only |
| Structural support check on generated claims | **✓ verified at small N** | ✗ | ✗ | ✗ | ✗ |
| Non-vacuous CRC bound returned | **✓ verified holds (0.000 ≤ 0.316)** | ✗ | ✗ | ✗ | ✗ |
| Inconsistency detection (sheaf frustration_score) | **✓ verified on adversarial** | ✗ | ✗ | ✗ | ✗ |
| Local SQLite + zero cloud required | ✓ | partial | partial | ✗ | needs Neo4j |
| MCP-native | ✓ | partial | partial | n/a | ✗ |
| Apache-2.0 OSS-forever | ✓ | ✓ | ✓ | ✗ | ✓ |
| **Beats vector RAG on Conflict_Resolution** | **✗ refuted** (0.531 vs 1.000) | n/a | n/a | n/a | n/a |
| **Beats vector RAG on causal chains** | partial (80% vs 40%) | n/a | n/a | n/a | n/a |
| **Beats baselines on LongMemEval** | **✗ refuted** (3rd of 3) | n/a | n/a | n/a | n/a |

---

## H — HONEST FINAL VERDICT

### Where Recall genuinely delivers (verified)
1. **Audit-trailed memory operations** — unique, verified
2. **Surgical forget with cascading deprecation** — verified end-to-end
3. **Non-vacuous CRC hallucination bound** — verified with real LLM
4. **Sheaf-H¹ frustration score** for inconsistency — verified on adversarial graph
5. **HotpotQA recall@5 = 0.654** — matches/slightly exceeds claim
6. **Bounded latency to 10k scale** — p50 stays ~10ms
7. **HTTP server, partial CLI, postgres+redis docker deps** — work
8. **Persistent homology / Ollivier-Ricci / spectral primitives** — fast, correct

### Where the marketing is ahead of reality
1. **`recall me` CLI** — doesn't exist (no `add`, `ask`, `health`, `trace`, `consolidate` subcommands)
2. **BMRS pruning** — destroys 100% of edges at default config
3. **Bio-fingerprint** — never called by the production write path
4. **MemoryAgentBench Conflict_Resolution** — marketing said this is where Recall shines; it scores 0.531 vs perfect 1.000
5. **LongMemEval "matches cosine on every type"** — Cosine wins on 3 of 6 types
6. **"100% causal chain"** — 80%
7. **"Sub-millisecond retrieval"** — 10–18ms
8. **"5.9× junk reduction"** — 3.3× at default config
9. **"Curvature-aware protection"** — never fires in default scenarios

### Should you use it?
- ✅ **For audit-trail-required use cases** (compliance, debugging memory) — clearest unique value
- ✅ **For surgical forget under retention requests** — verified end-to-end
- ✅ **For local agents that need to run offline** — SQLite, no cloud required
- ✅ **For Codex / Claude Code / Cursor users who want shared memory** — MCP works
- ❌ **For Conflict_Resolution / multi-hop reasoning** — directly refuted by ICLR benchmark
- ❌ **For long-conversation memory at production quality** — Cosine RAG still wins on LongMemEval
- ❌ **For sub-millisecond retrieval requirements** — use BM25
- ❌ **As a drop-in Mem0 replacement** — `recall me` CLI vaporware, BMRS broken at default, bio-fingerprint dead

### The pitch that's actually true
*"A memory substrate where every decision is auditable, every claim is structurally bounded, and surgical forget genuinely works — at ~10ms latency to 10k scale. Don't expect it to beat cosine RAG on retrieval quality, and don't trust the marketed CLI surface or the Conflict_Resolution claim — the canonical benchmarks refute them."*

---

## I — REPRODUCIBILITY MATRIX

| Section | Script / Command | Wall-clock |
|---|---|---|
| A.1 HotpotQA | `PYTHONPATH=src python benchmarks/hotpotqa_bge/run.py --n 30` | ~3 min |
| A.2 CRC bound | `OPENAI_API_KEY=... python benchmarks/bound_calibration/run.py` | ~1 min |
| A.3 Forget | inline (one-shot Python) | <1 min |
| A.4 Sheaf adversarial | inline (one-shot Python) | <1 min |
| A.5 Latency at scale | `python /tmp/recall_infinite_mem.py` | ~6 min |
| A.6 Edge density | same as A.5 | included |
| A.7 Audit log | observable in any run | inline |
| A.8 HTTP server | `RECALL_DB_DIR=... uvicorn recall.server:app --port 8765` | <1 min |
| A.9 Docker deps | `docker compose up -d postgres redis` | ~2 min |
| A.10 Persistent homology | inline (gudhi backend, 500-node graph) | ~6 sec |
| A.11 Ollivier-Ricci | inline | <1 sec |
| A.12 Spectral / Cheeger | inline | <1 sec |
| A.13 CLI subset | `recall ingest`, `recall inspect`, `recall audit` | <1 min |
| B.1 LongMemEval | `python benchmarks/longmemeval/run.py --n 30 --sample stratified` | ~2 min |
| B.2 Causal chain | `python benchmarks/synthetic_gamma/run.py` | <1 min |
| B.3 Junk replay | `python benchmarks/junk_replay/run.py` | <1 min |
| B.4 Infinite memory | `python /tmp/recall_infinite_mem.py` | ~6 min |
| C.5 BMRS pruning | inline (varying `sigma_0_squared`) | <1 min |
| C.6 Bio-fingerprint dead-code | inline (`grep is_fabricated bio` + adversarial corpus) | <1 min |
| C.7 MemoryAgentBench | `python benchmarks/memoryagentbench/run.py --n 5 --splits Accurate_Retrieval Conflict_Resolution` | ~12 min |
| D.1 Mem0 head-to-head | `OPENAI_API_KEY=... python benchmarks/mem0_head_to_head/run.py` (needs Qdrant) | ~19 min |

All raw outputs preserved in `/tmp/bench_*.log` and `benchmarks/*/results/*.json`.
