# Recall — System State Document

**Date:** 2026-05-08
**Code state:** v0.7 + 5 round-2 fixes
**Test suite:** 154/154 passing
**Verified scale:** 10,000 nodes / 79,113 edges with bounded latency

> A complete, honest accounting of what Recall is today: what works, what
> doesn't, what can be engineered around, what needs research.

---

## 1 · Executive summary

Recall is a **memory substrate for AI agents** that stores conversational
memories as a typed-edge graph, retrieves connected reasoning paths, and
returns answers bounded by structural support. It runs locally on SQLite,
exposes an MCP server (Codex / Claude Code / Cursor), an HTTP server, and
a CLI.

**What it's genuinely best at:** audit-trailed memory operations, surgical
forget, structurally-bounded generation, sheaf-based inconsistency
detection, bio-fingerprint quality gating, multi-hop entity-expanded
retrieval, local-first deployment, MCP integration. None of the major
competitors (Mem0, Letta, ChatGPT memory, Graphiti) ship this combination.

**What it's not:** a drop-in production replacement for Mem0/Letta. It's
solo-engineered, has only Python SDK, no cloud sync, no hosted offering,
and is structurally disadvantaged on document-QA benchmarks where
whole-chunk retrieval beats fine-grained nodes (e.g., MemoryAgentBench
Conflict_Resolution).

---

## 2 · What it does (verified working surfaces)

### 2.1 Storage & Ingestion
- **SQLite-backed multi-tenant storage** — one DB per tenant, scope filters
  by JSON subset semantics
- **Hash deduplication on drawer + node text** — exact repeats dropped
  silently with audit log
- **Provenance firewall** — sources tagged `recall_artifact` /
  `agent_self_recall` / `memory_dump` are rejected at write time
- **Quality gate** — template-match negative patterns (boilerplate,
  system-prompt leaks, hallucinated bios) reject low-quality content
- **Bio-fingerprint anchor-checking** — rejects fabricated profile claims
  when ≥3 attribute types co-occur with no anchor in conversation buffer
  (80% detection / 100% precision on adversarial test, post-fix)
- **Sentence splitter** — coalesces sentences into ≥80-token chunks for
  meaningful node granularity
- **Optional LLM splitter** — opt-in via `use_llm_splitter=True` for
  paragraph-scale ingest (default off because TokenRouter calls per-write
  drop throughput from 86/s to 0.5/s)
- **Bulk-mode fast path** — sources tagged `document` / `corpus` skip the
  conversation buffer scan and Γ-edge induction (still split sentences)
- **Append-only audit log** — every WRITE / PROMOTE / REJECT / SKIP /
  PRUNE / FORGET / GENERATE operation logged with seq, timestamp, actor,
  reason, payload — queryable per target_id, exportable as JSONL

### 2.2 Edge formation (v0.7 factorized)
- **Cosine gates existence** — top-K symmetric-similarity neighbors
- **Γ_anti gates direction** — normalized by ||c_i||·||c_j|| for embedder
  scale invariance
- **Γ_sym + text patterns gate type** — `supports / contradicts /
  corrects / agrees / pivots / temporal_next / superseded`
- **Bounded density** — `MAX_EDGES_PER_NODE=6` cap per write; verified
  steady at 7.9 edges/node from 100 → 10,000 scale

### 2.3 Retrieval modes
| Mode | Description | When to use |
|---|---|---|
| `symmetric` | Top-K cosine on raw symmetric embedding | Fast, factual lookup |
| `path` | Γ-walk seeded by cosine, PCST extraction with `must_include` seeds | Direction-aware reasoning |
| `hybrid` | Reciprocal Rank Fusion of symmetric + path | Default for `bounded_generate` |
| `auto` | Graph-aware router picks mode based on query cues + graph density | General use |
| `multi_hop` | HippoRAG-style entity expansion with score = max(query_sim, 0.6·entity_sim) | Compositional questions where answer isn't directly cosine-similar |
| (algo opts) | `pcsf`, `networkx` Steiner-tree, `ppr` Personalized PageRank | Special-case retrieval algorithms |

### 2.4 Bounded generation
- **`bounded_generate(query, bound={strict|soft|off})`** — retrieves a
  subgraph, generates an answer using only that context, structurally
  checks each generated claim against the retrieval, returns:
  - generated text
  - per-claim flagged list (claims with no structural support)
  - composite hallucination bound (RAG-noisy-ICL + spectral, Zhang 2025 +
    arXiv 2508.19366)
- **`strict` mode** raises `HallucinationBlocked` on any flagged claim
- **`soft` mode** returns flagged claims as a list
- **CRC bound calibration** — verified non-vacuous (0.316 at n=15,
  empirical 0.000) with real LLM via TokenRouter

### 2.5 Forget
- **Surgical node deletion with cascading edges** — `forget(node_id,
  reason)` deprecates the node + all incident edges, audit-logs both
- **Deprecated nodes/edges remain in storage** for audit replay but are
  excluded from active retrieval
- **Verified end-to-end** — Alice node forgotten → 0 leaks across 4
  follow-up queries → audit preserved

### 2.6 Sleep-time consolidation
- **BMRS pruning** — Friston-Penny BMR formula (corrected sign convention
  in v0.7); per-edge variance estimated from cosine measurement noise.
  At default `sigma_0_squared=1.0`, prunes 0% of edges (conservative);
  user tunes higher to prune more aggressively
- **Mean-field GNN refinement** — smooths edge weights over kept edges
- **Motif extraction** — finds recurring subgraph patterns
- **Curvature-aware bottleneck protection** — Forman-Ricci κ < 0 edges
  protected from pruning even at high `sigma_0_squared`
- **Optional Γ-edge induction for isolated nodes** —
  `consolidate(induce_edges=True)` after bulk ingest

### 2.7 Graph health metrics
- **Spectral gap λ₂ + Cheeger lower/upper bounds** —
  `recall.graph.spectral.graph_health()`
- **Persistent homology** — Betti numbers, persistence intervals via
  `gudhi`
- **Forman-Ricci curvature** — combinatorial, per-edge, fast (<1ms for
  5k edges); identifies bottlenecks
- **Ollivier-Ricci curvature** (legacy, TV approximation) — kept for
  backward compat; blind to bottlenecks (don't use for protection)
- **Sheaf H¹ frustration score** — verified to detect planted
  contradictions (0.0 baseline → 0.3 with adversarial contradiction)
- **Personalized PageRank** — for HippoRAG-style retrieval over typed-edge
  graph

### 2.8 Public surfaces
- **Python library**: `from recall import Memory` — 5 public methods
  (observe, recall, bounded_generate, forget, consolidate) + `trace`,
  `audit`, `stats`
- **CLI**: `recall ingest/inspect/forget/audit` (low-level) + `recall me
  add/ask/health/trace/consolidate` (personal namespace) — all 9
  subcommands implemented and tested
- **HTTP server (FastAPI)**: 6 REST endpoints —
  `POST /v1/memory/observe`, `POST /v1/memory/recall`,
  `POST /v1/memory/bounded_generate`, `POST /v1/memory/forget`,
  `GET /v1/memory/audit`, `GET /v1/memory/stats`
- **MCP server (stdio)**: 8 tools — `add_memory`, `search_memory`,
  `bounded_answer`, `forget`, `audit`, `graph_health`, `consolidate`,
  `stats` — registered in Claude Code (user scope) and Codex CLI
- **Docker compose**: postgres + redis services come up healthy; the
  recall-server image references unpublished `ghcr.io/yash194/recall:latest`

---

## 3 · What's been achieved (verified evidence)

### 3.1 Real benchmark results

| Benchmark | Result | Comparison |
|---|---|---|
| HotpotQA distractor n=30 Recall@5 | **0.654** | matches/exceeds claimed 0.643 |
| HotpotQA MRR | **0.814** | matches claimed 0.810 |
| HotpotQA p50 latency | 18.2ms | claim was 9ms (overstated) |
| LongMemEval n=30 stratified Recall@5 | 0.833 | Cosine 0.900 (loses by 7pts) |
| LongMemEval MRR | 0.834 | BM25 0.883 (loses) |
| Synthetic causal chain (5-step) | 4/5 = 80% | Cosine 40% (wins; claim was 100%) |
| Scale-stress 5k mems gold-recall@5 | 0.80 | matches Cosine 0.80 |
| Infinite-memory 10k needle retention | 17/20 = 85% | claim implied lossless |
| Junk rate (template gate, n=300) | 30% | Mem0 97.8% (3.3× cleaner; claim was 5.9×) |
| MAB Accurate_Retrieval n=5 | 0.729 | BM25 0.919 / Cosine 0.862 (loses) |
| MAB Conflict_Resolution n=5 | 0.531 (sym), 0.662 (multi_hop k=8) | BM25 / Cosine 1.000 (loses by design) |
| CRC bound calibration | 0.316 bound, 0.000 empirical (holds) | claim was 0.175 |

### 3.2 Adversarial / unit-tested capabilities

| Test | Result |
|---|---|
| Surgical forget end-to-end | 0/4 leaks after forget; audit preserved (verified) |
| Sheaf H¹ on planted contradiction | frustration 0.0 → 0.3 (verified) |
| Bio-fingerprint on 50 FAB + 50 GEN | 80% recall, 100% precision (verified) |
| Forman-Ricci on K4-bridge-K4 | bridge κ=−4, cluster +1 to +2 (verified) |
| Multi-hop entity-expansion | +25% over symmetric on Conflict_Resolution at k=8 |
| Graph density bound at 10k | 7.9 edges/node steady (verified) |
| Latency at 10k | p50 = 9.7ms steady (verified) |
| Insert throughput | 86/s steady at 1.5k (verified, post splitter fix) |
| BMRS at default config | 0% pruned (post-fix; was 100% pre-fix) |

### 3.3 Bugs fixed (round 1 + round 2)

**Round 1 (algorithmic):**
1. **Edge formation degenerate with BGE** (`active_edges: 0`) — refactored
   to factorize: cosine gates existence, Γ_anti gates direction, Γ_sym +
   patterns gate type. Edges now form (10,909 at 10k nodes).
2. **PCST drops query seeds** — added `must_include` parameter so the
   high-cosine seed is a required terminal. Path-mode needle hit rate
   went from 1/5 to 5/5.
3. **`bounded_generate` defaulted to broken `path` mode** — switched
   default to `hybrid`.
4. **LLM splitter on per-write** (0.5/s with TokenRouter) — made opt-in
   via `use_llm_splitter=True`. Insert rate jumped to 86/s.

**Round 2 (rigorous, research-backed):**

5. **Bio-fingerprint dead code** — `is_fabricated_bio` was never called
   from write path; `QualityClassifier.classify()` lacked `conversation`
   kwarg. Added kwarg, wired up, fixed pipeline's `[raw]` fallback.
   Detection: 0% → 80%.
6. **BMRS math error** — formula had wrong sign convention (prune-rule
   inverted) and used fixed `s²=1.0` that destroyed all edges. Replaced
   with correct Friston-Penny 2011 BMR closed-form
   (`log BF = ½ log(σ₀²/s²) − ½ μ²(σ₀²−s²)/(s²σ₀²)`) and per-edge
   `cosine_edge_variance` from measurement noise.
7. **`recall me` CLI vaporware** — README documented 5 subcommands that
   didn't exist. Implemented all 5 (`add`, `ask`, `health`, `trace`,
   `consolidate`).
8. **Curvature-aware protection blind to bottlenecks** — Ollivier-Ricci
   used a TV-approximation that scored bridges at +0.7. Added correct
   combinatorial Forman-Ricci as default. Bridge in K4-bridge-K4 now
   correctly scores κ=−4.
9. **Multi-hop retrieval** — added `mode="multi_hop"` (HippoRAG-inspired)
   for compositional questions where the gold answer isn't cosine-similar
   to the question.
10. **MAB marketing claim** — corrected the "Conflict_Resolution is where
    Recall shines" claim in `benchmarks/memoryagentbench/run.py` to
    honestly document the structural mismatch between fine-grained
    sentence-split design and document-QA benchmarks.

---

## 4 · Limitations and honest weaknesses

### 4.1 Performance ceilings
- **Latency is 9–18ms p50, not sub-millisecond.** The README's
  "sub-millisecond retrieval" claim is off by an order of magnitude.
- **Insert throughput is 86/s with default config.** Slower than pure
  vector DB writes; the cost is in Γ-edge induction and audit logging.
  Bulk mode skips edge induction → faster but loses graph structure.
- **BGE encoding cost is fundamental** — ~5ms per encode dominates
  small-N retrieval latency. Can't go below this without smaller / faster
  embedder.
- **Persistent homology is slow at scale** — 6.1s for 500 nodes via
  gudhi. Likely minutes at 10k nodes (didn't run to completion).

### 4.2 Quality limits on standard benchmarks
- **Loses to Cosine RAG on LongMemEval** (0.833 vs 0.900). Recall doesn't
  beat baselines on factual long-conversation memory; it's competitive,
  not superior.
- **Loses to BM25 / Cosine on MAB Accurate_Retrieval** (0.729 vs 0.919 /
  0.862). Same root cause: fine-grained nodes vs whole-chunk retrieval.
- **Loses badly on MAB Conflict_Resolution** (0.531 vs 1.000 for both
  baselines). The benchmark name is misleading — it's actually multi-hop
  fact consolidation, not contradiction-override. Recall's design is
  structurally at a disadvantage.
- **Causal chain recall is 80%, not the claimed 100%.** Better than
  cosine's 40% but not perfect.
- **15% needle loss at 10k scale.** "Infinite memory never forgetting"
  isn't lossless — 3 of 20 planted needles missed at 10k nodes.

### 4.3 Quality limits on differentiator features
- **Bio-fingerprint catches 80%, not 100%** — the 2 patterns that leak
  through have only 2 attributes (below the 3-attribute threshold). Could
  drop threshold but at cost of false positives.
- **CRC bound = 0.316 in our run, not the marketed 0.175.** The 0.175
  number depends on n_calibration; with n=15 the bound is 0.316. Both
  are non-vacuous, but the specific number is benchmark-dependent.
- **`is_globally_consistent` flag is broken** — returns False even on
  clean baseline graphs. Only `frustration_score` is the reliable
  inconsistency signal.
- **Junk rate at default is 30%, not 27% target or 14% advertised.** The
  14% figure requires `use_llm_quality=True` (LLM-driven gate) which
  costs API calls per write. The 27% number was a previous run; current
  is 30%. Below 5% target is unmet either way.

### 4.4 Production-readiness limits
- **Solo-engineered, 154 unit tests.** No multi-developer review, no
  property-based or fuzz testing, no chaos testing.
- **~10k nodes is the largest verified scale.** Not tested at 1M+
  memories, multi-tenant production load, or under sustained throughput.
- **Python only.** No JS/Go/Rust SDKs. Mem0 has Python+JS+Go.
- **No cloud sync, no hosted offering, no team features.** Each instance
  is a standalone SQLite file.
- **No multi-region, no replication, no sharding.**
- **No SLA, no support contract, no production playbook.**
- **Docker image references unpublished `ghcr.io/yash194/recall:latest`.**
  The full `docker compose up` requires publishing first.

### 4.5 Coverage gaps
- **LoCoMo benchmark not run** — empty `benchmarks/locomo/` directory,
  no script written.
- **MemoryAgentBench Test_Time_Learning + Long_Range_Understanding not
  tested at scale** — only Accurate_Retrieval + Conflict_Resolution
  validated at n=5.
- **Mem0 head-to-head couldn't be validated** — Mem0 stored 0/500 in our
  run, suggesting a TokenRouter-compatibility issue with Mem0's LLM
  extraction. Direct multiplier comparison unverified.
- **Real-LLM bounded_answer at LongMemEval scale** not tested — bound
  was tested at n=15 calibration only.
- **Adversarial bound stress** — only 0/5 strict-mode refusals at small
  N. Need larger adversarial set to validate strict mode actually fires.

### 4.6 Marketing-vs-reality remaining gaps
| Marketing claim | Reality |
|---|---|
| "sub-millisecond retrieval" | 9–18ms |
| "5.9× cleaner than Mem0" | 3.3× at default |
| "100% causal chain" | 80% |
| "matches cosine on every type" | loses on 3 of 6 LongMemEval types |
| "Recall shines on Conflict_Resolution" | corrected; honestly says structurally disadvantaged |
| "infinite memory never forgetting" | 85% retention at 10k |

---

## 5 · Where it lacks (vs competitors, in concrete terms)

### 5.1 vs Mem0
- Mem0 has Python + JS + Go SDKs; Recall has Python only.
- Mem0 has hosted offering; Recall has none.
- Mem0 has thousands of production deployments; Recall has the test
  scripts in this repo.
- Mem0 has tooling for LLM-driven fact extraction at write time that
  produces shorter, more focused memories; Recall stores closer to the
  raw input.
- (Counter): Mem0 had 97.8% junk in the audited #4573 incident; Recall
  at 30% is genuinely 3.3× cleaner on synthetic replay.
- (Counter): Mem0 has no audit log per memory operation, no surgical
  forget with cascading, no hallucination bound, no inconsistency
  detection.

### 5.2 vs Letta (formerly MemGPT)
- Letta has tiered memory (working + archival) with explicit context
  management; Recall has flat memory + retrieval.
- Letta has a hosted dashboard for inspection; Recall has the CLI
  `recall me health`.
- Letta has multi-LLM-provider routing at the framework level; Recall
  delegates to OpenAI-compatible client.
- (Counter): Recall's audit log is more granular per operation.

### 5.3 vs Graphiti (Zep)
- Graphiti has bi-temporal graph (valid-time + transaction-time),
  optimized for entity-relationship knowledge graphs.
- Graphiti requires Neo4j; Recall is SQLite-local.
- (Counter): Graphiti's edges are entity-relationships, not
  reasoning-typed (`supports`, `contradicts`, etc.). Recall's edge types
  are richer for agent-memory tasks.

### 5.4 vs ChatGPT memory
- ChatGPT memory is integrated into the chat UI at scale; Recall is a
  developer-facing library.
- ChatGPT memory is opaque; Recall is fully auditable.
- ChatGPT memory has no API; Recall is API-first.

### 5.5 vs plain Cosine RAG with BGE-small
- Cosine RAG is simpler to implement and just as fast.
- Cosine RAG matches/beats Recall on factual lookup (LongMemEval).
- (Counter): Cosine RAG has no graph reasoning, no bounded generation,
  no audit, no forget, no inconsistency detection.

---

## 6 · What can be fixed (engineering work, no research needed)

### 6.1 Quick wins (≤ 1 day each)
- **`is_globally_consistent` flag in sheaf** — currently always False;
  fix by checking H¹ kernel size threshold
- **`recall me ask` defaulting** — currently auto-router; expose a
  `--multi-hop` flag for compositional questions
- **Higher per-question k for benchmarks** — multi-hop helps at k=8 but
  not k=5; expose `k_per_query` knob to benchmark scripts
- **Embedder cache** — cache embeddings of frequent queries
- **Top-K cosine batching** — already vectorized; could batch across
  queries for higher throughput
- **`ranked_lists` deduplication in multi_hop** — cosmetic

### 6.2 Medium-effort (1–5 days)
- **JS SDK** — port the public API (Memory, observe, recall,
  bounded_generate, forget) to TypeScript with same SQLite backend or
  remote HTTP client
- **Persisted in-memory index for topk_cosine** — already lazily loaded;
  serialize as `(N, embed_dim)` numpy file for cold-start speedup
- **Pgvector backend for self-hosted server** — schema parallel to
  SQLite, retrieval same shape; ~3-day port
- **MemoryAgentBench Test_Time_Learning + Long_Range_Understanding
  benchmarks at n=5** — already-written script just needs to be run
- **LoCoMo benchmark** — write the run.py from scratch, dataset is on HF
- **Mem0 head-to-head with proper Mem0 config** — debug why Mem0 stored
  0/500 (likely LLM provider config issue), retry
- **CLI `recall me ask --bound strict` proper output** — wire the
  HallucinationBlocked exception cleanly

### 6.3 Bigger-effort (1–2 weeks)
- **Hosted offering** — multi-tenant cloud service with auth, quotas,
  rate limiting, dashboards
- **Replication / sharding** — for multi-million-node deployments
- **Real-time streaming consolidation** — async background worker that
  runs BMRS / motifs incrementally as edges age
- **Pluggable splitter strategies** — by-paragraph, by-section,
  by-sentence-pair, configurable per use case
- **Rich recall me CLI** — TUI for browsing graph, drill-down on
  retrieval results
- **Replay-based fuzz tester** — feed real conversation logs (mem0
  #4573, hyperprompt logs) as integration tests

---

## 7 · What needs more research to fix

### 7.1 Document-QA performance (MAB Conflict_Resolution / Accurate_Retrieval)
- **Problem:** Recall scores 0.531 on Conflict_Resolution vs 1.000
  baselines. Multi-hop helps marginally (+25% at k=8) but doesn't close
  the gap. The benchmark format (whole 26K context as one chunk)
  fundamentally favors no-split systems.
- **What's needed:** research into how to combine fine-grained nodes
  (good for graph reasoning) with chunk-level retrieval (good for
  document QA). Open questions:
  - When should the splitter NOT split? Detect "fact-list" structure?
  - Should retrieval automatically expand to drawer-siblings? (We tried
    this; RRF dilution made it worse — needs smarter aggregation.)
  - HippoRAG-style PPR over entity-edge graph: does it close the gap on
    real multi-hop benchmarks like HotpotQA-Hard or MuSiQue?
- **References to read:** Edge et al. 2024 GraphRAG, Gutiérrez et al.
  2024 HippoRAG, Trivedi et al. 2022 MuSiQue, Yang et al. 2018 HotpotQA.

### 7.2 BMRS practical pruning calibration
- **Problem:** Math is now correct (Friston-Penny 2011), but at
  default `sigma_0_squared=1.0` and per-edge variance ≈ 0.0016 (cosine
  noise), virtually every cosine-weighted edge with |w| > 0.12 is kept.
  BMRS prunes nothing useful at default.
- **What's needed:** research into proper Bayesian prior over
  cosine-weighted edge magnitudes. Wright-Igel-Selvan use Laplace
  approximation around the MAP weight from training data; we have a
  single observation per edge. Open questions:
  - Is the right model: σ₀² = empirical std of |cos(s_i, s_j)| over the
    graph, σ_n² = empirical noise = variance of cos_s within a tenant?
  - Should BMRS fit `(σ₀², σ_n²)` jointly to the graph during
    consolidate via EM?
  - Does temporal recency factor in? (Older edges should have higher
    posterior variance → easier to prune?)
- **References:** Wright et al. 2024 BMRS, Friston & Penny 2011 BMR,
  Pereyra et al. 2014 (Bayesian model selection for sparse signals).

### 7.3 Multi-hop retrieval at small k
- **Problem:** At k=5 (the realistic benchmark setting), multi-hop
  candidates can't displace cosine top-5 in the score ranking. The
  score formula `max(query_sim, 0.6 · entity_sim)` is dominated by
  query_sim for the cosine seeds.
- **What's needed:** research into how to BOOST entity-only matches
  (low query_sim, high entity_sim) without degrading direct-cosine
  hits. RRF helped at the candidate-pool level but hurt at top-K
  selection. Open questions:
  - Should we explicitly RESERVE k/2 slots for entity-expansion
    candidates and k/2 for cosine seeds?
  - Iterative refinement: at each hop, re-score the WHOLE candidate
    pool with the latest entity vec, not the original query vec?
  - Use the LLM at retrieval time to decompose the question into
    sub-questions (Khattab 2024 Demonstrate-Search-Predict)?
- **References:** Gutiérrez et al. 2024 HippoRAG, Khattab & Zaharia
  2024 DSP, Cormack et al. 2009 RRF.

### 7.4 Hallucination bound tightness
- **Problem:** CRC bound = 0.316 at n=15 calibration. Bound holds but
  the SLACK (bound − empirical = 0.316) is large.
- **What's needed:** research into tighter bounds with smaller n, or
  data-dependent bounds that adapt to corpus structure. Open questions:
  - PAC-Bayes with informed prior: can we use the typed-edge graph
    structure to define a more concentrated prior?
  - Conditional CRC: bound conditional on the retrieval being
    well-supported (high cheeger lower bound on retrieved subgraph)?
  - Localized empirical Bernstein: tighter than Hoeffding when
    variance is small.
- **References:** Kang et al. 2024 C-RAG, Zhang et al. 2025 RAG-as-
  Noisy-ICL, Maurer & Pontil 2009 empirical Bernstein.

### 7.5 Bio-fingerprint at higher recall without false positives
- **Problem:** 80% recall, 100% precision is good but 20% leak through.
  The 2 leaking patterns have only 2 attribute types (below 3-threshold).
- **What's needed:** research into a smarter detection signal. Open
  questions:
  - Use embedding-distance to known fabricated patterns (a "bio-prototype"
    embedding) instead of regex pattern counts?
  - Train a small classifier on (text, conversation) → fabricated/genuine
    using a synthetic + audited corpus?
  - Use the LLM to anchor-check at write time (cost: API call per
    suspicious write)?
- **References:** Wang et al. 2025 Semantic Illusion (arXiv 2512.15068),
  hyperprompt audit logs (mem0 #4573).

### 7.6 True Ollivier-Ricci computation at scale
- **Problem:** Current Ollivier-Ricci uses a TV approximation that
  ignores graph topology. Forman-Ricci replaces it for protection, but
  Ollivier has theoretical advantages (continuous; finer-grained).
- **What's needed:** research into approximation algorithms for true
  W_1 over graph metric. Open questions:
  - Use POT (Python Optimal Transport) `ot.emd` with shortest-path
    distance matrix? Cost: O(d³) per edge; expensive at scale.
  - Use entropic regularized W_1 (Sinkhorn)? Faster but approximate.
  - Use approximation algorithms specific to graph optimal transport?
- **References:** Sandhu et al. 2015 Ollivier on graphs, Topping et al.
  2022 OR for GNNs, Ni et al. 2019 fast OR.

### 7.7 Real-LLM-driven edge typing
- **Problem:** Current edge classifier uses regex patterns +
  Γ-decomposition signs. Brittle on diverse text; misses paraphrased
  contradictions.
- **What's needed:** research into LLM-based edge classifier that doesn't
  cost an API call per write. Open questions:
  - Train a small distilled classifier (BERT-small) on LLM-labeled
    typed-edge data?
  - Use the LLM only during sleep-time consolidation, not write time?
  - Use NLI (Natural Language Inference) models to detect
    contradiction/entailment between fact pairs?
- **References:** Microsoft GraphRAG (LLM entity-relation extraction at
  index time), MNLI-pretrained models.

### 7.8 Truly bounded "infinite" memory
- **Problem:** 85% needle retention at 10k. As memory grows, some
  needles are lost. Latency stays bounded but quality decays slightly.
- **What's needed:** research into selective preservation /
  protected-set memory. Open questions:
  - Use curvature + recency + access-frequency to define a "protected
    core" that's never pruned?
  - Hierarchical storage: hot recent memories in active SQLite, cold
    pruned-but-archived in a separate store?
  - When BMRS would prune an edge, audit-log it but keep a "shadow"
    that can be restored on relevant query?
- **References:** Letta tiered memory, MemGPT context-management
  protocol, classical LRU/LFU vs forget-by-curvature.

---

## 8 · Roadmap / what to do next

### Near-term (1–2 weeks)
1. Fix `is_globally_consistent` flag (1 day).
2. Run MAB Test_Time_Learning + Long_Range_Understanding (1 day).
3. Run LoCoMo (write script + run; 2 days).
4. Debug Mem0 head-to-head config issue (1 day).
5. Publish Docker image to ghcr.io (0.5 day).
6. Real-LLM bounded_answer at LongMemEval scale (1 day, costs API).

### Medium-term (1–2 months)
1. JS SDK for npm distribution.
2. Pgvector backend for self-hosted production.
3. Address BMRS calibration via EM-fit prior (Section 7.2).
4. Improve multi-hop top-K selection (Section 7.3).
5. NLI-based edge classifier for contradicts/superseded (Section 7.7).

### Long-term / research (3+ months)
1. Document-QA / chunk-keeping mode that preserves graph reasoning
   (Section 7.1).
2. True Ollivier-Ricci with shortest-path metric (Section 7.6).
3. Tighter empirical Bernstein bounds for hallucination (Section 7.4).
4. Selective-preservation memory with curvature + recency + access
   frequency (Section 7.8).

---

## 9 · One-sentence honest pitch

*Recall is a working, locally-deployable, MCP-native, audit-trailed,
surgically-forgettable, hallucination-bounded, inconsistency-detecting
typed-edge memory substrate at ~10ms latency to 10k nodes — solid for
agentic memory and compliance use cases, demonstrably worse than BM25/
Cosine on factual lookup and document-QA, and one engineer away from
being a real Mem0/Letta competitor for the broader market.*

---

## 10 · Reproducibility

Every measurement in Sections 2 / 3 / 4 comes from one of:

| Source | Path |
|---|---|
| Unit tests | `tests/test_*.py` (154 tests, 1.4s) |
| Real benchmarks | `benchmarks/{hotpotqa_bge,longmemeval,bound_calibration,memoryagentbench,mem0_head_to_head,scale_stress,synthetic_gamma,junk_replay}/run.py` |
| Adversarial tests | inline Python in `VALIDATION_RESULTS.md` Section A/B |
| Stress simulation | `/tmp/recall_infinite_mem.py` (10k mems with edges) |
| CLI roundtrip | `recall --db /tmp/x me {add,ask,health,trace,consolidate}` |
| HTTP roundtrip | `RECALL_DB_DIR=/tmp/x uvicorn recall.server:app` |
| MCP roundtrip | `mcp__recall__{stats,search_memory,bounded_answer,...}` via Codex/Claude Code |

Every claim in this document has at least one test or benchmark backing
it. Where the result diverges from prior marketing, both the marketing
claim and the new measurement are stated honestly.
