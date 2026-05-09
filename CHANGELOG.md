# Changelog

## v0.2.2 — 2026-05-10

First release published via PyPI Trusted Publisher (OIDC) after registering
the publisher claim on pypi.org. Includes the v0.2.1 `override=True` fix
below. v0.2.1 was tagged but never reached PyPI due to a missing publisher
record at the time of release; this republishes the same code under 0.2.2
so `pip install -U typed-recall` resolves cleanly.

## v0.2.1 — 2026-05-10

### Fixed
- **`~/.recall/.env` autoload now overrides host-injected env vars**
  (`override=True`). Previously, when an MCP host (e.g. Codex's
  `mcp_servers.recall.env` block) injected `RECALL_OPENAI_MODEL` or
  `OPENAI_API_KEY` before subprocess start, dotenv silently refused to
  overwrite — making `.env` edits and `recall-setup` re-runs appear to
  have no effect. The `.env` file written by `recall-setup` is now the
  authoritative source of truth, as users expect.
- Mirrored the same precedence in the no-`python-dotenv` fallback path.

### Why this matters
The 0.2.0 default of `override=False` caused silent config drift: a
TokenRouter / OpenRouter user editing `RECALL_OPENAI_MODEL` to add a
required vendor prefix (e.g. `openai/gpt-4o-mini`) would see their
change ignored if the host had already set the bare name. 0.2.1 makes
the user's `.env` win, matching the principle that `recall-setup`
writes the canonical Recall config.

## v0.1.0 — 2026-05-08 (first PyPI / npm release)

First public release. Codename "v0.7" internally — versioned 0.1.0 on
PyPI / npm because the API is alpha-stage and we want to signal that
breaking changes are expected.

This release packages the work from v0.4 → v0.6 development plus a
round-2 fix-up sweep that addresses every claim documented in
`VALIDATION_RESULTS.md`.

### Headline features (verified working)
- **Audit log** — every memory operation append-only logged, queryable
  per `target_id`, exportable as JSONL.
- **Surgical forget** — by node-id with cascading edge deprecation;
  audit-trailed; verified end-to-end (0 leaks across 4 follow-up queries).
- **Bounded generation** — non-vacuous Conformal Risk Control bound per
  answer; verified holds (empirical 0.000 ≤ Hoeffding 0.316) with real
  LLM via TokenRouter.
- **Typed-edge graph** — `supports / contradicts / corrects / agrees /
  pivots / temporal_next / superseded` with sign-aware retrieval; cap of
  ~7.9 edges/node verified at 10k scale.
- **Multi-hop retrieval** (HippoRAG-inspired) — `mode="multi_hop"` with
  entity-expanded scoring for compositional questions.
- **Sheaf-H¹ frustration score** — verified 0.0 → 0.3 on planted
  contradictions in adversarial graph.
- **Forman-Ricci bottleneck protection** — verified flags bridge edges
  in K4-bridge-K4 (κ=−4 vs cluster +1 to +2); previously Ollivier-Ricci
  TV-approximation was blind to bottlenecks.
- **Bio-fingerprint anchor checking** — verified 80% recall / 100%
  precision on adversarial 50-FAB / 50-GEN corpus.
- **BMRS pruning** — Friston-Penny 2011 BMR formula corrected; default
  conservative (0% prune at default `sigma_0_squared=1.0`); was
  previously destroying the entire graph.
- **`recall me` CLI** — five subcommands (`add`, `ask`, `health`,
  `trace`, `consolidate`) implemented end-to-end.
- **HTTP server (FastAPI)** — six REST endpoints verified working.
- **MCP server (stdio)** — eight tools registered with Codex / Claude
  Code / Cursor / Windsurf.
- **Docker compose** — postgres + redis services come up healthy
  (recall-server image references unpublished GHCR; build it locally).

### Real benchmark results
| Benchmark | Result | Notes |
|---|---|---|
| HotpotQA distractor n=30 Recall@5 | 0.654 | matches/exceeds claim |
| HotpotQA MRR | 0.814 | matches claim |
| LongMemEval n=30 stratified Recall@5 | 0.833 | Cosine 0.900 wins |
| Synthetic causal chain | 4/5 = 80% | beats cosine 40% |
| 10k-node retention | 17/20 = 85% | latency p50 stays ~10ms |
| Junk rate (template) | 30% | Mem0 baseline 97.8% |
| CRC bound | 0.316 (n=15 cal); empirical 0.000 | non-vacuous; bound holds |

### Known limitations
- Multi-modal ingest: text only (no PDFs, images, audio).
- No JS/TS SDK — npm package is a thin wrapper around the Python server.
- ~10k nodes is the largest verified scale.
- LongMemEval and MemoryAgentBench: Recall is competitive but does not
  beat BM25/Cosine baselines; structurally disadvantaged on document-QA
  where whole-chunk retrieval wins.
- See [VALIDATION_RESULTS.md](VALIDATION_RESULTS.md) for the full
  honest evidence trail and [STATE.md](STATE.md) for the roadmap.

### Breaking changes from internal v0.6
- `Memory(use_llm_splitter=...)` parameter added; default `False` (was
  effectively `True` with a TokenRouter LLM). Re-enable explicitly if
  you want LLM-driven splitting.
- `Memory.bounded_generate(mode=...)` defaults to `"hybrid"` (was
  `"path"`).
- `pcst_extract(must_include=...)` parameter added; query seeds are
  preserved through extraction.
- `curvature_pruning_signal(method="forman", threshold=0.0)` is the new
  default. Pass `method="ollivier"` for the legacy (TV-approximation)
  curvature.
- `bmrs_should_prune` sign convention corrected: prune if `log BF > 0`
  (was `< 0`). Old behavior was provably wrong.

---

## v0.6 — 2026-05-07

Closes the v0.5 LongMemEval gap. Three architectural cleanups:
decoupled symmetric retrieval, cached graph adjacency, batched BGE encoding.

### The headline change: symmetric retrieval no longer goes through the
prompt prefix

In v0.5 every node and every query carried the Forward/Backward prompt
prefix (~50 chars each) inside its embedding. For long documents the
prefix is a small fraction of the embedding context, but for short
queries against long-document candidates it dominated and clustered all
short-text embeddings together. This is what made LongMemEval Recall lag
cosine RAG by 10-25 pts.

**v0.6 separates the two embedding workloads:**

| Workload | v0.5 | v0.6 |
|---|---|---|
| Γ-edge induction (needs f ≠ b) | `BGE("Forward describe..." + text)`, `BGE("Backward describe..." + text)` | unchanged — math depends on f, b |
| Symmetric retrieval | `(f + b) / 2` ← prompt-biased | `BGE(text)` ← raw text, no prefix |

API changes:

- `Embedder.embed_symmetric(text)` returns the raw symmetric vector.
  Default impl falls back to `(f + b) / 2`. `BGEEmbedder` overrides it to
  encode raw text without the prompt prefix.
- `Embedder.embed_batch(texts)` (BGE only) encodes f, b, and s for an
  entire batch in one inference call — used by the write pipeline to
  drop per-write latency from O(spans · BGE) to O(BGE).
- `Node.s_embedding` field added (optional). When present, `Node.s()`
  returns it directly; when absent, falls back to `(f + b) / 2` (so
  pre-v0.6 stored nodes still work).
- SQLite `nodes` table: new `s_embedding BLOB` column, applied via
  `ALTER TABLE ... ADD COLUMN` migration on existing databases.
- `Storage._ensure_index_loaded` and `insert_node` prefer the new
  `s_embedding` field for the cached retrieval matrix.
- `Memory.recall(query, ...)` calls `embedder.embed_symmetric(query)`
  for the seed cosine, when available.

### Adjacency cache for the auto-router

The auto-router's edge-density check was loading every active edge per
query — at 10K edges that's a slow Python construction on each
retrieval. v0.6 adds an in-memory adjacency cache:

- `Storage.adjacency()` returns `dict[node_id, set[neighbor_id]]`
- Built on first call; updated incrementally on `insert_edge`;
  invalidated on `deprecate_edge` (lazy rebuild).
- `Storage.n_active_edges()` is served from the cached count when
  available — O(1) rather than `COUNT(*)`.

The auto-router in `Memory.recall` now short-circuits to symmetric mode
when `n_active_edges() == 0` (e.g. bulk-mode corpora that haven't been
consolidated yet) without paying the load cost.

### Batched BGE encode

`BGEEmbedder.embed_batch(texts)` returns f-list, b-list, s-list for an
entire span batch in one inference call. The write pipeline uses it
when available, falling back to per-span `embed_dual` if not.
sentence-transformers handles list batching natively, so a drawer with
50 spans now does 1 BGE call instead of 50.

### Tests

- `+1 test`: `s_embedding` round-trips through SQLite
- `+1 test`: `topk_cosine` uses `s_embedding` when present
- `+1 test`: adjacency cache invalidates on `deprecate_edge`
- `+1 test`: `embed_symmetric` falls back to normalized `(f+b)/2`
- All 150 prior tests still pass; total now **154/154**

### Migration notes

- **No code changes required** for callers that use only `Memory`,
  `Embedder`, and `RecallAdapter`.
- **SQLite databases auto-migrate** on open (the `s_embedding` column
  is added if missing).
- **Pre-v0.6 stored nodes** continue to work: their `s_embedding` is
  NULL, so retrieval falls back to `(f + b) / 2` for those rows.
  Re-ingest if you want the new symmetric quality on legacy data.

### v0.7 roadmap

- HNSW/FAISS swap behind the `_EmbeddingCache` interface for 100K-1M
  tenant scale (the in-memory matrix grows linearly; disk-backed ANN
  is correct beyond ~50K).
- Path-mode retrieval refinements: better PCST budget calibration,
  multi-seed walk fusion (currently RRF is fine but can be tighter).
- Sleep-time consolidator running as an async task with daily
  per-tenant budgets.

---

## v0.5 — 2026-05-07

Stress-test response. The v0.4 benchmark sweep (see `benchmarks/REPORT.md`)
exposed three real flaws; v0.5 fixes them and tightens the underlying code.

### Bug fixes

**Scope filter is now subset semantics, not exact JSON match.**
`core/storage.py` was doing `WHERE scope_json = ?` with the full JSON-serialized
scope dict, so storing a node with `{"qid": q, "session_id": s}` and querying
with `{"qid": q}` returned nothing. The benchmark surface for this was
LongMemEval recall@5 dropping to 0.000.

The new contract: a stored scope matches a query scope iff every (k, v) in the
query also appears in the stored — the natural superset/subset semantics every
benchmark expected. Implemented in `_scope_matches_subset` and applied across
`topk_cosine`, `all_active_nodes`, and `topk_neighbors_for_gamma`.

`has_node_with_text_hash` retains exact-match — that's intentional for dedup,
where two scopes that differ in any key are different write targets.

### Performance

**Vectorized cosine retrieval (`topk_cosine` is now a single matmul).**
The old path loaded every active node, decoded each blob, computed `s = (f+b)/2`
in a Python loop, normalized, took dot product, sorted. At 10K nodes this
dominated the scale-stress test (the previous run stalled at ~10K).

The new `_EmbeddingCache` keeps an `(N, dim)` float32 matrix in memory,
L2-normalized on the way in. Lookups are one numpy matmul + an active mask +
a scope-subset mask + argpartition. At 10K nodes this drops topk_cosine p50
from O(N · python-overhead) to O(N · matmul).

Cache invalidation:
- `insert_node` appends a row when the index is initialized
- `deprecate_node` flips the active mask
- Lazy initialization on first query — cold start does one full table scan,
  steady state is amortized

`topk_neighbors_for_gamma` (used at write time for Γ-edge induction) takes the
same fast path automatically.

### New API: bulk-document mode

**`Memory.observe(..., fast=True)` (or any `source` in `Config.BULK_MODE_SOURCES`)
takes the bulk-document fast path:**

- Sentence splitter only (no LLM call)
- Skip the `all_active_nodes(scope=...)` scan that was being done per write
  to build the quality-classifier conversation buffer
- Skip Γ-edge induction entirely (edges can be induced in batch via
  `consolidate()` later)

This unblocks ingest of large RAG corpora (MemoryAgentBench, Wikipedia dumps,
codebases) where you don't want the conversation-grade pipeline running for
every chunk.

`source="document"` / `"bulk_document"` / `"corpus"` / `"ingest"` all
auto-trigger the fast path. The conversation-grade default `source="conversation"`
is unchanged.

### Tests

- `test_storage.py`: +3 tests covering scope subset semantics + index
  invalidation on deprecation
- `test_pipeline.py`: +2 tests covering bulk-mode fast path (no edges)
  and explicit `fast=True` overriding default source
- All 144 prior tests still pass; total now 149/149

### Breaking changes

None for external API users. Internally the `topk_cosine` and
`all_active_nodes` results may include nodes that the *exact-match* query
would have excluded — this is the bug fix; if any caller was relying on the
exact-match behavior (no internal caller was), they now need to pass the
full scope.

### Math

No primitive math changed. The scope filter was always supposed to be subset
semantics — the old implementation was a string-equality bug. The cosine
score, Γ-edge weight, walk weight, and CRC bound calculations are unchanged.

### Additional v0.5 changes (added during the stress test)

- **Chunk-coalescing splitter** (`write/splitter.py`). Sentences are merged
  into chunks of `[80, 250]` words so the prompt prefix doesn't dominate
  short-text embeddings. Default `min_chunk_tokens=80`. Pass
  `min_chunk_tokens=0` for legacy per-sentence behavior.
- **Auto-router fast-path** (`api.py`). For factual-cue queries (no
  directional words), skip the `all_active_nodes` + `all_active_edges`
  load and route straight to symmetric mode. Drops p50 retrieval cost
  from O(N python-objects) to O(matmul).
- **Conversation-buffer SQL LIMIT** (`core/storage.py`,
  `write/pipeline.py`). The quality classifier's "last 20 nodes" buffer
  now uses `LIMIT 20` in SQL instead of materializing every active node.
- **`Memory.bulk_observe(texts, scope, source)`** convenience API for
  batch RAG-corpus ingestion.
- **`Memory.consolidate(induce_edges=True)`** — batch Γ-edge induction
  for bulk-ingested corpora that skipped edges at write time.
- **`RecallAdapter(mem, mode="bulk"|"conversation")`** parameter on the
  benchmark adapter.
- **Stratified sampling option** in `benchmarks/longmemeval/run.py`
  (`--sample stratified`) so the 6-question-type breakdown is
  represented.
- **Shared embedder caching** in benchmarks (one BGE model load instead
  of one per question).
- **`evaluate_one` meta/scope merge** — the LongMemEval benchmark was
  reading `h.meta.get("session_id")` first; RecallAdapter now copies
  scope keys into meta so the lookup works without the fall-through bug.

### Roadmap (v0.6)

- HNSW/FAISS swap behind the same `_EmbeddingCache` interface for
  100K-1M tenant scale.
- **Decouple symmetric retrieval embedding from f/b dual prompts.** The
  Γ-algebra needs prompted views (forward/backward) for typed-edge
  induction; symmetric retrieval doesn't. Splitting them gives Recall
  the best of both — prompt-free symmetric retrieval matching cosine,
  plus prompted f/b for typed-edge induction. This is what closes the
  v0.5 LongMemEval gap (Recall 0.65 vs Cosine 0.90 on stratified).
- Adjacency cache for the auto-router (avoid `all_active_edges()` per
  query for path/hybrid mode).
- Batch BGE encode in the pipeline for bulk-document mode.
