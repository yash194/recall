# ARCHITECTURE — Complete System Design of Recall

> Every interface, every data structure, every algorithm. Implementation reference.
> Pairs with `MATH.md` (the math) and the `src/recall/` codebase (the realization).

---

## 0. The five units of memory

| Unit | What it is | Lives in |
|---|---|---|
| **Drawer** | Verbatim immutable text fragment with metadata | `drawers` table |
| **Node** | Versioned thought, points at one or more drawer ranges, has dual embeddings `(f, b)` | `nodes` table |
| **Edge** | Typed, asymmetric, signed-weighted directed edge between two nodes | `edges` table |
| **Motif** | Recurring subgraph pattern stored as a meta-node | `motifs` table |
| **AuditEntry** | Immutable record of every state change | `audit_log` table |

Every operation in Recall is a function over one or more of these.

---

## 1. The five public API calls

```python
class Memory:
    def __init__(self, tenant: str, storage: str = "sqlite://./recall.db",
                 embedder: Embedder | None = None, llm: LLMClient | None = None): ...

    # --- WRITE
    def observe(self, user_msg: str, agent_msg: str, scope: dict | None = None,
                source: str = "conversation") -> WriteResult: ...

    # --- READ (path-only)
    def recall(self, query: str, scope: dict | None = None,
               mode: Literal["path", "symmetric", "hybrid"] = "path",
               k: int = 10, depth: int = 4) -> RetrievalResult: ...

    # --- READ + GENERATE (with hallucination bound)
    def bounded_generate(self, query: str, scope: dict | None = None,
                         bound: Literal["strict", "soft", "off"] = "strict",
                         k: int = 10, depth: int = 4) -> GenerationResult: ...

    # --- AUDIT
    def trace(self, generation: GenerationResult) -> Trace: ...

    # --- FORGET
    def forget(self, node_id: str, reason: str, actor: str = "system") -> ForgetResult: ...
```

Five calls. Public surface. Stable across versions.

---

## 2. Data schema

### 2.1 `drawers`

```sql
CREATE TABLE drawers (
    id TEXT PRIMARY KEY,               -- sha256 of text + scope + source
    text TEXT NOT NULL,                -- verbatim
    source TEXT NOT NULL,              -- 'conversation' | 'document' | 'recall_artifact'
    tenant TEXT NOT NULL,
    scope_json TEXT NOT NULL,          -- JSON of scope dict
    created_at TIMESTAMP NOT NULL,
    valid_from TIMESTAMP NOT NULL,     -- bi-temporal: when fact became true
    valid_to TIMESTAMP,                -- when fact stopped being true (null = still true)
    transaction_time TIMESTAMP NOT NULL  -- when written to DB
);
CREATE INDEX drawers_tenant_scope ON drawers (tenant, scope_json);
CREATE INDEX drawers_source ON drawers (source);
```

### 2.2 `nodes`

```sql
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,               -- uuid
    tenant TEXT NOT NULL,
    text TEXT NOT NULL,                -- distilled summary; 1–3 sentences
    drawer_ids TEXT NOT NULL,          -- JSON list of drawer ids this node distills
    f_embedding BLOB,                  -- forward (prompted), used for Γ-edges
    b_embedding BLOB,                  -- backward (prompted), used for Γ-edges
    s_embedding BLOB,                  -- v0.6: raw symmetric, used for retrieval
    embed_dim INTEGER,                 -- vector dimension for blob decode
    role TEXT,                          -- 'fact' | 'attempt' | 'decision' | 'pivot' | 'outcome' | 'correction'
    quality_score REAL NOT NULL,       -- 0..1
    quality_status TEXT NOT NULL,      -- 'pending' | 'promoted' | 'rejected'
    scope_json TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    parent_node_id TEXT,                -- forward link for versioning
    deprecated_at TIMESTAMP,
    deprecated_reason TEXT,
    created_at TIMESTAMP NOT NULL,
    transaction_time TIMESTAMP NOT NULL
);
CREATE INDEX nodes_tenant_scope ON nodes (tenant, scope_json);
CREATE INDEX nodes_quality ON nodes (quality_status);
CREATE INDEX nodes_deprecated ON nodes (deprecated_at);
CREATE INDEX nodes_text_hash ON nodes (text_hash, scope_json);
```

The dual embeddings `(f, b)` are the prompted views used by Γ-edge
induction; both are stored verbatim. The symmetric retrieval embedding
`s_embedding` is computed from raw text (no prompt prefix) and used for
the seed-cosine step (v0.6 — see MATH §2.6).

For nodes ingested before v0.6, `s_embedding` is NULL and retrieval
falls back to the legacy `(f+b)/2` average. New nodes always store
`s_embedding`. Schema migration is automatic on database open
(`ALTER TABLE ... ADD COLUMN`).

### 2.3 `edges`

```sql
CREATE TABLE edges (
    id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    src_node_id TEXT NOT NULL,
    dst_node_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,           -- 'supports' | 'contradicts' | 'corrects' | 'agrees' | 'pivots' | 'temporal_next' | 'superseded'
    weight REAL NOT NULL,              -- signed; contradicts → negative
    gamma_score REAL NOT NULL,         -- raw Γ (i → j)
    gamma_anti REAL,                   -- antisymmetric component for diagnostics
    s_squared REAL NOT NULL,           -- BMRS variance estimate
    bmrs_log_ratio REAL,               -- evidence ratio cache
    deprecated_at TIMESTAMP,
    deprecated_reason TEXT,
    created_at TIMESTAMP NOT NULL,
    last_validated_at TIMESTAMP,
    FOREIGN KEY (src_node_id) REFERENCES nodes(id),
    FOREIGN KEY (dst_node_id) REFERENCES nodes(id)
);
CREATE INDEX edges_src ON edges (src_node_id);
CREATE INDEX edges_dst ON edges (dst_node_id);
CREATE INDEX edges_type ON edges (edge_type);
CREATE INDEX edges_tenant ON edges (tenant);
```

Asymmetric edges stored as a single directed row. `Γ(j → i)` is a separate row.

### 2.4 `motifs`

```sql
CREATE TABLE motifs (
    id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    pattern_json TEXT NOT NULL,        -- JSON of typed-edge subgraph template
    instances_json TEXT NOT NULL,      -- JSON list of (node_id_subset, mapping)
    occurrence_count INTEGER NOT NULL,
    parameter_summary TEXT,             -- LLM-distilled summary of what varies across instances
    created_at TIMESTAMP NOT NULL
);
```

### 2.5 `audit_log` (append-only)

```sql
CREATE TABLE audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    operation TEXT NOT NULL,           -- 'WRITE' | 'PROMOTE' | 'REJECT' | 'GENERATE' | 'FORGET' | 'CONSOLIDATE' | 'PRUNE' | 'MOTIF'
    actor TEXT NOT NULL,                -- 'system' | 'user:<uid>' | 'agent:<aid>' | 'consolidator'
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,         -- full state diff
    reason TEXT
);
CREATE INDEX audit_log_target ON audit_log (target_type, target_id);
CREATE INDEX audit_log_tenant_time ON audit_log (tenant, timestamp);
```

Append-only. Never truncated. Exportable as JSONL for procurement audits.

---

## 3. Module map

```
src/recall/
├── __init__.py              # re-exports public API
├── api.py                   # Memory class, the public surface
├── types.py                 # Drawer, Node, Edge, EdgeType, etc.
├── config.py                # tenant config, thresholds
├── embeddings.py            # Embedder protocol + reference impls (BGE-M3 hook, hash-stub for tests)
│
├── core/
│   ├── __init__.py
│   └── storage.py           # SQLite-backed Storage class with the schema above
│
├── geometry/
│   ├── __init__.py
│   └── gamma.py             # Γ score, split, identifiability hooks
│
├── write/
│   ├── __init__.py
│   ├── pipeline.py          # observe() pipeline: dedup, provenance, quality, split, edge induction
│   ├── splitter.py          # node_split: LLM with sentence-level fallback
│   └── quality.py           # quality_classifier
│
├── retrieval/
│   ├── __init__.py
│   ├── walk.py              # Γ-walk
│   ├── pcst.py              # Steiner extraction with signed weights
│   ├── intent.py            # query intent classifier
│   └── linearize.py         # subgraph → context string
│
├── bound/
│   ├── __init__.py
│   ├── pac_bayes.py         # bound computation (Theorem 3.2)
│   └── support.py           # structurally_supported(claim, R)
│
├── audit/
│   ├── __init__.py
│   └── log.py               # AuditLog: append, query, export
│
├── consolidate/
│   ├── __init__.py
│   ├── bmrs.py              # BMRS edge pruning (§4 of MATH.md)
│   ├── mean_field.py        # mean-field refinement
│   ├── motif.py             # motif detection / replacement
│   └── scheduler.py         # priority queue, budget allocation
│
├── llm.py                   # LLMClient protocol (for generation + classification)
└── cli.py                   # `recall ingest`, `recall inspect`, `recall forget`, `recall audit`
```

### Dependency graph (no circular imports)

```
api.py ──→ core.storage, embeddings, llm,
           write.pipeline, retrieval.{walk,pcst,intent,linearize},
           bound.{pac_bayes,support}, audit.log

write.pipeline ──→ core.storage, embeddings, geometry.gamma,
                   write.{splitter,quality}, audit.log

retrieval.walk ──→ core.storage, geometry.gamma
retrieval.pcst ──→ core.storage
retrieval.intent ──→ llm

bound.pac_bayes ──→ (no internal deps)
bound.support  ──→ core.storage, llm

audit.log ──→ core.storage

consolidate.* ──→ core.storage, geometry.gamma, audit.log

geometry.gamma ──→ (no internal deps)
core.storage   ──→ types
embeddings     ──→ types
llm            ──→ (no internal deps)
types          ──→ (no internal deps)
```

`types` and `geometry.gamma` are leaves. `api` is the root. No cycles.

---

## 4. The protocol classes (interfaces)

### 4.1 `Embedder`

```python
class Embedder(Protocol):
    """
    Pluggable dual-view embedding. Returns (f, b) for a given text.
    Reference impls:
      - HashEmbedder    (deterministic, for tests; uses sha-derived random vectors)
      - BGEEmbedder     (BAAI/bge-m3 via sentence-transformers, prompt-prefixed)
      - VoyageEmbedder  (voyage-4 with asymmetric prompts; production)
    """

    @property
    def dim(self) -> int: ...

    def embed_dual(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Returns (f, b), each L2-normalized, dtype float32."""
```

The implementation prefixes the text with a forward / backward instruction:

```python
FORWARD_PROMPT  = "Forward describe — what comes next or follows from: "
BACKWARD_PROMPT = "Backward describe — what came before or causes: "

# Both embedded by the SAME underlying model. The asymmetry is purely from the prompt.
```

Dimension `d` is fixed per embedder (BGE-M3: 1024; HashEmbedder: 256 for tests).

### 4.2 `LLMClient`

```python
class LLMClient(Protocol):
    def complete(self, prompt: str, max_tokens: int = 256) -> str: ...

    def classify(self, prompt: str, options: list[str]) -> str:
        """Returns one of `options` based on the LLM's choice."""

    def split_into_thoughts(self, text: str) -> list[ThoughtSpan]:
        """Returns 0..N (text_span, role) pairs for a given utterance."""
```

Reference impls: `OpenAIClient`, `AnthropicClient`, `LocalLlamaClient`, `MockLLMClient` (for tests).

### 4.3 `Storage`

```python
class Storage(Protocol):
    """SQLite-backed in v1; KuzuDB-backed in v2."""

    def insert_drawer(self, d: Drawer) -> None: ...
    def insert_node(self, n: Node) -> None: ...
    def insert_edge(self, e: Edge) -> None: ...

    def get_node(self, node_id: str) -> Node | None: ...
    def get_edges_from(self, node_id: str, edge_type: str | None = None) -> list[Edge]: ...
    def get_edges_to(self, node_id: str, edge_type: str | None = None) -> list[Edge]: ...

    def topk_cosine(self, query_emb: np.ndarray, scope: dict, k: int) -> list[Node]: ...
    def topk_neighbors_for_gamma(self, node: Node, scope: dict, k: int) -> list[Node]: ...

    def deprecate_node(self, node_id: str, reason: str) -> None: ...
    def deprecate_edge(self, edge_id: str, reason: str) -> None: ...

    def append_audit(self, entry: AuditEntry) -> None: ...
    def query_audit(self, target_id: str | None = None, since: datetime | None = None) -> list[AuditEntry]: ...
```

---

## 5. Algorithms in pseudocode

### 5.1 `observe()` — Tier-1 write

```
function observe(user_msg, agent_msg, scope, source="conversation"):
    raw = user_msg + " | AGENT: " + agent_msg
    drawer = Drawer(text=raw, source=source, scope=scope, tenant=self.tenant)

    # 1. Hash dedup
    if storage.has_drawer_with_id(drawer.id): return SKIP_DUPLICATE

    # 2. Provenance firewall
    if source == "recall_artifact":
        return REJECT_RECALL_LOOP

    # 3. Persist drawer
    storage.insert_drawer(drawer)
    audit_log.append("WRITE_DRAWER", drawer.id)

    # 4. Node split
    spans = llm.split_into_thoughts(raw)
    if not spans: spans = sentence_fallback(raw)

    written_nodes = []
    for span in spans:
        # 5. Compute dual embeddings
        f, b = embedder.embed_dual(span.text)
        node = Node(text=span.text, drawer_ids=[drawer.id], role=span.role,
                    f=f, b=b, scope=scope, tenant=self.tenant)

        # 6. Quality gate
        node.quality_score = quality_classifier(node)
        if node.quality_score < THRESH_QUALITY:
            node.quality_status = "rejected"
            audit_log.append("REJECT", node.id, reason="low_quality")
            storage.insert_node(node)  # persist for audit even if rejected
            continue

        # 7. Hash-dedup at node level (Mem0 trick)
        if storage.has_node_with_text_hash(node.text_hash, scope):
            audit_log.append("SKIP_NODE_DUPLICATE", node.id)
            continue

        # 8. Top-k neighbor candidates for Γ-edge induction
        neighbors = storage.topk_neighbors_for_gamma(node, scope, k=10)

        # 9. Score Γ-edges (cheap fast path — no LLM)
        edges = []
        for n_neighbor in neighbors:
            gamma_ij = gamma_score(node.f, node.b, n_neighbor.f, n_neighbor.b)
            gamma_ji = gamma_score(n_neighbor.f, n_neighbor.b, node.f, node.b)
            for (src, dst, g) in [(node, n_neighbor, gamma_ij), (n_neighbor, node, gamma_ji)]:
                if abs(g) >= THRESH_GAMMA:
                    edges.append(Edge(src=src, dst=dst, gamma=g, type="pending"))

        # 10. Persist
        node.quality_status = "promoted"
        storage.insert_node(node)
        for e in edges: storage.insert_edge(e)
        audit_log.append("PROMOTE", node.id, edges=len(edges))
        written_nodes.append(node)

    return WriteResult(nodes=written_nodes, drawer_id=drawer.id)
```

### 5.2 `recall()` — read

```
function recall(query, scope, mode="path", k=10, depth=4):
    if mode == "symmetric":
        f_q, b_q = embedder.embed_dual(query)
        s_q = (f_q + b_q) / 2
        return storage.topk_cosine(s_q, scope, k)

    intent = classify_intent(query)  # "directional" | "symmetric" | "hybrid"

    # Symmetric seed (cheap)
    f_q, b_q = embedder.embed_dual(query)
    s_q = (f_q + b_q) / 2
    seeds = storage.topk_cosine(s_q, scope, k=k)

    if mode == "path" or intent == "directional" or intent == "hybrid":
        # Γ-walk
        paths = []
        for seed in seeds:
            paths.extend(gamma_walk(seed, depth=depth, weight_threshold=THRESH_WALK))

        # Subgraph extraction with signed weights (PCST approximation)
        subgraph = pcst_extract(paths, budget=BUDGET_SUBGRAPH)
        return subgraph

    return seeds  # mode='symmetric' early return; we shouldn't reach here, but defensive
```

### 5.3 `gamma_walk()`

```
function gamma_walk(start_node, depth, weight_threshold):
    """Beam-search forward over Γ-weighted edges."""
    paths = [Path(nodes=[start_node])]
    for d in range(depth):
        next_paths = []
        for p in paths:
            tail = p.nodes[-1]
            outgoing = storage.get_edges_from(tail.id)
            for edge in outgoing:
                if edge.deprecated_at: continue
                if edge.weight < weight_threshold: continue
                # negative-weight `contradicts` edges allowed but penalized
                next_node = storage.get_node(edge.dst_node_id)
                if next_node in p.nodes: continue  # acyclic
                next_paths.append(p.extend(next_node, edge))
        # beam: keep top-K paths by cumulative weight
        paths = sorted(next_paths, key=lambda p: p.cum_weight, reverse=True)[:BEAM_WIDTH]
    return paths
```

### 5.4 `pcst_extract()` — Steiner with signed weights

```
function pcst_extract(paths, budget):
    """
    Prize-Collecting Steiner Tree on the union of paths.
    Positive-weight edges are 'rewards'; negative-weight edges are 'costs'.
    Approximation via greedy + beam (1.79-approx in pcst-fast; we ship a simpler greedy).
    """
    nodes = union(p.nodes for p in paths)
    edges = union(p.edges for p in paths)
    prizes = {n: max(0, sum(e.weight for e in incident(n, edges))) for n in nodes}
    costs  = {e: max(0, -e.weight) for e in edges}  # contradicts → cost > 0

    # Greedy: start with highest-prize node, grow until budget exhausted
    selected = {argmax(prizes)}
    total_cost = 0.0
    while total_cost < budget:
        best_addition = argmax_over(nodes - selected,
                                    lambda n: prizes[n] - cost_to_connect(n, selected, edges))
        if prizes[best_addition] - cost_to_connect(best_addition, selected, edges) <= 0: break
        selected.add(best_addition)
        total_cost += cost_to_connect(best_addition, selected, edges)
    return induced_subgraph(selected, edges)
```

### 5.5 `bounded_generate()`

```
function bounded_generate(query, scope, bound="strict", k=10, depth=4):
    R = recall(query, scope, mode="path", k=k, depth=depth)
    context = linearize(R)
    raw = llm.complete(prompt=BOUNDED_PROMPT.format(query=query, context=context))

    claims = extract_claims(raw)  # LLM-driven sentence-level extraction
    flagged = []
    for claim in claims:
        if not structurally_supported(claim, R):
            flagged.append(claim)

    # Compute the empirical PAC-Bayes bound for telemetry
    bound_value = compute_bound_estimate(R, scope, query)

    if bound == "strict" and flagged:
        raise HallucinationBlocked(claims=flagged, retrieved=R, bound=bound_value)

    audit_log.append("GENERATE", query=query, retrieved=R, raw=raw, flagged=flagged, bound=bound_value)
    return GenerationResult(text=raw, retrieved=R, flagged=flagged, bound=bound_value)
```

### 5.6 `structurally_supported()`

```
function structurally_supported(claim, subgraph_R):
    """
    A claim is supported iff there exists a directed walk in R whose drawer text
    entails the claim.
    """
    for path in all_paths_in(subgraph_R):
        drawer_text = concat(node.drawer_text() for node in path.nodes)
        if entails(drawer_text, claim):  # NLI-style check: small LLM or trained model
            return True
    return False
```

### 5.7 `forget()`

```
function forget(node_id, reason, actor):
    n = storage.get_node(node_id)
    if not n: return ForgetResult(error="not_found")
    storage.deprecate_node(node_id, reason)
    for edge in storage.get_edges_from(node_id) + storage.get_edges_to(node_id):
        storage.deprecate_edge(edge.id, reason="cascading_node_forget")
    audit_log.append("FORGET", node_id, reason=reason, actor=actor)
    return ForgetResult(deprecated_node_id=node_id,
                       deprecated_edges=[edge.id for edge in ...])
```

### 5.8 `consolidate()` — Tier-3 background

```
function consolidate(scope, budget=100):
    """Run periodically (e.g. nightly). Bounded by `budget` regions per pass."""
    regions = priority_queue_of_dirty_regions(scope)
    for region in regions[:budget]:
        # 1. BMRS pruning (§4 of MATH.md)
        for edge in region.edges:
            log_ratio = bmrs_log_evidence_ratio(edge)
            edge.bmrs_log_ratio = log_ratio
            if log_ratio < 0:
                storage.deprecate_edge(edge.id, reason="bmrs_pruned")
                audit_log.append("PRUNE", edge.id, reason="bmrs", ratio=log_ratio)

        # 2. Mean-field refinement (§4.6)
        mean_field_iterate(region, T=5)

        # 3. Motif extraction (§4.5)
        motifs = find_recurring_subgraphs(region, min_occurrences=3)
        for h in motifs:
            create_motif_node(h)
            audit_log.append("MOTIF", motif_id=h.id, instances=len(h.instances))

        # 4. Bound bookkeeping
        update_bound_quantities(region)
```

### 5.9 `trace()`

```
function trace(generation_result):
    """Return the full provenance of a generation."""
    R = generation_result.retrieved
    return Trace(
        nodes = [serialize(n) for n in R.nodes],
        edges = [serialize(e) for e in R.edges],
        drawers = [serialize(storage.get_drawer(did))
                   for n in R.nodes for did in n.drawer_ids],
        bound_value = generation_result.bound,
        generated_text = generation_result.text,
        flagged_claims = generation_result.flagged,
        audit_entries = audit_log.entries_for_targets(R.node_ids() + R.edge_ids())
    )
```

---

## 6. Configuration / thresholds

```python
# src/recall/config.py
class Config:
    # Quality gating
    THRESH_QUALITY: float = 0.4          # below → reject

    # Γ-edge induction
    THRESH_GAMMA: float = 0.05           # min |Γ| for edge creation
    THRESH_WALK: float = 0.0             # min weight for walk traversal
    BEAM_WIDTH: int = 8                   # Γ-walk beam

    # Quality classifier
    QUALITY_NEGATIVE_TEMPLATES: list[str] = [
        "I don't have access to that information",
        "Let me clarify",
        "Sure, I can help with that",
        # ... patterns observable in mem0 #4573 audit ...
    ]

    # PAC-Bayes bound
    DELTA: float = 0.05                   # confidence parameter

    # BMRS
    EDGE_PRIOR_VAR: float = 1.0
    EDGE_VAR_FLOOR: float = 1e-3

    # Subgraph extraction
    BUDGET_SUBGRAPH: float = 10.0

    # Embedder
    EMBED_DIM: int = 1024                 # BGE-M3 default

    # Edge type vocabulary
    EDGE_TYPES: list[str] = [
        "supports", "contradicts", "corrects", "agrees",
        "pivots", "temporal_next", "superseded"
    ]
```

All thresholds are tenant-overridable via `Memory(tenant=..., config=...)`.

---

## 7. Concurrency & deployment shape

### 7.1 Embedded mode (v1 default)

- Single Python process.
- SQLite with WAL mode for concurrent reads.
- One global asyncio event loop; consolidation runs in a background task.
- Suitable for: single agent, single user, single machine.

### 7.2 Server mode (v2)

- HTTP + gRPC server on top of embedded core.
- WAL replication / SQLite-backed; switch to KuzuDB at higher scale.
- Stateless API workers; sticky tenant → consolidator workers.
- Suitable for: multi-tenant, multi-agent, distributed.

### 7.3 Sleep-time consolidator

- Background process; same Python interpreter in v1 (asyncio task).
- Triggered by:
  1. Time (every N minutes).
  2. Volume (every M new nodes).
  3. Explicit call (`mem.consolidate(scope=...)`).
- Bounded by daily budget per tenant.

---

## 8. Testing strategy

### 8.1 Unit tests (`tests/`)

| File | What it tests |
|---|---|
| `test_gamma.py` | Γ asymmetry theorem, identity expansions, edge cases (zero vectors, duplicate texts) |
| `test_storage.py` | Schema migrations, CRUD on all five tables, audit-log appending |
| `test_pipeline.py` | observe() with mock embedder/LLM: dedup, provenance firewall, quality gate, edge induction |
| `test_walk.py` | gamma_walk on synthetic graphs: termination, beam correctness, deprecated edges excluded |
| `test_pcst.py` | PCST extraction: greedy correctness on toy signed-weight graphs |
| `test_bound.py` | PAC-Bayes bound formulae: monotonicity in n, vacuity threshold |
| `test_bmrs.py` | BMRS evidence-ratio sign for known cases |
| `test_audit.py` | Audit log immutability, query, export |
| `test_api.py` | End-to-end Memory class: 3-line quickstart works |
| `test_forget.py` | Forget cascades to edges; deprecated nodes excluded from retrieval |

### 8.2 Integration tests

| File | What it tests |
|---|---|
| `test_quickstart_smoke.py` | The `examples/quickstart.py` runs end-to-end with HashEmbedder + MockLLM |
| `test_synthetic_gamma.py` | On synthetic causal data, Γ-walk recovers the planted DAG |
| `test_junk_replay.py` | The mem0-#4573 replay benchmark scaffold runs (with synthetic dataset) |

### 8.3 Property tests (hypothesis)

- Γ asymmetry: for random `(f, b)` pairs, `Γ(i,j) − Γ(j,i)` correctly computes `2·(c_i·s_j − c_j·s_i)`.
- Edge sign consistency: `contradicts` edges always have `weight < 0`.
- Audit log monotone: every operation appends, never deletes.

---

## 9. Performance budgets

| Operation | Target latency | Notes |
|---|---|---|
| `observe()` Tier-1 | < 100 ms | with HashEmbedder; BGE-M3 raises this to 200ms |
| `recall(mode='path')` | < 200 ms at 10K nodes | beam-walk, pre-computed FAISS index |
| `bounded_generate()` | < 2s | dominated by LLM call |
| `forget()` | < 50 ms | single deprecation transaction |
| `trace()` | < 100 ms | sub-cursor over audit log |
| Sleep-time per region | < 5s | bounded by mean-field iterations |

Every operation has a timer; metrics are recorded for telemetry.

---

## 10. Versioning & schema migration

- Schema versioned via `PRAGMA user_version` in SQLite.
- Migrations live in `src/recall/core/migrations/` as numbered SQL scripts.
- Migration runs on `Memory(tenant=...)` instantiation if `user_version` is behind.
- Backward-compat: read old schemas, write new schemas, never break old read paths within a major version.

---

## 11. Out-of-scope for v1

Explicit non-goals:

- KuzuDB integration (use SQLite). v2.
- Multi-modal nodes (image/audio drawers). v3.
- Federated graph sync. Never.
- Real-time multi-agent write contention. v2 (Postgres-style locks).
- Production-grade BGE-M3 distillation. v2.
- Full PCST 1.79-approximation via pcst-fast. v1 ships greedy; v2 uses pcst-fast.

---

## 12. v0.5 / v0.6 architectural additions

This section documents architectural changes that landed after the
v0.4 stress-test sweep.

### 12.1 In-memory embedding cache (`_EmbeddingCache`)

`src/recall/core/storage.py:_EmbeddingCache` keeps an `(N, dim)`
float32 matrix of all promoted-active node embeddings, L2-normalized
on insert. Top-k retrieval is one numpy matmul + active mask + scope
filter + argpartition.

**Capacity management.** The matrix is pre-allocated with capacity
`max(256, 2·initial_size)`; when the live size exceeds capacity, the
matrix is reallocated at `2× capacity` (doubling). Per-insert cost is
amortized O(1) — fixes the original O(N²) `np.vstack` pitfall.

**Lifecycle:**
- `bulk_load(rows)` — invoked on first query; loads all promoted-active
  rows from SQLite, normalizes, populates the matrix.
- `add(node_id, s_vec, scope)` — appended to the live prefix. If the
  cache is not yet initialized, this is a no-op (the next query will
  bulk-load and pick it up).
- `deprecate(node_id)` — flips a bool in the active mask; the row
  stays allocated.
- `topk(query, scope_filter, k)` — masks deprecated, filters by scope
  subset semantics, runs `live_matrix @ query`, argpartition top-k.

**Performance.** At 10K nodes / 384-dim:
- 10K appends total: 35 ms (3.5 µs each)
- 100 top-k@5 queries: 13 ms (0.13 ms each)

The cache replaces the previous brute-force "load every row from
SQLite, decode each blob, compute `s` in Python loop, sort" path that
stalled the v0.4 scale-stress test at 10K nodes.

### 12.2 Scope subset semantics

v0.5 fixed a correctness bug in the SQL scope filter. The previous
implementation matched scope by exact JSON-string equality:
```sql
SELECT ... WHERE scope_json = ?
```
This rejected any node whose stored scope had additional keys beyond
the query — e.g. a query `{"qid": q}` would not match a node stored
with `{"qid": q, "session_id": s}`.

**v0.5 fix.** `_scope_matches_subset(query, stored)` returns True iff
every `(k, v) in query` is also in `stored`. Applied in `topk_cosine`,
`all_active_nodes`, and `topk_neighbors_for_gamma`. Empty query
matches every stored scope.

`has_node_with_text_hash` retains exact-match (intentional for dedup,
where two scopes that differ in any key are different write targets).

### 12.3 Bulk-document mode

`Memory.observe(..., fast=True)` (or any `source` in
`Config.BULK_MODE_SOURCES`) takes a fast path:

1. Sentence splitter only (no LLM)
2. Skip the per-write `all_active_nodes(scope)` scan that built the
   quality classifier's conversation buffer
3. Skip Γ-edge induction (edges can be induced offline via
   `consolidate(induce_edges=True)`)

The auto-router (§4) automatically falls back to symmetric mode when
`Storage.n_active_edges() == 0`, so bulk-mode corpora retrieve via
plain cosine until a consolidation pass adds edges.

`Memory.bulk_observe(texts, scope)` is a convenience wrapper: one
`observe()` call per text with `fast=True` and `source="bulk_document"`.

### 12.4 Adjacency cache

The auto-router needs the active edge graph for the path/hybrid
decision. v0.5 loaded all active edges per query and built the
adjacency dict on the fly — O(E) per retrieval. v0.6 adds a cache:

```python
storage._adj_cache: dict[str, set[str]] | None
storage._n_active_edges_cached: int
```

- Built lazily on first `Storage.adjacency()` call.
- Maintained incrementally on `insert_edge`.
- Invalidated on `deprecate_edge` (set to None; rebuilt on next read).

`Storage.n_active_edges()` is served from the cached count when
available — O(1). `Memory.recall(mode="auto")` short-circuits to
symmetric mode when this returns 0, avoiding the full edge load on
bulk-mode corpora.

### 12.5 Decoupled symmetric retrieval (v0.6)

In v0.5 the symmetric component was `s = (f + b) / 2` — averaged
prompted views. The forward/backward prompts (~50 chars each)
dominated the embedding for short queries against long-document
candidates, causing Recall to lag plain cosine RAG by 0.10–0.25 on
LongMemEval factual question types.

**v0.6 fix.** New `Embedder.embed_symmetric(text)` method: returns the
symmetric retrieval vector. `BGEEmbedder` overrides it to encode raw
text without prompt prefix. `Node.s_embedding` is a new optional field
that stores this vector. Storage prefers `s_embedding` when present,
falls back to `(f+b)/2` for legacy nodes.

The Γ-algebra (which needs `f ≠ b`) continues to use the prompted
views. Only the symmetric retrieval cosine is decoupled. See MATH §2.6
for the identifiability argument.

### 12.6 Batch BGE encoding

`BGEEmbedder.embed_batch(texts)` returns `(f_list, b_list, s_list)` for
an entire span batch in one inference call. The write pipeline uses it
when available, falling back to per-span `embed_dual + embed_symmetric`
otherwise. Drops per-write latency from O(spans · BGE_inference) to
O(BGE_inference) — sentence-transformers handles list batching
natively.

### 12.7 Consolidator batch edge induction

`Memory.consolidate(induce_edges=True)` scans active nodes that have
no incident edges, runs `topk_neighbors_for_gamma` per isolated node,
and inserts edges where `|γ| > THRESH_GAMMA`. This is the offline
counterpart to bulk-document ingest: ingest a corpus fast (no edges),
consolidate later (induce edges in batch, prune via BMRS, refine via
mean-field).

---

## 13. Test coverage map

| Layer | Tests | Pass/Total |
|---|---|---:|
| Γ-algebra | `test_gamma.py`, `test_identifiability.py` | 12/12 |
| Storage | `test_storage.py` (incl. v0.5 scope subset, v0.6 s_embedding + adjacency cache) | 14/14 |
| Pipeline | `test_pipeline.py` (incl. v0.5 bulk-mode) | 8/8 |
| Quality classifier | `test_quality.py`, `test_bio_fingerprint.py` | 4 + 4 |
| Edge classifier | `test_edge_classifier.py` | 6/6 |
| Splitter | (within pipeline) | covered |
| Walk | `test_walk.py` | 5/5 |
| Subgraph extraction | `test_pcst.py`, `test_pcsf.py` | 7/7 |
| Auto-router | `test_router.py`, `test_intent.py` | 12/12 |
| Spectral / topology / curvature / projector | `test_graph_*.py` | 19/19 |
| Sheaf H¹ | `test_sheaf.py` | 5/5 |
| Bound (PAC-Bayes + CRC) | `test_bound.py`, `test_rag_bound.py`, `test_conformal.py` | 12 + |
| Consolidator (BMRS, mean-field, motif, induce_edges) | `test_consolidate.py` (incl. v0.5 induce_edges) | 4 + |
| API | `test_api.py` | 7/7 |
| Server | `test_server.py` | 4/4 |
| Letta adapter | `test_letta_adapter.py` | 3/3 |
| Telemetry | `test_telemetry.py` | 4/4 |
| TF-IDF embedder | `test_tfidf_embedder.py` (incl. v0.6 embed_symmetric) | 5/5 |
| PMED scoring | `test_pmed_score.py` | 6/6 |
| Spectral projector | `test_spectral_projector.py` | 4/4 |

**Total: 154 passing.** Suite runs in ~1.6 sec on the reference
hardware. Every behavior the public API or math doc claims has at
least one test.

---

*Status: architecture spec v1. Maps 1:1 to implementation in `src/recall/`. Update when interfaces change.*
