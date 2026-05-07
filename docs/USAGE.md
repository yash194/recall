# USAGE — How to actually use Recall

> Every supported way to use Recall, with copy-pasteable commands and config
> files. Pick the path that matches your situation. Skip the rest.

---

## 0. Pick your path in 30 seconds

| You are… | Use this | Section |
|---|---|---|
| Building an AI agent / app in Python | **Python library** | [§2](#2-python-library) |
| Using Claude Desktop, Cursor, Cline, Continue, Zed, or Windsurf | **MCP server** | [§3](#3-mcp-server) |
| Running scripted ingest from the shell (a cron job, a one-off pipeline) | **`recall` CLI** | [§4](#4-recall-cli) |
| Want a personal "second brain" / knowledge graph | **`recall me` personal CLI** | [§5](#5-recall-me-personal-cli) |
| Hosting Recall as a service for a team | **FastAPI HTTP server** | [§6](#6-self-hosted-http-server) |
| Just want to see numbers from the benchmarks | **Examples + benchmarks** | [§7](#7-examples-and-benchmarks) |

You can run any combination of these against the **same SQLite database** —
the MCP server, the CLI, and the Python library all read/write the same file
when pointed at it.

---

## 1. Install

The package is `recall`. Required: Python 3.10+. SQLite is stdlib.

### 1.1 Quick install (Python lib only)

```bash
pip install typed-recall
```

This gives you `recall.Memory`, the deterministic `HashEmbedder` and
`MockLLMClient`, the SQLite storage, and the CLI scripts (`recall`,
`recall me`). Zero non-Python dependencies.

### 1.2 With BGE neural embedder (recommended for real work)

```bash
pip install "recall[embed-bge]"
```

Adds `sentence-transformers` + `torch`. Lets you use
`BGEEmbedder("BAAI/bge-small-en-v1.5")` — the one used for every benchmark.

### 1.3 With everything

```bash
pip install "recall[embed-bge,llm-openai,server,mcp,graph]"
```

| Extra | Adds |
|---|---|
| `embed-bge` | `sentence-transformers`, `torch` — neural embeddings |
| `llm-openai` | `openai` SDK — for `OpenAIClient` and `bounded_generate` with real LLM |
| `llm-anthropic` | `anthropic` SDK — same, with Claude |
| `server` | `fastapi`, `uvicorn`, `pydantic` — HTTP server |
| `mcp` | `mcp>=1.0` — MCP stdio server |
| `graph` | `scipy`, `networkx`, `POT`, `gudhi` — full spectral / topology / curvature / sheaf primitives |

### 1.4 Develop from source

```bash
git clone https://github.com/yash194/recall.git
cd recall
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/ -q   # 154 passing
```

---

## 2. Python library

### 2.1 Three lines

```python
from recall import Memory

mem = Memory(tenant="my_app")
mem.observe("user message", "agent reply", scope={"project": "platform"})
result = mem.recall("what did we decide about queues?", scope={"project": "platform"})
```

That's it. SQLite database is created at `:memory:` by default. To persist:

```python
mem = Memory(tenant="my_app", storage="sqlite:///path/to/recall.db")
```

### 2.2 The five public methods

| Method | Purpose |
|---|---|
| `observe(user_msg, agent_msg, scope, source)` | Store one turn. Quality-gated, dedup'd, provenance-checked. |
| `recall(query, scope, mode, k, depth)` | Retrieve a connected reasoning subgraph. Modes: `auto` (default), `symmetric`, `path`, `hybrid`. |
| `bounded_generate(query, scope, bound, k, depth)` | Recall + LLM generate with structural-support check. Modes: `strict` (raise on unsupported), `soft` (flag), `off`. |
| `trace(generation)` | Full provenance trail — drawers, nodes, edges, audit entries, hallucination bound. |
| `forget(node_id, reason, actor)` | Surgical deprecate + cascade through edges + audit-log. |

Plus three convenience helpers added in v0.5/v0.6:

| Method | Purpose |
|---|---|
| `bulk_observe(texts, scope, source)` | Batch-ingest a list of texts via the bulk-document fast path (skips Γ-edge induction). |
| `consolidate(budget, scope, induce_edges, induce_k)` | Sleep-time worker — BMRS pruning + mean-field + motif extraction; with `induce_edges=True`, induces edges for bulk-ingested isolated nodes. |
| `stats()` | Health snapshot: active nodes, active edges, audit entries. |

### 2.3 With BGE + OpenAI (production-grade)

```python
import os
from recall import Memory, BGEEmbedder
from recall.llm import OpenAIClient
from recall.config import Config

cfg = Config()
cfg.THRESH_GAMMA = 0.005   # tighter Γ-edge threshold for noisy corpora
cfg.THRESH_WALK  = -0.02

mem = Memory(
    tenant="prod_app",
    storage="sqlite:///prod_recall.db",
    embedder=BGEEmbedder("BAAI/bge-small-en-v1.5"),
    llm=OpenAIClient(
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
    ),
    config=cfg,
)

# Observe
r = mem.observe(
    "Customer Acme signed a 3-year contract worth $4.2M starting Q3 2024.",
    "Confirmed — added to CRM.",
    scope={"team": "sales"},
)

# Bounded generate (hallucination bound enforced)
gen = mem.bounded_generate(
    "What is Acme's contract value?",
    scope={"team": "sales"},
    bound="strict",   # raises HallucinationBlocked if any claim is unsupported
)
print(gen.text)
print(f"CRC bound: {gen.bound_value:.3f}")
```

### 2.4 Bulk-document ingest (RAG over a corpus)

```python
from recall import Memory, BGEEmbedder

mem = Memory(
    tenant="docs",
    storage="sqlite:///docs_recall.db",
    embedder=BGEEmbedder("BAAI/bge-small-en-v1.5"),
)

with open("handbook.md") as f:
    chunks = [p.strip() for p in f.read().split("\n\n") if len(p.strip()) > 40]

# Fast path — no per-write edge induction
mem.bulk_observe(chunks, scope={"doc": "handbook"})

# Optional: induce Γ-edges offline
mem.consolidate(induce_edges=True, induce_k=6)

# Retrieve (auto-router will pick symmetric mode since the graph is sparse)
result = mem.recall("what's our refund policy?", scope={"doc": "handbook"}, k=5)
for n in result.subgraph_nodes:
    print(f"  {n.text[:140]}")
```

### 2.5 Multi-tenant isolation

Every `Memory(tenant="...")` is a hard boundary. Cross-tenant reads are
not possible from the public API.

```python
alice = Memory(tenant="alice", storage="sqlite:///shared.db")
bob   = Memory(tenant="bob",   storage="sqlite:///shared.db")

alice.observe("my favorite color is blue", "")
bob.recall("favorite color")        # returns nothing — bob can't see alice
```

### 2.6 Surgical forget + audit

```python
result = mem.recall("queue tech")
node_to_forget = result.subgraph_nodes[0].id

forget = mem.forget(node_to_forget, reason="user said outdated", actor="alice@co")
print(f"deprecated 1 node, {len(forget.deprecated_edge_ids)} cascading edges")

# Audit log shows it
for entry in mem.audit.for_target(node_to_forget):
    print(f"[{entry.timestamp}] {entry.operation} by {entry.actor}: {entry.reason}")
```

The deprecated node + edges remain in storage (not deleted) but are excluded
from retrieval. The audit row is append-only and exportable as JSONL.

---

## 3. MCP server

For **Claude Desktop, Cursor, Cline, Codex, Continue, Zed, Windsurf** —
any MCP-aware client. Recall exposes 8 tools the LLM can call directly.

### 3.1 Install

```bash
pip install "recall[mcp]"
# Or with uvx (preferred — no local install):
uvx --from recall recall-mcp --help
```

### 3.2 What tools the LLM gets

| Tool | What it does |
|---|---|
| `add_memory(text, scope?, agent_response?)` | Write a memory (gated, dedup'd) |
| `search_memory(query, k?, mode?)` | Top-k connected nodes + edges |
| `bounded_answer(query, scope?, mode?)` | Generate an answer, structurally bounded |
| `forget(node_id, reason)` | Surgical delete |
| `audit(target?)` | Provenance trail |
| `graph_health()` | Spectral / curvature / persistence summary |
| `consolidate(budget?)` | Run sleep-time consolidator |
| `stats()` | Active nodes / edges / audit count |

### 3.3 Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "recall": {
      "command": "uvx",
      "args": ["--from", "recall", "recall-mcp"],
      "env": {
        "RECALL_DB_DIR": "/Users/you/.recall"
      }
    }
  }
}
```

Restart Claude Desktop. The 8 Recall tools appear in the tool picker.

If you installed locally instead of via uvx:

```json
{
  "mcpServers": {
    "recall": {
      "command": "python",
      "args": ["-m", "recall.mcp_server"]
    }
  }
}
```

### 3.4 Cursor

Cursor reads MCP config from `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "recall": {
      "command": "uvx",
      "args": ["--from", "recall", "recall-mcp"]
    }
  }
}
```

Restart Cursor. In Composer, the Recall tools become callable.

### 3.5 Cline

Cline (VS Code extension) — settings: `cline.mcpServers`:

```json
{
  "cline.mcpServers": {
    "recall": {
      "command": "uvx",
      "args": ["--from", "recall", "recall-mcp"]
    }
  }
}
```

### 3.6 Continue

Continue (VS Code / JetBrains): edit `~/.continue/config.json`:

```json
{
  "mcpServers": [
    {
      "name": "recall",
      "command": "uvx",
      "args": ["--from", "recall", "recall-mcp"]
    }
  ]
}
```

### 3.7 Zed

Zed: edit `~/.config/zed/settings.json` or use the in-editor MCP UI:

```json
{
  "mcp": {
    "servers": {
      "recall": {
        "command": "uvx",
        "args": ["--from", "recall", "recall-mcp"]
      }
    }
  }
}
```

### 3.8 Windsurf

Windsurf: settings → MCP servers → add new with command `uvx --from recall
recall-mcp`.

### 3.9 Codex CLI

```bash
claude mcp add recall -- uvx --from recall recall-mcp
```

### 3.10 Test the MCP server stand-alone

```bash
RECALL_DB_DIR=~/.recall uvx --from recall recall-mcp
# It will sit on stdin/stdout speaking JSON-RPC. Send a list_tools request
# to verify it's alive. Most users won't do this — they just configure
# their MCP client and restart it.
```

### 3.11 Where the data lives

By default `~/.recall/<tenant>.db` (SQLite). Override with env var
`RECALL_DB_DIR`. Each tenant gets its own database file.

The same database file can be inspected/modified with the regular `recall`
CLI (§4), the personal CLI (§5), or the Python library — they're all the
same SQLite schema.

---

## 4. `recall` CLI

For one-off scripted use. Installed via `pip install recall` as the
`recall` command.

### 4.1 Subcommands

```bash
recall ingest <file>         # ingest a text file (or stdin)
recall inspect [--node ID]   # show stats / one node
recall forget <node_id>      # forget a node (cascade edges, audit log)
recall audit [--target ID]   # view audit log entries
recall audit --export        # export full log as JSONL
```

Common flags on every subcommand:

```
--tenant <name>     # default: "default"
--db <path>         # default: "./recall.db"
```

### 4.2 Examples

Ingest a markdown file with a project scope:

```bash
recall --tenant alice --db ~/recall.db ingest README.md \
  --scope '{"project": "infra"}'
```

Pipe arbitrary text in:

```bash
echo "We decided to adopt Bun 1.1.4 for the monorepo." | \
  recall --tenant alice --db ~/recall.db ingest \
  --scope '{"project": "platform"}'
```

Inspect one node:

```bash
recall --db ~/recall.db inspect --node abc1234567...
```

Forget with a reason:

```bash
recall --db ~/recall.db forget abc1234... --reason "user said outdated"
```

Export the full audit log as JSONL:

```bash
recall --db ~/recall.db audit --export > audit.jsonl
```

### 4.3 Combining with `find` / shell

Bulk-ingest a directory of markdown notes:

```bash
find ~/notes -name "*.md" -exec sh -c '
  recall --tenant me --db ~/.recall/me.db ingest "$1" \
    --scope "{\"file\": \"$(basename "$1")\"}"
' _ {} \;
```

---

## 5. `recall me` personal CLI

A purpose-built CLI for using Recall as a personal knowledge graph —
the "second brain" use case. Stores at `~/.recall/personal.db` by
default. Installed alongside `recall` via the same package.

### 5.1 Subcommands

```bash
recall me add "..."                    # add a memory
recall me ask "..."                    # search / query
recall me ask "..." --bounded          # generate bounded answer (needs LLM)
recall me trace                        # show audit log
recall me forget <node_id>             # delete a memory
recall me health                       # graph health (spectral, topology, curvature)
recall me consolidate                  # run sleep-time consolidator
recall me ingest <file_or_dir>         # bulk-ingest .md/.txt/.org files
```

Common flag:

```
--db <path>        # default: ~/.recall/personal.db
```

### 5.2 Walkthrough

```bash
# Add a few memories
recall me add "decided to use Postgres LISTEN/NOTIFY for the queue"
recall me add "switched to Redis Streams after we hit message-loss issues" --tag queue
recall me add "Bun 1.1.4 is the monorepo's package manager" --tag stack

# Ask a question — graph-walk retrieval
recall me ask "what queue tech are we using?"
# # 4 memories, 3 connections
#   [decision] switched to Redis Streams after we hit message-loss issues
#   [pivot]    decided to use Postgres LISTEN/NOTIFY for the queue
#   ...

# Bounded answer with a real LLM (set OPENAI_API_KEY first)
recall me ask "what queue tech and why?" --bounded
# Redis Streams. We adopted it after Postgres LISTEN/NOTIFY caused message
# loss under load.
# (hallucination bound: 0.187)

# Audit log
recall me trace --limit 10

# Memory health — spectral, topology, curvature
recall me health
# === Memory health ===
#   nodes: 14
#   edges: 22
#   density: 1.57
# === Connectivity (spectral) ===
#   λ_2 (spectral gap): 0.4218
#   ...

# Run sleep-time consolidation
recall me consolidate --budget 20

# Bulk-ingest a notes directory
recall me ingest ~/notes/
# ingested 312 memories from 47 file(s)
```

### 5.3 Bounded vs un-bounded `ask`

Without `--bounded`, `ask` returns the retrieved subgraph (your memories
that match). With `--bounded`, it adds an LLM call that *only sees* the
retrieved subgraph and refuses (or flags) any claim not structurally
supported. The bound value is printed below the answer.

For `--bounded` to work you need an LLM client configured. The default
falls back to a deterministic stub for local-only use. Set
`OPENAI_API_KEY` to enable real bounded generation.

---

## 6. Self-hosted HTTP server

For team deployments where you want a service exposing Recall over REST.

### 6.1 Install + run

```bash
pip install "recall[server]"
uvicorn recall.server:app --host 0.0.0.0 --port 8765
```

Database directory defaults to `./recall_data/`. Override with
`RECALL_DB_DIR=/var/lib/recall`.

### 6.2 Endpoints

| Method | Path | Body |
|---|---|---|
| `POST` | `/v1/memory/observe` | `{tenant, user_msg, agent_msg, scope?, source?}` |
| `POST` | `/v1/memory/recall` | `{tenant, query, scope?, mode?, k?, depth?}` |
| `POST` | `/v1/memory/bounded_generate` | `{tenant, query, scope?, bound?, k?, depth?}` |
| `POST` | `/v1/memory/forget` | `{tenant, node_id, reason?, actor?}` |
| `GET`  | `/v1/memory/audit?tenant=...&target=...&limit=...` | — |
| `GET`  | `/v1/memory/stats?tenant=...` | — |
| `POST` | `/v1/memory/consolidate?tenant=...&budget=...` | — |

### 6.3 Curl examples

Observe:

```bash
curl -sX POST localhost:8765/v1/memory/observe \
  -H 'Content-Type: application/json' \
  -d '{"tenant":"alice","user_msg":"we use Redis Streams","agent_msg":"OK","scope":{"project":"infra"}}'
```

Recall:

```bash
curl -sX POST localhost:8765/v1/memory/recall \
  -H 'Content-Type: application/json' \
  -d '{"tenant":"alice","query":"queue tech","mode":"auto","k":5}' | jq
```

Bounded generate:

```bash
curl -sX POST localhost:8765/v1/memory/bounded_generate \
  -H 'Content-Type: application/json' \
  -d '{"tenant":"alice","query":"what queue tech?","bound":"soft"}' | jq
```

### 6.4 Docker compose

```yaml
# docker-compose.yml
version: "3.8"
services:
  recall:
    image: python:3.11-slim
    working_dir: /app
    volumes:
      - .:/app
      - recall-data:/var/lib/recall
    environment:
      RECALL_DB_DIR: /var/lib/recall
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    ports: ["8765:8765"]
    command: >
      sh -c 'pip install "recall[server,embed-bge,llm-openai]" &&
             uvicorn recall.server:app --host 0.0.0.0 --port 8765'
volumes:
  recall-data:
```

```bash
docker compose up
```

---

## 7. Examples and benchmarks

### 7.1 Examples (run directly)

```bash
git clone https://github.com/yash194/recall.git
cd recall
pip install -e ".[embed-bge]"

# Quickstart — three observe + recall + bounded_generate
PYTHONPATH=src python examples/quickstart.py

# Full demo with all features
PYTHONPATH=src python examples/full_demo.py

# Real-LLM demo (needs OPENAI_API_KEY)
PYTHONPATH=src python examples/real_llm_demo.py
```

### 7.2 Run a benchmark

Each lives under `benchmarks/<name>/`:

```bash
PYTHONPATH=src python benchmarks/longmemeval/run.py --n 30 --systems bm25 cosine recall
PYTHONPATH=src python benchmarks/scale_stress/run.py --max-n 5000
PYTHONPATH=src python benchmarks/synthetic_gamma/run.py
```

JSON results in `benchmarks/<name>/results/`. PNG charts via:

```bash
PYTHONPATH=src python benchmarks/visualize.py
```

Full methodology + numbers in [`docs/BENCHMARKS.md`](BENCHMARKS.md).

---

## 8. Common recipes

### 8.1 Conversation memory for a chat agent

```python
mem = Memory(tenant=user_id, storage=f"sqlite:///agents/{user_id}.db",
             embedder=BGEEmbedder())

# At every turn:
mem.observe(user_message, agent_reply, scope={"session": session_id})

# When the agent needs to recall:
ctx = mem.recall(user_message, scope={"session": session_id}, mode="auto")
context_text = "\n".join(n.text for n in ctx.subgraph_nodes)
# Pass context_text to your prompt template.
```

### 8.2 RAG over a document corpus

```python
mem = Memory(tenant="docs", storage="sqlite:///docs.db", embedder=BGEEmbedder())

# Ingest once
chunks = chunk_documents("./corpus")  # your chunking
mem.bulk_observe(chunks, scope={"corpus": "v1"})
mem.consolidate(induce_edges=True)     # offline edge induction

# Query — auto-router picks symmetric for factual lookup
result = mem.recall(query, scope={"corpus": "v1"}, k=5)

# Or with a hallucination bound:
gen = mem.bounded_generate(query, scope={"corpus": "v1"}, bound="strict")
```

### 8.3 Multi-tenant SaaS

```python
def memory_for(user_id: str) -> Memory:
    return Memory(
        tenant=user_id,
        storage="sqlite:///shared.db",  # one file, scoped by tenant column
        embedder=BGEEmbedder(),
    )

# Per-request:
mem = memory_for(request.user_id)
mem.observe(...)
```

### 8.4 Auditable agent action trail

```python
gen = mem.bounded_generate(query, bound="strict")
trace = mem.trace(gen)

# trace contains:
#   - the exact subgraph the LLM was conditioned on
#   - the structurally-flagged claims (if any)
#   - the CRC hallucination bound
#   - every audit-log entry for every node / edge in the subgraph
trace_json = json.dumps({
    "answer": gen.text,
    "bound": gen.bound_value,
    "flagged": gen.flagged_claims,
    "audit": [{"op": e.operation, "ts": str(e.timestamp), "target": e.target_id}
              for e in trace.audit_entries],
}, indent=2)
```

Store `trace_json` next to the answer for compliance review.

### 8.5 Personal knowledge graph synced across devices

```bash
# On both machines
pip install typed-recall

# Sync ~/.recall/personal.db with your favorite mechanism (Syncthing,
# Resilio, Dropbox; SQLite is single-writer, so make sure only one
# device is writing at a time).

# Or use the HTTP server on a home server:
RECALL_DB_DIR=/srv/recall uvicorn recall.server:app --port 8765
# Then point the personal CLI at it (planned, v0.7).
```

---

## 9. Configuration

### 9.1 Environment variables

| Var | Effect | Default |
|---|---|---|
| `RECALL_DB_DIR` | Default DB directory for MCP / HTTP server | `~/.recall` (MCP) · `./recall_data` (HTTP) |
| `OPENAI_API_KEY` | Used by `OpenAIClient` if you instantiate it | none |
| `ANTHROPIC_API_KEY` | Used by Anthropic LLM client (planned) | none |

### 9.2 Tunable thresholds (`recall.config.Config`)

```python
from recall.config import Config
cfg = Config()
cfg.THRESH_QUALITY = 0.4         # quality gate (0–1)
cfg.THRESH_GAMMA   = 0.05        # min |Γ| for edge creation
cfg.THRESH_WALK    = -0.10       # min weight for walk traversal
cfg.K_NEIGHBORS    = 10          # top-k neighbors at write
cfg.DELTA          = 0.05        # PAC-Bayes / CRC confidence
cfg.BUDGET_SUBGRAPH = 10.0       # PCST budget
mem = Memory(tenant="t", config=cfg)
```

The defaults are calibrated for general-purpose conversational memory.
For dense reference corpora, lower `THRESH_GAMMA` to `0.005`. For
ultra-noisy chat logs, raise `THRESH_QUALITY` to `0.6`.

### 9.3 Picking an embedder

| Embedder | When to use |
|---|---|
| `HashEmbedder` | Tests, deterministic, zero-dep. Default if no embedder is provided. |
| `TfidfEmbedder` | No torch / transformers needed. Real lexical-semantic embeddings via char-ngram TF-IDF. |
| `BGEEmbedder("BAAI/bge-small-en-v1.5")` | The benchmark choice. 384-dim, CPU-friendly. |
| `BGEEmbedder("BAAI/bge-large-en-v1.5")` | Higher quality, slower. |

Pass via `Memory(embedder=...)`.

### 9.4 Picking a retrieval mode

```python
mem.recall(query, mode="auto")        # let the router decide (default)
mem.recall(query, mode="symmetric")   # cosine over s, no walk
mem.recall(query, mode="path")        # Γ-walk + PCST
mem.recall(query, mode="hybrid")      # both, fused via reciprocal rank
```

`auto` is correct for ~90% of use cases. Force `path` only when you know
the data has connected reasoning structure (causal chains, decisions,
debate transcripts).

---

## 10. Troubleshooting

### 10.1 "ImportError: BGEEmbedder requires …"

```bash
pip install "recall[embed-bge]"
```

### 10.2 "ImportError: MCP server requires …"

```bash
pip install "recall[mcp]"
```

### 10.3 MCP tools don't show up in Claude Desktop

1. Check the config file path is correct (macOS:
   `~/Library/Application Support/Claude/claude_desktop_config.json`).
2. Run `uvx --from recall recall-mcp --help` in a terminal — it should
   print the help. If it errors, install with
   `pip install "recall[mcp]"`.
3. Fully quit Claude Desktop (cmd+Q) and restart. A reload from the
   menu doesn't pick up new MCP servers.
4. Check Console.app / `~/Library/Logs/Claude/` for MCP startup errors.

### 10.4 Recall returns empty results

```python
print(mem.stats())
# {'active_nodes': 0, ...} → quality gate rejected everything.
```

Either:
- Your text matched a `QUALITY_NEGATIVE_TEMPLATES` pattern (boilerplate /
  system-prompt leak / hallucinated profile). Inspect with
  `mem.audit.since(...)` — rejection reasons are in `payload`.
- The scope you queried doesn't match any stored scope. v0.5 fixed exact-
  match → subset semantics; if you're on an older version, upgrade.

### 10.5 Latency climbs above 1 K nodes

You're probably on a pre-v0.5 install. Upgrade:

```bash
pip install --upgrade recall
```

v0.5 introduced the vectorized embedding cache; v0.6 added the adjacency
cache. Together they keep p50 retrieval in the 10–25 ms band through
N = 5 000.

### 10.6 `bounded_generate` raises `HallucinationBlocked`

Working as designed. The LLM's output contained a claim with no
structural support in the retrieved subgraph. Either:
- The subgraph is too narrow — increase `k`, expand `scope`, or run
  `consolidate(induce_edges=True)` to add edges.
- Use `bound="soft"` to flag instead of raise.
- Use `bound="off"` to skip the check entirely (not recommended).

### 10.7 Where do I see the audit log?

```python
for e in mem.audit.since(datetime.fromisoformat("2026-05-01")):
    print(e)
# Or export to JSONL:
print(mem.audit.export_jsonl())
```

CLI:

```bash
recall --db ./recall.db audit
recall --db ./recall.db audit --export > audit.jsonl
```

---

## 11. Where to look next

| Want | Read |
|---|---|
| Math behind every primitive | [`docs/MATH.md`](MATH.md) |
| System design + protocols | [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) |
| All benchmark numbers + charts | [`docs/BENCHMARKS.md`](BENCHMARKS.md) |
| Design rules / contributing | [`docs/PRINCIPLES.md`](PRINCIPLES.md), [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
| Citations | [`CITATIONS.bib`](../CITATIONS.bib) |
| What changed v0.4 → v0.6 | [`CHANGELOG.md`](../CHANGELOG.md) |

---

*Usage guide — Recall v0.6 — 2026-05-07. Every command in this document
runs against the current code. If a command fails on your install, please
file an issue.*
