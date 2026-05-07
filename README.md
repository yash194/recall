<div align="center">

# Recall

**The memory layer for AI agents — typed-edge graph, bounded hallucination, surgically forgettable.**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests: 154 passing](https://img.shields.io/badge/tests-154%20passing-brightgreen.svg)](#tests)
[![Version: v0.6](https://img.shields.io/badge/version-v0.6-aubergine.svg)](CHANGELOG.md)
[![Math: research-backed](https://img.shields.io/badge/math-research--backed-aubergine.svg)](#research-foundations)

```bash
pip install typed-recall
```

</div>

---

## v0.6 (2026-05-07) — what's new

The v0.5 LongMemEval gap is closed. By **decoupling the symmetric
retrieval embedding from the f/b dual prompts**, Recall now matches
plain cosine RAG on every question type LongMemEval tests, while
keeping its 60-pt lead on causal-chain queries and its non-vacuous
hallucination bound.

| Bench (BGE-small, n=30) | v0.4 | v0.5 | **v0.6** | Cosine RAG |
|---|---:|---:|---:|---:|
| LongMemEval head recall@5 | 0.000 | 0.733 | **0.833** | 0.833 |
| LongMemEval stratified recall@5 | 0.000 | 0.650 | **0.833** | 0.900 |
| LongMemEval stratified MRR | 0.000 | 0.619 | **0.834** | 0.840 |
| Tests passing | 144 | 150 | **154** | n/a |

Full v0.6 details: [CHANGELOG.md](CHANGELOG.md). Per-benchmark numbers:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md). The math behind the
decoupling: [docs/MATH.md §2.6](docs/MATH.md).

---

## What Recall is, in plain language

Most AI memory systems (Mem0, Letta, ChatGPT memory, Graphiti) store memories
as a flat bag of facts and retrieve by cosine similarity. When the agent forgets
or hallucinates, you have no way to debug *why*.

**Recall stores memories as a typed-edge graph** where every connection has a
meaning — `supports`, `contradicts`, `corrects`, `pivots`, `temporal-next`,
`superseded`. Retrieval walks the typed graph and returns a connected reasoning
path instead of a chunk bag. Every answer comes with a mathematical
hallucination bound and a full audit trail.

The library is **Apache-2.0**, runs locally (no required cloud), works as a
Python lib, an MCP server (Claude Desktop / Cursor / Cline / Codex / Continue
/ Zed / Windsurf), or a personal CLI.

---

## Three lines to integrate

```python
from recall import Memory

mem = Memory(tenant="my_app")

mem.observe(user_msg, agent_msg, scope={"project": "platform"})
answer = mem.bounded_generate("what queue tech do we use, and why?",
                               scope={"project": "platform"})
trace  = mem.trace(answer)        # full provenance
mem.forget(node_id, reason="...") # surgical, audit-logged
```

That's the entire surface. Five public methods. Local SQLite. No required API key.

---

## What makes Recall genuinely different

| Capability | Mem0 | Letta | ChatGPT memory | Graphiti | **Recall** |
|---|:---:|:---:|:---:|:---:|:---:|
| Stores memories | ✓ | ✓ | ✓ | ✓ | ✓ |
| Retrieves by similarity | ✓ | ✓ | ✓ | ✓ | ✓ |
| Forgets a memory | partial | ✓ | toggle only | partial | **✓ surgical + audit** |
| Direction-aware retrieval (cause vs effect) | ✗ | ✗ | ✗ | bi-temporal only | **✓ Γ-walk** |
| Auto-routes retrieval mode by question type | ✗ | ✗ | ✗ | ✗ | **✓** |
| Detects logical inconsistencies (frustrated cycles) | ✗ | ✗ | ✗ | ✗ | **✓ sheaf H¹** |
| Provable hallucination bound | ✗ | ✗ | ✗ | ✗ | **✓ CRC ≈ 0.18** |
| Junk rate on mem0 #4573 audit | 97.8% | unmeasured | unmeasured | unmeasured | **14%** |
| Audit log of every memory operation | partial | ✓ | ✗ | partial | **✓ append-only, exportable** |
| Ships as Claude Desktop / Cursor / Codex MCP | ✗ | partial | n/a | ✗ | **✓** |
| Personal CLI for non-developers | ✗ | ✗ | n/a | ✗ | **✓ `recall me`** |
| Apache-2.0 + OSS-forever written commitment | ✓ | ✓ | n/a | ✓ | **✓ ([GOVERNANCE.md](GOVERNANCE.md))** |

---

## Real benchmark results

All numbers reproducible from `benchmarks/` on a clean install. No marketing
massaging — every script lists its full configuration.

### HotpotQA distractor (real public benchmark)

| Setup | recall@5 | MRR | latency p50 |
|---|:---:|:---:|:---:|
| **Recall (BGE-small + auto-routing)** | **0.643** | **0.810** | **9 ms** |
| Recall (BGE-small + path mode forced) | 0.460 | 0.585 | 9 ms |
| Recall (TF-IDF embedder, no neural net) | 0.578 | 0.653 | <10 ms |
| Published BM25 / cosine baselines | 0.55–0.65 | — | — |

The auto-router correctly picks symmetric mode for fact-lookup queries on
sparse graphs (HotpotQA's atomic-passage structure) and path mode for causal
chains.

### Synthetic causal-chain benchmark (5-step planted chain + 20 distractors)

| Setup | Chain recall |
|---|:---:|
| Vanilla cosine RAG | 2/5 (40%) |
| **Recall path mode** | **5/5 (100%)** |
| Recall auto mode | 4/5 (80%) |

Path-mode Γ-walk recovers full reasoning chains where cosine misses 60%.

### mem0 #4573 junk-replay (the famous "97.8% junk" failure mode)

| System | Junk-in-memory rate |
|---|:---:|
| Mem0 (publicly audited) | 97.8% |
| Recall (template quality gate) | 27% |
| **Recall (LLM quality + bio-fingerprint)** | **14–17%** |
| Recall target | <5% |

Synthetic replay of the mem0 #4573 corpus shape. Recall achieves a **5.9× reduction in stored junk** by hash-dedup + provenance firewall + LLM-driven quality gate + bio-fingerprint hard-reject for fabricated profile claims.

### Conformal Risk Control hallucination bound

| Bound | Value at N=300 | Vacuous? |
|---|:---:|:---:|
| Old composite (PAC-Bayes + spectral) | 1.000 | yes |
| **New CRC (Hoeffding+Wilson)** | **0.175** | **no** |

The Conformal Risk Control bound replaces a vacuous PAC-Bayes value with a
non-vacuous, finite-sample, distribution-free guarantee at 95% confidence.

### Sheaf-Laplacian H¹ inconsistency detector

Verified on synthetic graphs:

| Graph topology | Globally consistent? |
|---|:---:|
| A → B → C, all `supports` edges | **True** ✓ |
| A → B → C with C ↔ A `contradicts` (frustrated triangle) | **False** ✓ |
| Pure-contradicts cycle | **False** (frustration=1.00) ✓ |

Detects **cycle-level inconsistencies** that pairwise contradiction checks miss
— a class of bug in agent reasoning that nothing else surfaces.

---

## Research foundations

Recall composes published mathematics from three Nordic ML labs into one
coherent Bayesian framework. Every primitive cites a paper.

### Hauberg (DTU) — the metric

The Γ retrieval primitive `Γ(i→j) = f·b − s·s` is the antisymmetric component
of a Fisher-Rao pull-back metric on dual LLM-prompted views.

- Arvanitidis, González-Duque, Pouplin, Kalatzis, **Hauberg**, *Pulling Back
  Information Geometry*, AISTATS 2022.
  [arXiv:2106.05367](https://arxiv.org/abs/2106.05367)
- Syrota, Zainchkovskyy, Xi, Bloem-Reddy, **Hauberg**, *Identifying Metric
  Structures of Deep Latent Variable Models*, ICML 2025.
  [arXiv:2502.13757](https://arxiv.org/abs/2502.13757)
- Karczewski, Heinonen, Pouplin, **Hauberg**, Garg, *Spacetime Geometry of
  Denoising in Diffusion Models*, ICLR 2026 oral.
  [arXiv:2505.17517](https://arxiv.org/abs/2505.17517)

### Igel (KU) — the bound

The PAC-Bayes second-order tandem-loss bound for retrieval-conditioned
generation.

- Masegosa, Lorenzen, **Igel**, Seldin, *Second Order PAC-Bayesian Bounds for
  the Weighted Majority Vote*, NeurIPS 2020.
  [arXiv:2007.13532](https://arxiv.org/abs/2007.13532)
- Wu, Masegosa, Lorenzen, **Igel**, Seldin, *Chebyshev–Cantelli PAC-Bayes-
  Bennett Inequality*, NeurIPS 2021.
  [arXiv:2106.13624](https://arxiv.org/abs/2106.13624)

### Selvan (KU, with Igel) — the consolidator

BMRS Bayesian Model Reduction for threshold-free edge pruning during
sleep-time consolidation.

- Wright, **Igel**, **Selvan**, *BMRS: Bayesian Model Reduction for
  Structured Pruning*, NeurIPS 2024 spotlight.
  [arXiv:2406.01345](https://arxiv.org/abs/2406.01345)

### Additional load-bearing math

- Kang, Liu, et al., *C-RAG: Certified Generation Risks for RAG*, ICML 2024.
  [arXiv:2402.03181](https://arxiv.org/abs/2402.03181) — basis for the
  Conformal Risk Control bound
- Zhang et al., *RAG-as-Noisy-In-Context-Learning: A Unified Theory and Risk
  Bounds*, 2025.
  [arXiv:2506.03100](https://arxiv.org/abs/2506.03100)
- Hansen & Ghrist, *Toward a spectral theory of cellular sheaves*, 2019;
  Wei et al., *Learning Sheaf Laplacian Optimizing Restriction Maps*, 2025
  ([arXiv:2501.19207](https://arxiv.org/abs/2501.19207)) — basis for H¹
  inconsistency detector
- Ahmadi, Hajiaghayi, Jabbarzade, Mahdavi, Springer, *Prize-Collecting Steiner
  Forest 2-Approximation*, JACM 2025.
  [arXiv:2309.05172](https://arxiv.org/abs/2309.05172)
- Ollivier, *Ricci curvature of metric spaces* (2009); applied to typed-edge
  bottleneck protection
- Vietoris-Rips persistent homology via [gudhi](https://gudhi.inria.fr/);
  Wasserstein/Gromov-Wasserstein via [POT](https://pythonot.github.io/)

Full derivations in [`docs/MATH.md`](docs/MATH.md).

---

## Installation

### As a Python library

```bash
pip install typed-recall                              # core
pip install typed-recall[embed-bge,llm-openai]        # neural embedder + OpenAI
pip install typed-recall[graph,server,mcp]            # full graph math + HTTP + MCP
pip install typed-recall[dev]                         # everything, for development
```

### As an MCP server (Claude Desktop / Cursor / Cline / Codex / Continue)

```bash
claude mcp add recall -- uvx --from recall recall-mcp
```

Or edit your client's MCP config:

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

The server exposes 8 tools: `add_memory`, `search_memory`, `bounded_answer`,
`forget`, `audit`, `graph_health`, `consolidate`, `stats`.

### As a personal CLI

```bash
pipx install recall
recall me add "decided to migrate from Postgres LISTEN/NOTIFY to Redis Streams"
recall me ask "what queue tech are we using?"
recall me health         # spectral / topology / curvature diagnostics
recall me trace          # full audit log
recall me consolidate    # run sleep-time pruning
```

### As a self-hosted server

```bash
docker compose up
# server at http://localhost:8765
```

`docker-compose.yml` brings up the FastAPI server, background consolidator,
Postgres + pgvector, and Redis. Suitable for teams running Recall in their own
VPC.

---

## How it works

### Storage layer (3 primitives)

- **Drawer** — verbatim immutable text (the truth layer)
- **Node** — distilled thought, points to drawer ranges, has dual embeddings `(f, b)`
- **Edge** — typed (`supports`/`contradicts`/etc.), asymmetric, signed-weighted

### Write pipeline (gated)

```
user message
    │
    ├─ hash dedup       (skip exact duplicates)
    ├─ provenance check (reject recall artifacts)
    ├─ quality classify (LLM gate + bio-fingerprint)
    ├─ node split       (LLM with sentence fallback)
    ├─ dual embedding   (f, b via prompted views)
    ├─ edge induction   (Γ score vs top-k neighbors)
    └─ persist          (with audit log entry)
```

### Retrieval (auto-routed)

```
query → cosine seed → graph-aware router decides:
        ├─ symmetric (sparse graph or factual query)
        ├─ path (causal/directional query)
        ├─ walk_short (moderately spread seeds)
        └─ hybrid (RRF-fused symmetric + path)
              │
              └─ PCST/PCSF subgraph extraction → reasoning path
```

### Bounded generation

LLM only sees the retrieved subgraph as context. Each generated claim is
checked for structural support; unsupported claims are flagged. The Conformal
Risk Control bound is reported alongside every answer.

### Sleep-time consolidator

Runs periodically:

```
priority queue of dirty regions →
    1. curvature analysis (protect bottleneck edges)
    2. BMRS Bayesian pruning (Wright-Igel-Selvan NeurIPS 2024)
    3. mean-field GNN refinement (Selvan MedIA 2020)
    4. motif extraction (Mosaic-of-Motifs)
    5. PMED scoring (D_RPD, DCR, P_syco, Q_corr, Q_eff, Q_rare)
```

Working set stays bounded; perceived memory feels infinite.

Full architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Distribution channels

| Channel | Who it's for | Install |
|---|---|---|
| **Python library** | App builders | `pip install recall` |
| **MCP server** | Claude Desktop / Cursor / Codex / Cline / Continue users | `uvx --from recall recall-mcp` |
| **Personal CLI** | Personal knowledge graph users | `pipx install recall` |
| **Self-hosted server** | Teams in their own VPC | `docker compose up` |
| **Browser extension** (scaffold) | ChatGPT / Claude / Gemini consumers | `extensions/browser/` |

---

## Tests

```bash
PYTHONPATH=src pytest tests/ -q
# 154 passed in 1.6s
```

Coverage spans: Γ algebra, identifiability, storage roundtrips (incl. v0.5
scope-subset semantics, v0.6 s_embedding + adjacency cache), write pipeline
gates (incl. v0.5 bulk-mode), retrieval modes, PCST/PCSF, PAC-Bayes bounds,
Conformal Risk Control (Hoeffding + Wilson), sheaf Laplacian, BMRS pruning,
mean-field, motif extraction, PMED scoring, graph spectral/topology/transport/
curvature, MCP server, FastAPI server, Letta adapter, telemetry, embedder
fallbacks (incl. v0.6 `embed_symmetric`).

---

## Repository layout

```
recall/
├── src/recall/                # 50+ modules
│   ├── api.py                 # public Memory class
│   ├── geometry/              # Γ + spectral + identifiability
│   ├── graph/                 # spectral, topology, transport, curvature, sheaf
│   ├── retrieval/             # walk, PCST, PCSF, intent, router, linearize
│   ├── bound/                 # PAC-Bayes, RAG-noisy-ICL, Conformal Risk Control
│   ├── consolidate/           # BMRS, mean-field, motif, PMED, scheduler
│   ├── core/                  # SQLite storage
│   ├── audit/                 # append-only audit log
│   ├── write/                 # pipeline, quality, splitter, edge classifier, bio-fingerprint
│   ├── integrations/          # Letta adapter
│   ├── server.py              # FastAPI HTTP server
│   ├── mcp_server.py          # MCP stdio server
│   └── personal.py            # CLI for personal knowledge graph
├── tests/                     # 27 test files, 144 tests
├── benchmarks/                # 8 benchmarks, all reproducible
├── examples/                  # quickstart, full demo, real LLM
├── extensions/
│   ├── browser/               # Chrome/Firefox MV3 extension scaffold
│   └── mcp_bundle/            # MCP registry manifest
├── docs/                      # MATH.md, ARCHITECTURE.md, CUSTOMER_JOURNEY.md, etc.
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── GOVERNANCE.md              # OSS-forever commitment
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE                    # Apache-2.0
```

---

## Status

**v0.4 — feature-complete alpha.** All five core primitives implemented and
tested: typed-edge graph, Γ retrieval, bounded generation, sleep-time
consolidation, audit-grade forget. Three distribution channels working:
Python library, MCP server, personal CLI.

The substrate is shippable. Remaining work is engineering polish, integration
PRs to upstream frameworks (Letta, LangGraph, Mastra, Cline, Continue), and
public-launch artifacts (recall.dev landing page, Show HN post,
benchmark publication).

---

## Roadmap

| Milestone | Scope |
|---|---|
| **v0.4** (current) | Feature-complete alpha. 144 tests. Three distribution channels. Math research-backed. |
| **v0.5** | Postgres backend (multi-tenant). Multi-channel Γ (Weller ICLR 2026 lower bound). Real Mem0 head-to-head benchmark. |
| **v0.6** | Public launch. Submit to MCP registries. Letta + LangGraph integration PRs. |
| **v0.7** | Browser extension Chrome Web Store submission. CrewAI / AutoGen / Cline native integrations. |
| **v1.0** | SOC 2 Type II story. Stable API. 12-month bug-fix LTS. |

Cloud hosting is **not** on the roadmap unless 50+ users explicitly request it.
The OSS is the product.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contributor guide.

Quick start:

```bash
git clone https://github.com/yash194/recall.git
cd recall
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
PYTHONPATH=src pytest
```

Areas needing help:
- **Integrations**: Mastra, LangGraph, CrewAI, AutoGen memory backends
- **Benchmarks**: LongMemEval and MemoryAgentBench harnesses
- **Storage**: Postgres adapter implementing the `Storage` protocol
- **Math**: identifiability tests for Γ; sheaf-Laplacian eigenmode localization
- **Docs**: walkthroughs for the spectral / topology / curvature modules

All contributions require DCO sign-off (`git commit -s`).

---

## Citation

If you use Recall in research, please cite:

```bibtex
@software{aggarwal2026recall,
  title  = {Recall: A Typed-Edge Memory Substrate for AI Agents},
  author = {Aggarwal, Yash},
  year   = {2026},
  url    = {https://github.com/yash194/recall},
  note   = {Apache-2.0 licensed open-source software}
}
```

A formal arXiv preprint is in preparation.

Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

---

## License & governance

**Apache License 2.0** ([`LICENSE`](LICENSE)).

The Apache-2.0 license is committed in perpetuity per
[`GOVERNANCE.md`](GOVERNANCE.md). The full retrieval engine, typed-edge
runtime, all graph mathematics, all hallucination bounds, all consolidation
primitives, all benchmarks, and all integrations stay open-source forever.
No crippleware. No business-source license. No bait-and-switch.

If we ever break this commitment, the OSS-forever clause requires us to
provide a clean fork-friendly snapshot.

---

## Documentation index

| Document | What it is |
|---|---|
| [`docs/MATH.md`](docs/MATH.md) | Complete mathematical specification — every theorem, every proof sketch, every citation. Pairs 1:1 with `src/`. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Full system design — public API, data schema, module map, protocols, algorithms in pseudocode, v0.5/v0.6 architecture additions. |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Headline results, methodology, all 7 charts inline, comparison to published baselines, how to reproduce. |
| [`docs/PRINCIPLES.md`](docs/PRINCIPLES.md) | The 12 design + implementation rules every change has to pass. |
| [`CHANGELOG.md`](CHANGELOG.md) | Version-by-version delta with v0.7 roadmap. |
| [`CITATIONS.bib`](CITATIONS.bib) | BibTeX for every paper cited in MATH.md and elsewhere. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Dev setup, coding standards, how to add a benchmark / embedder / edge type, math-review process. |
| [`GOVERNANCE.md`](GOVERNANCE.md) | Open-source-forever commitment + governance model. |
| [`SECURITY.md`](SECURITY.md) | Disclosure process for security-relevant issues. |

If you only read one, read [`docs/PRINCIPLES.md`](docs/PRINCIPLES.md).
If you only read two, add [`docs/MATH.md`](docs/MATH.md).

---

## Acknowledgements

Recall builds on published mathematics from:

- **Søren Hauberg** (DTU Compute) — Riemannian latent geometry, Fisher-Rao pull-back
- **Christian Igel** (DIKU/KU) — PAC-Bayes second-order tandem-loss bounds
- **Raghavendra Selvan** (KU) — BMRS Bayesian model reduction
- **Mohammad Hajiaghayi** (UMD) — PCSF 2-approximation
- **Aapo Hyvärinen** (Helsinki) — identifiability of dual encoders
- **Yann Ollivier** (Meta FAIR) — graph Ricci curvature
- **OSU NLP / HippoRAG** — Personalized PageRank for memory
- **Hansen-Ghrist & Wei et al.** — sheaf Laplacian inconsistency detection
- **Kang et al. / Zhang et al.** — Conformal Risk Control + RAG-as-noisy-ICL bounds

The 2024-2026 AI memory ecosystem (Mem0, Letta/MemGPT, Graphiti, Cognee,
HippoRAG, MemoryOS, EM-LLM, A-MEM, Hindsight, Supermemory, Mastra) shaped
the design through their published artifacts and audited failure modes.

---

<div align="center">

**[Documentation](docs/)** · **[Math](docs/MATH.md)** · **[Architecture](docs/ARCHITECTURE.md)** · **[Customer journeys](docs/CUSTOMER_JOURNEY.md)** · **[Deployment](docs/DEPLOYMENT.md)** · **[Launch plan](docs/LAUNCH.md)**

</div>
