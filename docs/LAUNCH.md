# Launch Plan — Recall v0.4 → public release

**Target launch**: Tuesday, June 2, 2026, 7:00 AM PT.
**License**: Apache-2.0 (committed in `GOVERNANCE.md`).
**Pricing at launch**: 100% OSS. Cloud waitlist opens T+30. Cloud paid tier T+90.

---

## Day-by-day timeline

### T-21 → T-15 (3 weeks out) — repo readiness
- [ ] All files in `Repo health checklist` (below) green
- [ ] CI passing on Python 3.10/3.11/3.12 matrix
- [ ] `pip install recall` works on a fresh Mac/Linux venv
- [ ] Quickstart in README runs end-to-end in <60s
- [ ] Demo GIF recorded (split-screen Mem0-vs-Recall, ≤8MB, in repo)
- [ ] 90s screencast for Twitter (with captions)
- [ ] Beta testers recruited: 10 indie + 2 framework integrators
- [ ] Domains live: `recall.dev`, `docs.recall.dev`, `bench.recall.dev`

### T-14 — distribution setup
- [ ] PyPI: register `recall`, `recall-ai`, `recall-mcp`, `recallai`, `recall-py` (defensive squats)
- [ ] PyPI Trusted Publisher OIDC configured (no long-lived tokens)
- [ ] npm: `@recall/sdk`, `recall-mcp`, `recall-client` with 2FA + provenance
- [ ] Docker Hub: `recall/server:0.4.0`, multi-arch (amd64+arm64)
- [ ] Homebrew tap (optional, week 2)

### T-10 → T-7 — GitHub setup
- [ ] Branch protection on `main`: require 1 review + CI green + signed commits + no force-push
- [ ] Labels: `good first issue`, `help wanted`, `bug`, `enhancement`, `integration:*`, `area:*`, `priority:p0/p1/p2`, `triage`
- [ ] CODEOWNERS (2 maintainers minimum on every path)
- [ ] Discussions enabled with Q&A / Show & Tell / Ideas
- [ ] GitHub Sponsors live
- [ ] `dependabot.yml` for security updates

### T-7 → T-3 — MCP registry submissions
- [ ] Official MCP Registry (`registry.modelcontextprotocol.io`) — file PR with `server.json`
- [ ] SkillsIndex
- [ ] PulseMCP
- [ ] Lobehub MCP marketplace
- [ ] FindMCP / GlobalMCP / AgentRank (aggregator tail)
- [ ] Kong MCP Registry (enterprise discovery)

### T-3 → T-1 — final dry-run
- [ ] Smoke test on a fresh box: `pip install recall` + run quickstart
- [ ] Run all benchmarks; numbers in README match
- [ ] Demo GIF confirmed playing in README on github.com
- [ ] Twitter thread drafted, ready to schedule
- [ ] Show HN post drafted (≤2000 char)
- [ ] LinkedIn post drafted
- [ ] Email blast drafted to 200-person warm list
- [ ] Pre-prepared HN comment seeds drafted (3 variants for different question types)

---

### T-0 — LAUNCH DAY (Tuesday, June 2, 2026)

| Time (PT) | Action |
|---|---|
| 05:30 | Final smoke test: fresh venv, install, run quickstart, run benchmark |
| 06:00 | Tag `v0.4.0`; release CI publishes to PyPI/npm/Docker Hub |
| 06:55 | Schedule Twitter thread for 7:00 PT |
| **07:00** | **Show HN goes live** |
| 07:00 | Twitter thread auto-posts |
| 07:05 | LinkedIn long-form post |
| 07:15 | r/LocalLLaMA self-post |
| 08:30 | r/AI_Agents self-post (stagger to avoid mod cross-post auto-removal) |
| 09:00 | Email blast to 200-person warm list |
| 10:00 | First-comment monitoring: respond to every HN comment within 15 min for first 4 hours |
| 12:00 | r/MachineLearning [P] post (link to math doc, not marketing) |
| 15:00 | If still on HN front page: post in Letta / Mastra / LangChain Discords |
| 18:00 | Post-mortem snapshot: stars, downloads, signups, top 5 issues |

### T+1 → T+7 — post-launch SLAs
- Issue first response: ≤4 hours during launch week
- PR first review: ≤24 hours during launch week
- Hotfix release: ≤6 hours for security or breaking-install
- v0.4.1 patch release at T+5 to prove the lights are on

### T+7 → T+30 — consolidation
- T+7: Blog post 1 — "The math of typed-edge memory" (2,500 words)
- T+14: Blog post 2 — "Recall vs Mem0 vs Letta vs Hindsight" (head-to-head)
- T+21: Blog post 3 — "Integrating Recall with Letta in 9 lines"
- T+30: Cloud waitlist opens at recall.dev/cloud
- T+45: First podcast episode (Latent Space targeted)

### T+90 — cloud paid tier launch
- Free / Pro $29 / Team $199 / Enterprise (see `DEPLOYMENT.md` cost shape)
- SOC 2 Type II in progress
- Public OSS-forever commitment renewed in changelog

---

## Show HN post (final copy)

**Title**:
> Show HN: Recall – a typed-edge memory substrate for AI agents (Apache-2.0)

**Body**:

> I built Recall because the current memory layer for agents (Mem0, Letta, Cognee, Graphiti) treats memory as opaque embeddings or untyped graph nodes. When you debug an agent that "forgot" something, you can't tell *why* the retrieval missed.
>
> Recall is a typed-edge substrate: every memory write declares an edge type (caused-by, refers-to, contradicts, supersedes, observed-at, temporal-next), and retrieval reasons over the typed graph instead of cosine similarity alone. The retrieval mode is graph-aware — it picks symmetric/path/hybrid based on seed-node dispersion, so factual lookups don't pay the path-extraction tax.
>
> What's in the repo today:
> - 12-line quickstart (works with OpenAI, Anthropic, Ollama)
> - Reproducible benchmarks vs Mem0, vector RAG, BM25 baselines
>   - HotpotQA distractor recall@5 = 0.643 (within published BM25/cosine range)
>   - Synthetic causal-chain recall = 100% with path-mode (vs 40% cosine)
>   - mem0-#4573 junk-replay: 16% junk-in-memory vs Mem0's 97.8%
> - MCP server + Letta / LangGraph / Mastra integration scaffolds
> - Sheaf-Laplacian H¹ inconsistency detector — catches frustrated cycles in the typed-edge graph (logical inconsistencies invisible to pairwise heuristics)
> - Conformal Risk Control bound on hallucination rate (non-vacuous, ~0.18 vs PAC-Bayes vacuous 1.0)
> - Math doc derives Γ retrieval primitive as antisymmetric Fisher-Rao pull-back (Hauberg ICML 2025 framework)
>
> Apache-2.0, no waitlist, `pip install recall`. Cloud tier comes at T+90 — OSS will never be crippleware. Commitment in `GOVERNANCE.md`.
>
> Demo: recall.dev · Repo: github.com/[org]/recall · Math: recall.dev/docs/math

---

## Twitter thread (final copy)

> **1/** Memory in AI agents is broken. When your agent forgets something, you can't tell why.
>
> Today: Recall — a typed-edge memory substrate that makes agent memory debuggable.
>
> Apache-2.0. `pip install recall`.
> [DEMO GIF — split-screen Mem0 vs Recall]
>
> **2/** Why typed edges?
>
> Mem0 / Letta / Supermemory store memory as embeddings + metadata. Cosine decides what comes back. When recall misses, you're guessing.
>
> Recall declares edge types at write-time: caused-by, contradicts, supersedes, observed-at. Retrieval traverses the type graph.
>
> **3/** The graph-aware router decides retrieval mode automatically:
>
> - Causal queries → path mode
> - Fact lookups → symmetric mode
> - Sparse graph → fallback to symmetric
>
> No more "do I use path or symmetric?" — the system picks.
>
> **4/** The numbers (full reproducible scripts in /benchmarks):
>
> - HotpotQA distractor recall@5: 0.643 (matches BGE/cosine baseline range)
> - Synthetic causal-chain recall: 100% (vs 40% cosine)
> - mem0-#4573 junk-replay: 16% junk vs Mem0's 97.8%
> - 144 unit tests passing
>
> **5/** The math: Γ = f·b − s·s as the antisymmetric Fisher-Rao pull-back of dual LLM-prompted views. Composes with PAC-Bayes (Igel) and BMRS pruning (Selvan).
>
> Plus: cellular sheaf H¹ for cycle-level inconsistency detection.
>
> Full derivation in docs/math.md.
>
> **6/** Distribution day-one:
> - Python lib (`pip install recall`)
> - MCP server (Claude Desktop / Cursor / Cline / Continue / Zed)
> - Personal CLI (`recall me`)
> - FastAPI server
> - Browser extension scaffold
>
> No framework lock-in. No waitlist.
>
> **7/** Apache-2.0 forever — committed in GOVERNANCE.md.
>
> Cloud tier comes Q3 for teams that want managed infra. OSS will never be crippleware.
>
> ⭐ github.com/[org]/recall
> 🌐 recall.dev
> 💬 discord.gg/recall

---

## Repo health checklist (must be green by T-7)

```
recall/
├── README.md                    # hero GIF → install → comparison → math link
├── LICENSE                      # Apache-2.0
├── NOTICE                       # required for Apache-2.0 if redistributing
├── CONTRIBUTING.md              # DCO sign-off + dev setup
├── CODE_OF_CONDUCT.md           # Contributor Covenant 2.1
├── SECURITY.md                  # security@recall.dev + 90-day disclosure
├── CHANGELOG.md                 # Keep-a-Changelog format
├── GOVERNANCE.md                # OSS-forever commitment
├── CITATION.cff                 # citeable for academic uptake
├── .github/
│   ├── ISSUE_TEMPLATE/{bug,feature,question,integration}.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── CODEOWNERS
│   ├── FUNDING.yml              # GitHub Sponsors + Open Collective
│   ├── dependabot.yml
│   └── workflows/
│       ├── test.yml             # py3.10/3.11/3.12 matrix; macOS+ubuntu
│       ├── release.yml          # PyPI Trusted Publisher OIDC
│       ├── security.yml         # CodeQL + pip-audit
│       ├── docs.yml             # mkdocs-material → docs.recall.dev
│       └── benchmarks.yml       # nightly LongMemEval + HotpotQA regression
├── docs/
│   ├── math.md
│   ├── architecture.md
│   ├── CUSTOMER_JOURNEY.md
│   ├── DEPLOYMENT.md
│   ├── LAUNCH.md (this)
│   ├── api/                     # auto-generated mkdocstrings
│   └── integration/{letta,langgraph,mastra,cline,continue}.md
├── examples/
│   ├── quickstart.py
│   ├── full_demo.py
│   ├── real_llm_demo.py
│   └── integration_letta.py
├── benchmarks/
│   ├── junk_replay/
│   ├── hotpotqa/
│   ├── synthetic_gamma/
│   ├── mem0_head_to_head/
│   ├── longmemeval/
│   └── README.md                # reproduction instructions
└── tests/                       # ≥80% coverage badge before launch
```

---

## Anti-patterns to avoid (from research agent)

1. **The "we're #1 on HN!" trap** — high engagement on launch day, no follow-through. Antidote: founder personally responds to every HN comment for 48 hours.
2. **Crippleware OSS to drive cloud** — kills credibility instantly. Public OSS-forever commitment in `GOVERNANCE.md` defuses this.
3. **Benchmarks that aren't reproducible** — instant credibility loss. Run benchmarks on a fresh AWS box from clean `pip install` before launch.
4. **Vendoring proprietary models in the demo** — quickstart requires API key user doesn't have → 60% conversion drop. Default to Ollama or include hosted demo endpoint.
5. **Ignoring the math doc** — researchers cite what's citeable. CITATION.cff + arXiv preprint by T+30.
6. **Week-2 silence** — schedule v0.4.1 release at T+5 with 3 small fixes.
7. **Discord without moderation** — 1,000 members + 1 bad actor = community is over. Have 2 mods on rotation by T+7.

---

## Success metrics — first 30 days

| Metric | Target | Stretch |
|---|---|---|
| GitHub stars | 1,000 | 5,000 |
| PyPI downloads / week | 1,000 | 10,000 |
| Discord members | 200 | 1,000 |
| Issues filed | 50 | 200 |
| PRs from contributors | 5 | 25 |
| Integration PRs landed (Letta/Mastra/LangGraph/Cline) | 2 | 4 |
| Reproducible benchmark runs by external users | 10 | 50 |
| Cloud waitlist signups | 100 | 1,000 |

If we miss the targets by >50%, the launch was wrong. If we hit stretch
on stars but miss on integration PRs, we got viral traffic but not
adoption — pivot to integration outreach in week 2.
