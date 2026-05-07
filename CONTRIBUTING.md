# Contributing to Recall

Thanks for your interest in contributing. Recall is open-source under
**Apache-2.0** with an [OSS-forever commitment](GOVERNANCE.md). All
contributions of substance — bug reports, documentation, tests, integrations,
benchmarks, math — are welcome.

This document covers everything you need to know to make a high-quality
contribution.

---

## Table of contents

1. [Code of conduct](#code-of-conduct)
2. [Ways to contribute](#ways-to-contribute)
3. [Getting started locally](#getting-started-locally)
4. [Development workflow](#development-workflow)
5. [Submitting a pull request](#submitting-a-pull-request)
6. [Code style](#code-style)
7. [Math contributions](#math-contributions)
8. [Integration contributions](#integration-contributions)
9. [Benchmark contributions](#benchmark-contributions)
10. [Documentation contributions](#documentation-contributions)
11. [Reporting security issues](#reporting-security-issues)
12. [Maintainer SLAs](#maintainer-slas)
13. [License](#license)

---

## Code of conduct

By participating, you agree to abide by the
[Contributor Covenant 2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

Reports of code-of-conduct violations: `conduct@recall.dev`.

---

## Ways to contribute

You don't need to be an expert in graph theory or PAC-Bayes bounds to make
real contributions. Specific high-leverage areas:

| What | Hardness | Example |
|---|---|---|
| **Documentation typos** | trivial | open a 1-line PR |
| **Bug reports** | easy | minimal reproduction in `tests/` |
| **Integration adapters** | medium | wire Recall as memory for a framework you use |
| **Benchmark wirings** | medium | implement a memory benchmark we don't have yet |
| **Storage backends** | hard | implement the `Storage` protocol for Postgres / KuzuDB |
| **Math primitives** | hard | new bound, new graph operator, new consolidator step |

Issues labeled `good first issue` are explicitly newcomer-friendly.

---

## Getting started locally

Prerequisites: Python 3.10+, git.

```bash
git clone https://github.com/yash194/recall.git
cd recall
python3 -m venv .venv
source .venv/bin/activate          # or `.venv\Scripts\activate` on Windows
pip install -e .[dev]
```

Verify everything works:

```bash
PYTHONPATH=src pytest tests/ -q
# 154 passed in <2s
```

> Before opening a PR, also read [`docs/PRINCIPLES.md`](docs/PRINCIPLES.md)
> — 12 rules that summarize how we make decisions on what lands. Most PR
> rejection reasons trace back to a violation of one of those rules.

Run the quickstart:

```bash
PYTHONPATH=src python examples/quickstart.py
```

Run a benchmark:

```bash
PYTHONPATH=src python benchmarks/synthetic_gamma/run.py
```

---

## Development workflow

1. **Find or file an issue**. For docs typos or trivial fixes, skip this step
   and just open the PR.

2. **Fork the repo**, then clone your fork. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/yash194/recall.git
   ```

3. **Create a feature branch** named `<your-handle>/<short-description>`:
   ```bash
   git checkout -b alice/postgres-storage-adapter
   ```

4. **Write your code + tests**. Public-API changes require:
   - At least one test in `tests/` covering the new behavior
   - A docstring on every new public function (Google style)
   - A note in `CHANGELOG.md` under `## [Unreleased]`

5. **Run the test suite**:
   ```bash
   PYTHONPATH=src pytest tests/ -q
   ```
   New tests should pass; existing tests must stay green. We aim for ≥80%
   coverage on changed files.

6. **Run the linters** (advisory at v0.4):
   ```bash
   pip install ruff
   ruff check src/recall
   ```

7. **Sign your commits** with the
   [Developer Certificate of Origin (DCO)](https://developercertificate.org/):
   ```bash
   git commit -s -m "Add Postgres storage adapter"
   ```
   Unsigned PRs will be auto-flagged by the CI bot.

8. **Push to your fork** and open a PR against `main`.

---

## Submitting a pull request

Fill in the PR template. Specifically:

- **Link the issue** (`Fixes #123`)
- **Describe what changed** in 2-4 sentences
- **List the tests** you added or updated
- **Note any breaking changes** explicitly (we hold these for major version bumps)
- **Include before/after numbers** if the change touches benchmarks or
  performance

CI must pass before review. The maintainer SLAs are listed
[below](#maintainer-slas).

---

## Code style

- **Python 3.10+**. Use modern type annotations (`str | None`, `list[int]`).
- **Type-annotate public APIs**. The `recall.api.Memory` class must not have
  untyped `Any` in its signature.
- **Docstrings** on every public function (Google style):
  ```python
  def gamma_score(f_i, b_i, f_j, b_j) -> float:
      """Compute the directional retrieval primitive Γ(i → j).

      Args:
          f_i: forward embedding of source text.
          b_i: backward embedding of source text.
          f_j: forward embedding of target text.
          b_j: backward embedding of target text.

      Returns:
          Scalar Γ score; positive ⇒ forward direction.
      """
  ```
- **One module per concern**. If a class exceeds ~150 LOC, consider splitting.
- **Comments explain *why*, not *what*.** "We use Hoeffding here because the
  bound is tighter for low rates" beats "// add hoeffding".
- **No deep import paths in user-facing code.** Re-export from `recall/__init__.py`.

We don't enforce a formatter at v0.4; PEP 8 with `ruff` advisory checks is
sufficient. Black/Ruff-format may be enabled in v1.0 if the community wants it.

---

## Math contributions

If you're adding a new mathematical primitive (a new bound, a new graph
operation, a new embedder, a new consolidator step), the PR must include:

1. **A new section in `docs/MATH.md`** with the formal definition,
   theorem (with proof sketch or full proof), and exact equation
   numbering. Cite the source paper(s) inline.
2. **An entry in `CITATIONS.bib`** for every cited paper. Use the
   existing format. Add a `note = {Recall §X — what this is used for}`
   field so the citation explains its load-bearing role.
3. **A test in `tests/test_*.py`** verifying a non-trivial property of
   the primitive (asymmetry, monotonicity, sign of gradient,
   fixed-point convergence, identifiability, bound-holds-on-test, etc.).
   No test, no merge.
4. **An update to `docs/MATH.md §12 (status table)`** mapping your
   theorem number → implementation file → test file → benchmark (if
   applicable).
5. **A benchmark or extension of an existing benchmark** that
   demonstrates behavioral change vs the prior implementation. If your
   primitive is supposed to win on some workload, the benchmark
   should show it.

If your contribution requires citing unpublished work, link a public
preprint (arXiv, OpenReview, or a personal site). **Recall's published
math must be reproducible from public sources** — see
[`docs/PRINCIPLES.md §4`](docs/PRINCIPLES.md).

### Math review process

Math-touching PRs (`src/recall/geometry/`, `src/recall/graph/`,
`src/recall/bound/`, `src/recall/consolidate/`) require an additional
review step:

1. The proof sketch in MATH.md must be self-contained enough that a
   reviewer with the cited papers in hand can check correctness.
2. At least one reviewer reads the proof and confirms the test
   exercises a non-trivial consequence (not a tautology).
3. If the math involves a bound, the bound must be either provably
   non-vacuous on a benchmark (like CRC §3.5) or marked `decorative
   until validated` in MATH.md.

### Adding a new embedder

The minimum required surface (see `src/recall/embeddings.py`):

```python
class MyEmbedder:
    @property
    def dim(self) -> int: ...

    def embed_dual(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Returns (forward, backward) prompted vectors used for Γ-edges."""

    def embed_symmetric(self, text: str) -> np.ndarray:
        """v0.6: returns the symmetric retrieval vector. For neural
        embedders, prefer raw text without prompt prefix."""
```

Optional but recommended: `embed_batch(texts) -> (f_list, b_list, s_list)`
for batch inference. The pipeline auto-detects this and uses it when
available — see `src/recall/embeddings.py:BGEEmbedder.embed_batch`.

Add a test in `tests/test_<your_embedder>.py` covering: dim correctness,
L2 normalization, `f ≠ b` (Γ-algebra requirement), `embed_symmetric`
fallback behavior.

### Adding a new edge type

1. Append to `Config.EDGE_TYPES` and `EdgeType` enum in `src/recall/types.py`.
2. Update `EdgeType.is_negative(...)` if your edge has negative-weight
   semantics.
3. Update `src/recall/write/edge_classifier.py` with the heuristic that
   selects this edge type from `(node, neighbor, gamma_score, gamma_anti)`.
4. Update sheaf restriction signs in `src/recall/graph/sheaf.py` if your
   edge has a non-`+1` restriction (e.g. `contradicts` uses `-1`).
5. Add a test in `tests/test_edge_classifier.py`.

### Adding a new bound

1. Module under `src/recall/bound/<your_bound>.py`.
2. The function returns a `dict[str, float]` with at least keys
   `bound`, `n`, `delta`, plus any intermediate quantities you want
   in telemetry.
3. Wire into `composite_hallucination_bound` in
   `src/recall/bound/rag_bound.py` so it's part of the `min(...)`
   composite (or document why it's not).
4. Add a test that the bound holds on a calibrated dataset.
5. Add a benchmark under `benchmarks/bound_calibration/` that exercises
   it end-to-end on real LLM output.

---

## Integration contributions

If you're adding a new framework integration (e.g., CrewAI, AutoGen, BAML):

1. Place under `src/recall/integrations/<framework>/`
2. Make the integration's runtime an *optional* dependency
   (`recall[integrations-crewai]`)
3. Add a runnable example under `examples/integration_<framework>.py`
4. Add a docs page at `docs/integration/<framework>.md`
5. Include a smoke-test in `tests/test_<framework>_adapter.py`
6. Update the comparison table in `README.md` if appropriate

Bonus: open a PR upstream against the framework's repo demonstrating the
Recall integration, and link it in your PR description. We'll boost it.

---

## Benchmark contributions

A benchmark contribution is a high-leverage way to make Recall measurably
better. New benchmarks should:

1. Live under `benchmarks/<name>/`
2. Include a `run.py` with `argparse` for at minimum `--n` (sample size)
3. Print a clear summary including: setup, headline metric, baseline
   reference numbers from published literature, latency stats
4. Be reproducible from a clean `pip install -e .` checkout — no external
   credentials or paid APIs by default
5. Document any optional credentials in a `README.md` next to `run.py`

Particularly wanted:

- **MemoryAgentBench** (ICLR 2026) — selective forgetting, multi-hop conflict
- **LongMemEval** — long-conversation memory
- **LoCoMo** — multi-modal long memory
- **MEMTRACK** — state tracking
- **GraphRAG-Bench** (ICLR 2026)
- **FactConsolidation** subset of MemoryAgentBench

---

## Documentation contributions

Documentation PRs are first-class. Areas:

- `README.md` — keep accurate as features land
- `docs/MATH.md` — add walkthroughs for primitives that lack them
- `docs/ARCHITECTURE.md` — keep diagrams synced with code
- `docs/CUSTOMER_JOURNEY.md` — add new personas as they appear in the wild
- `docs/integration/*.md` — write new integration guides
- Inline docstrings — improve any public-API docstring

For larger doc rewrites, file an issue first to align on direction.

---

## Reporting security issues

Please do **not** file public GitHub issues for security vulnerabilities.

Email `security@recall.dev`. Disclosure policy: 90-day window. See
[`SECURITY.md`](SECURITY.md) for full details.

---

## Maintainer SLAs

These are commitments to contributors, published so the community can hold
us to them.

| Action | Launch week (T+0 to T+7) | Steady state |
|---|---|---|
| First response on a new issue | ≤4 hours | ≤24 hours |
| First review on a new PR | ≤24 hours | ≤48 hours |
| Hot-fix release for breaking install | ≤6 hours | ≤24 hours |
| Security-vulnerability initial response | ≤48 hours | ≤48 hours |

We will publicly explain any miss in `CHANGELOG.md` and apologize.

---

## License

By contributing, you agree your contributions are licensed under
**Apache License 2.0** and that the Developer Certificate of Origin applies.

You retain copyright on your contributions. Recall does not require you to
sign a Contributor License Agreement (CLA) — the DCO sign-off in your commit
is sufficient.

---

## Recognition

Substantial contributors are listed in `AUTHORS.md` (auto-generated from
git history) and credited by name in release notes when their PRs land.

We're delighted you're here. Open the PR.
