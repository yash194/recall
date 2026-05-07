# Recall Benchmarks

> Reproducible head-to-head against published baselines and standard memory benchmarks.
>
> Every numerical claim in `README.md` and `landing/index.html` traces back
> to one of the scripts here.

---

## Quick start

```bash
cd recall/

# Install with all benchmark deps
pip install -e ".[embed-bge,llm-openai,server,mcp,graph]"
pip install rank_bm25 matplotlib

# Run the quick suite (~10-20 min on a laptop)
./benchmarks/run_all.sh

# Run the full suite (LongMemEval n=500, ~2-4 hours)
./benchmarks/run_all.sh full

# With LLM-bound benchmarks
OPENAI_API_KEY=sk-... ./benchmarks/run_all.sh
```

Each benchmark is independently runnable too:

```bash
PYTHONPATH=src python benchmarks/longmemeval/run.py --n 20
PYTHONPATH=src python benchmarks/scale_stress/run.py --max-n 10000
PYTHONPATH=src python benchmarks/memoryagentbench/run.py --n 5
PYTHONPATH=src python benchmarks/junk_replay/run.py --llm --n 100
PYTHONPATH=src python benchmarks/bound_calibration/run.py --n-cal 30 --n-test 10
PYTHONPATH=src python benchmarks/hotpotqa_bge/run.py --n 30
```

After running, generate charts and the report:

```bash
PYTHONPATH=src python benchmarks/visualize.py
# charts saved to benchmarks/charts/
```

---

## What each benchmark measures

| Directory | What it tests | Public dataset |
|---|---|---|
| `hotpotqa/` & `hotpotqa_bge/` | Multi-hop fact retrieval (n=20+) | HotpotQA distractor (HF) |
| `longmemeval/` | 5-ability long-conversation memory | LongMemEval-cleaned (HF, ICLR 2025) |
| `memoryagentbench/` | 4-ability agent memory incl. Conflict_Resolution | ai-hyz/MemoryAgentBench (HF, ICLR 2026) |
| `scale_stress/` | "Infinite-feeling, finite-actual" memory under 10K-50K ingest | synthetic |
| `junk_replay/` | mem0 #4573 failure-mode replication | synthetic, models the public audit |
| `bound_calibration/` | Empirical validation of CRC hallucination bound | synthetic + real LLM |
| `synthetic_gamma/` | Planted causal-chain Γ retrieval | synthetic |
| `bge_gamma/` | Same as synthetic_gamma but with real BGE | synthetic |
| `mem0_head_to_head/` | Mem0 vs Recall side-by-side (requires Qdrant + OpenAI) | synthetic |
| `v03_full_stack/` | All v0.3 features at once | synthetic |
| `baselines_comparison/baselines.py` | CosineRAG + BM25 implementations | shared util |

---

## Baselines

`baselines_comparison/baselines.py` provides three drop-in retrieval systems
implementing the same `add(text, scope)` / `search(query, k)` interface:

- **`CosineRAG`** — vanilla vector RAG (BGE-small-en-v1.5 + cosine top-k).
  This is what every "embeddings + vector DB" implementation is.
- **`BM25Index`** — classic BM25 keyword retrieval (rank_bm25 implementation).
  The non-neural baseline.
- **`RecallAdapter`** — wraps `recall.Memory` in the same interface.
  This is the apples-to-apples comparison.

All three are used in `longmemeval/`, `memoryagentbench/`, and `scale_stress/`.

---

## Reproducibility guarantees

1. **All datasets are public.** HotpotQA, LongMemEval-cleaned, and
   MemoryAgentBench all on Hugging Face. Auto-downloaded by the scripts.
2. **All baselines are open-source.** rank_bm25, sentence-transformers, scikit-learn.
3. **All seeds are fixed.** `random.Random(2026)` and `np.random.default_rng(42)`
   appear in every place randomness is used.
4. **All scripts run on a laptop.** No GPU required (BGE-small-en-v1.5 is CPU-friendly).
5. **All numbers in the README + landing page reference one of the JSON files
   in `<bench>/results/`.** If a number disagrees with the JSON, the JSON is right.

---

## Comparison to published baselines

For the LongMemEval benchmark specifically, the 2024-2026 published numbers
(from each system's marketing) are:

| System | LongMemEval (self-reported) | Independent reproduction |
|---|---:|---:|
| Mastra Observational Memory | 94.87% (GPT-5-mini) | — |
| Hindsight | 91.4% (Gemini-3 Pro) | — |
| Supermemory | 85.4% | — |
| Emergence | 86% | — |
| Zep / Graphiti | 71.2% | 63.8% (Letta) |
| Mem0 | ~70% | 49.0% (Letta) |
| **Recall** | (see `longmemeval/results/`) | this repo |

Per [Letta's "benchmarking AI agent memory" blog post](https://www.letta.com/blog/benchmarking-ai-agent-memory),
self-reported numbers in this space are not directly comparable due to
methodology drift — different LLMs used as judge, different retrieval k,
different sub-benchmarks scored.

Recall's reported numbers always include:
- The exact LLM and embedder used
- The exact `k` for retrieval
- The exact subset of LongMemEval evaluated
- A reproducible script

---

## Anti-patterns we explicitly avoid

1. **Self-grading with the same LLM** — we use exact-match / token-overlap
   scoring, not an LLM judge.
2. **Cherry-picking subsets** — we report per-question-type breakdowns, not
   just the headline number.
3. **Comparing different models** — we use the same BGE-small embedder for
   `cosine`, `recall`, and any neural retrieval. Only `bm25` is non-neural.
4. **Hand-tuned hyperparameters per benchmark** — Recall's `auto` mode picks
   its retrieval mode automatically; no per-benchmark tuning.

---

## Adding a new benchmark

1. Make a new directory `benchmarks/<name>/`
2. Create `run.py` with `argparse` and a `main()` that ends by saving JSON to `<name>/results/`
3. Use `baselines.py` for cosine + BM25 + Recall comparison
4. Document in this README
5. Add to `run_all.sh`

---

## Charts

`visualize.py` generates matplotlib PNGs from the JSON outputs. Color
scheme matches the landing page (cream + aubergine purple).

```bash
PYTHONPATH=src python benchmarks/visualize.py
```

Output goes to `benchmarks/charts/`:
- `scale_stress_*.png` — recall@5 and latency vs N
- `longmemeval_*.png` — per-question-type bar chart
- `junk_rate.png` — Mem0 vs Recall (template) vs Recall (LLM gate)
- `bound_calibration.png` — empirical vs predicted bound

---

## See also

- [`REPORT.md`](REPORT.md) — final report with all numbers consolidated
- [`../docs/MATH.md`](../docs/MATH.md) — math behind each primitive
- [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) — system design
