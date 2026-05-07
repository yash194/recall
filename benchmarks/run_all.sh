#!/usr/bin/env bash
# Run the full Recall benchmark suite.
# Usage:
#   ./benchmarks/run_all.sh                           # quick (n=5-15)
#   ./benchmarks/run_all.sh full                      # full LongMemEval n=500
#   OPENAI_API_KEY=sk-... ./benchmarks/run_all.sh     # also runs LLM-bound calibration

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src

MODE="${1:-quick}"

if [[ "$MODE" == "full" ]]; then
  LME_N=500
  HPQ_N=100
  STRESS_N=50000
  STRESS_MILES="100 500 1000 5000 10000 25000 50000"
  MAB_N=10
else
  LME_N=15
  HPQ_N=20
  STRESS_N=10000
  STRESS_MILES="100 500 1000 2500 5000 10000"
  MAB_N=5
fi

echo "=================================================================="
echo "Recall benchmark suite — mode=$MODE"
echo "=================================================================="

echo ""
echo "[1/6] HotpotQA distractor (n=$HPQ_N)"
python benchmarks/hotpotqa_bge/run.py --n "$HPQ_N" --algo greedy --mode auto

echo ""
echo "[2/6] LongMemEval cleaned (n=$LME_N)"
python benchmarks/longmemeval/run.py --n "$LME_N" --k 5 --systems bm25 cosine recall

echo ""
echo "[3/6] Scale stress test (max=$STRESS_N)"
python benchmarks/scale_stress/run.py --max-n "$STRESS_N" --milestones $STRESS_MILES

echo ""
echo "[4/6] Junk replay (n=2000, template gate)"
python benchmarks/junk_replay/run.py --n 2000

if [[ -n "$OPENAI_API_KEY" ]]; then
  echo ""
  echo "[5/6] Junk replay with LLM gate (n=100)"
  python benchmarks/junk_replay/run.py --llm --n 100

  echo ""
  echo "[6/6] Hallucination bound calibration (real LLM)"
  python benchmarks/bound_calibration/run.py --n-cal 30 --n-test 10
else
  echo ""
  echo "[5,6/6] SKIPPED — set OPENAI_API_KEY to run LLM benchmarks"
fi

echo ""
echo "[7/7] MemoryAgentBench (n=$MAB_N per split)"
python benchmarks/memoryagentbench/run.py --n "$MAB_N" \
  --splits Accurate_Retrieval Conflict_Resolution \
  --systems bm25 cosine recall

echo ""
echo "=================================================================="
echo "Done. Aggregate report at benchmarks/REPORT.md"
echo "Raw JSON at benchmarks/<bench>/results/"
echo "=================================================================="
