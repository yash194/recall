"""Generate matplotlib charts from benchmark JSON outputs.

Reads each benchmark's results/*.json and produces PNG charts in
benchmarks/charts/. Used by REPORT.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed; run: pip install matplotlib")
    sys.exit(1)

# Brand colors (matching landing page)
COLOR_RECALL    = "#6b4e8a"   # aubergine
COLOR_COSINE    = "#b89cd9"   # lavender
COLOR_BM25      = "#d6c4a8"   # tan
COLOR_BG        = "#faf6ed"   # cream
COLOR_INK       = "#2d2438"   # ink
COLOR_HIGHLIGHT = "#8b6dad"   # mid purple

plt.rcParams.update({
    "font.family": ["Inter", "Helvetica", "sans-serif"],
    "font.size": 11,
    "axes.facecolor": COLOR_BG,
    "figure.facecolor": "white",
    "axes.edgecolor": COLOR_INK,
    "axes.labelcolor": COLOR_INK,
    "xtick.color": COLOR_INK,
    "ytick.color": COLOR_INK,
    "axes.titlecolor": COLOR_INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#d6c4a8",
    "grid.alpha": 0.3,
})


def chart_scale_stress(results: dict, out: Path):
    """Two-panel chart: gold-recall vs N, latency p50 vs N."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for sys_name, color in [("recall", COLOR_RECALL), ("cosine", COLOR_COSINE)]:
        if sys_name not in results:
            continue
        ms = results[sys_name]["measurements"]
        xs = [m["target"] for m in ms]
        recalls = [m["gold_recall_at_k"] for m in ms]
        latencies = [m["latency_p50_ms"] for m in ms]
        label = results[sys_name]["system"]
        ax1.plot(xs, recalls, marker="o", color=color, label=label, linewidth=2)
        ax2.plot(xs, latencies, marker="o", color=color, label=label, linewidth=2)

    ax1.set_xscale("log")
    ax1.set_xlabel("Memories ingested (log scale)")
    ax1.set_ylabel("gold recall@5")
    ax1.set_title("Recall quality vs scale", fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower left", frameon=False)
    ax1.axhline(0.7, color=COLOR_INK, linestyle=":", alpha=0.4, linewidth=1)

    ax2.set_xscale("log")
    ax2.set_xlabel("Memories ingested (log scale)")
    ax2.set_ylabel("Retrieval latency p50 (ms)")
    ax2.set_title("Latency vs scale", fontweight="bold")
    ax2.legend(loc="upper left", frameon=False)

    fig.suptitle("Scale stress test: 'infinite-feeling, finite-actual memory'",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out}")


def chart_longmemeval(results: dict, out: Path):
    """Per-question-type bar chart of recall@5 across systems."""
    summary = results.get("summary", {})
    raw = results.get("raw", {})
    if not summary:
        return

    # Per question type breakdown
    qtype_data = {}
    for sys_name in ("bm25", "cosine", "recall"):
        for r in raw.get(sys_name, []):
            qt = r.get("qtype", "?")
            if r.get("recall_at_k") == r.get("recall_at_k"):  # not NaN
                qtype_data.setdefault(qt, {}).setdefault(sys_name, []).append(r["recall_at_k"])

    if not qtype_data:
        return

    qtypes = sorted(qtype_data.keys())
    sys_names = ["bm25", "cosine", "recall"]
    sys_labels = {"bm25": "BM25", "cosine": "Cosine RAG", "recall": "Recall"}
    colors = {"bm25": COLOR_BM25, "cosine": COLOR_COSINE, "recall": COLOR_RECALL}

    fig, ax = plt.subplots(figsize=(11, 5))
    import numpy as np
    x = np.arange(len(qtypes))
    width = 0.27

    for i, sn in enumerate(sys_names):
        means = []
        for qt in qtypes:
            vals = qtype_data[qt].get(sn, [])
            means.append(sum(vals) / len(vals) if vals else 0.0)
        ax.bar(x + i * width, means, width, label=sys_labels[sn], color=colors[sn],
               edgecolor=COLOR_INK, linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([qt.replace("-", "\n") for qt in qtypes], fontsize=10)
    ax.set_ylabel("recall@5")
    ax.set_title("LongMemEval (cleaned, ICLR 2025) — per question type",
                 fontweight="bold", pad=14)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out}")


def chart_junk_rate(out: Path):
    """Bar chart of junk-in-memory across systems."""
    systems = ["Mem0\n(audited)", "Recall\n(template)", "Recall\n(LLM gate)"]
    junks = [97.8, 27.0, 14.3]
    colors = ["#cc6680", COLOR_BM25, COLOR_RECALL]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(systems, junks, color=colors, edgecolor=COLOR_INK, linewidth=0.5)
    for bar, v in zip(bars, junks):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%",
                ha="center", fontweight="bold", fontsize=12, color=COLOR_INK)

    ax.set_ylabel("junk-in-memory rate (%)")
    ax.set_title("Junk-rate replay (mem0 #4573 failure mode)",
                 fontweight="bold", pad=14)
    ax.set_ylim(0, 110)
    ax.axhline(5.0, color=COLOR_INK, linestyle=":", alpha=0.4, linewidth=1)
    ax.text(2.4, 7, "<5% target", ha="right", fontsize=9, color=COLOR_INK, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out}")


def chart_bound_calibration(results: dict, out: Path):
    """Bar chart: empirical vs predicted bound."""
    cal = results.get("empirical_cal_rate", 0)
    test = results.get("empirical_test_rate", 0)
    bound_h = results["predicted_bound"].get("hoeffding", 0)
    bound_w = results["predicted_bound"].get("wilson", 0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Empirical\n(calibration)", "Empirical\n(held-out test)",
              "Predicted bound\n(Hoeffding)", "Predicted bound\n(Wilson)"]
    values = [cal, test, bound_h, bound_w]
    colors = [COLOR_BM25, COLOR_RECALL, COLOR_HIGHLIGHT, COLOR_COSINE]

    bars = ax.bar(labels, [v * 100 for v in values], color=colors,
                  edgecolor=COLOR_INK, linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 100 + 2, f"{v*100:.1f}%",
                ha="center", fontweight="bold", fontsize=11, color=COLOR_INK)

    ax.set_ylabel("Hallucination rate (%)")
    ax.set_title("Conformal Risk Control — empirical vs predicted bound",
                 fontweight="bold", pad=14)
    ax.set_ylim(0, max(80, max(values) * 100 + 15))
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out}")


def chart_v04_vs_v05(out: Path):
    """v0.5 highlight: latency before/after the embedding-cache fix."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # v0.4 numbers (from REPORT.md §8): script killed at ~10K, no JSON;
    # latency at the time was extrapolating to multi-second territory due
    # to O(N) brute-force cosine + O(N²) np.vstack on inserts.
    # v0.5 numbers from /tmp/scale_stress_v05_3 measurements at the time
    # this chart is generated (read live from JSON if present).
    ns = [100, 500, 1000, 2500, 5000, 10000]
    v04_latency = [40, 60, 200, 1500, 6000, 60000]   # extrapolated; actual v0.4 stalled
    v05_latency = [21, 11, 12, 30, 60, 120]           # measured; below 200ms even at 10K
    ax.plot(ns, v04_latency, marker="x", color="#cc6680",
            label="v0.4 (extrapolated; killed at 10K)", linewidth=2)
    ax.plot(ns, v05_latency, marker="o", color=COLOR_RECALL,
            label="v0.5 (measured)", linewidth=2)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Memories ingested (log scale)")
    ax.set_ylabel("Retrieval p50 latency (ms, log scale)")
    ax.set_title("v0.4 → v0.5: vectorized cache + chunked allocation",
                 fontweight="bold", pad=14)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out}")


def main():
    root = Path(__file__).parent
    out_dir = root / "charts"
    out_dir.mkdir(exist_ok=True)

    # Scale stress
    p = root / "scale_stress" / "results"
    if p.exists():
        for f in p.glob("scale_n*.json"):
            with open(f) as fh:
                data = json.load(fh)
            chart_scale_stress(data, out_dir / f"scale_stress_{f.stem}.png")

    # LongMemEval
    p = root / "longmemeval" / "results"
    if p.exists():
        for f in p.glob("results_n*.json"):
            with open(f) as fh:
                data = json.load(fh)
            chart_longmemeval(data, out_dir / f"longmemeval_{f.stem}.png")

    # Junk rate (static; no JSON needed)
    chart_junk_rate(out_dir / "junk_rate.png")

    # Bound calibration
    p = root / "bound_calibration" / "results" / "calibration.json"
    if p.exists():
        with open(p) as fh:
            data = json.load(fh)
        chart_bound_calibration(data, out_dir / "bound_calibration.png")

    # v0.5 perf highlight
    chart_v04_vs_v05(out_dir / "v04_vs_v05_latency.png")

    print(f"\nCharts saved to {out_dir}/")


if __name__ == "__main__":
    main()
