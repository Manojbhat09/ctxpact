#!/usr/bin/env python3
"""Generate benchmark report with plots and tables from benchmark results.

Reads ~/ctxpact-bench/benchmark_results_{label}.json, produces:
  - ~/ctxpact-bench/benchmark_report_{label}.md  (markdown report per run)
  - ~/ctxpact-bench/benchmark_plots_{label}.png  (combined figure per run)

If multiple result files exist, also produces:
  - ~/ctxpact-bench/benchmark_comparison.md  (side-by-side comparison)
  - ~/ctxpact-bench/benchmark_comparison.png (comparison plots)

Usage:
  python benchmark_report.py                         # process all result files
  python benchmark_report.py --label standard        # process single run
  python benchmark_report.py --compare               # only generate comparison
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
STRATEGY_ORDER = ["header", "autosearch", "rlm", "rlm_v2", "rlm_v3", "toolcall", "chunking"]
STRATEGY_COLORS = {
    "header": "#4CAF50",
    "autosearch": "#2196F3",
    "rlm": "#FF9800",
    "rlm_v2": "#E65100",
    "rlm_v3": "#BF360C",
    "toolcall": "#9C27B0",
    "chunking": "#F44336",
    "embed": "#00BCD4",
    "compress": "#795548",
    "adaptive": "#607D8B",
    "icl": "#3F51B5",
    "rlm_v4": "#FF5722",
    "rlm_v5": "#009688",
    "rlm_v6": "#E91E63",
    "agentic": "#00BCD4",
    "subagent": "#8BC34A",
}
STRATEGY_DESC = {
    "header": "Section previews + recent full sections (no LLM extraction calls)",
    "autosearch": "Heuristic keyword extraction from query, grep search, assemble (no LLM extraction calls)",
    "rlm": "LLM generates semantic search terms, exact phrase grep, assemble (1 LLM call)",
    "rlm_v2": "Fixed RLM: word-level matching, IDF ranking, no summary bloat (1 LLM call)",
    "rlm_v3": "DSPy RLM: model writes Python to search the book iteratively (N LLM calls)",
    "toolcall": "Multi-turn tool-calling loop — model iteratively searches/reads (N LLM extraction calls)",
    "chunking": "Map-reduce chunked processing (N LLM extraction calls)",
    "embed": "Semantic embedding retrieval via chromadb/sentence-transformers (no LLM extraction calls)",
    "compress": "LLMLingua token-level prompt compression via GPT-2 (no LLM extraction calls)",
    "adaptive": "Hybrid query router — routes to embed or header based on query analysis (no LLM extraction calls)",
    "icl": "In-context learning compaction — turn-level selection with recency+similarity+role weighting (no LLM extraction calls)",
    "rlm_v4": "SRLM-inspired multi-candidate selection — runs embed+header+rlm_v2 in parallel, picks best by quality signals (1 LLM call)",
    "rlm_v5": "Enhanced programmatic exploration — 5 tools, IDF-scored search, exploration-first prompting (N LLM calls)",
    "rlm_v6": "Multi-signal RRF retrieval — chunked embedding + strict IDF + position boost, partial sections (1 LLM call)",
    "agentic": "Multi-agent A-RAG — query decomposition + parallel research agents with paragraph-level progressive disclosure (5-10 LLM calls)",
    "subagent": "Subagent architecture — planner decomposes query into subtasks, parallel subagents search independently, synthesizer ranks results (2 LLM calls)",
}
LLM_CALLS = {
    "header": "0", "autosearch": "0", "rlm": "1", "rlm_v2": "1",
    "rlm_v3": "N (DSPy)", "toolcall": "up to 10", "chunking": "N",
    "embed": "0", "compress": "0", "adaptive": "0", "icl": "0",
    "rlm_v4": "1", "rlm_v5": "up to 15",
    "rlm_v6": "1", "agentic": "5-10",
    "subagent": "2",
}


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def find_result_files() -> list[tuple[str, str]]:
    """Find all benchmark_results_*.json files. Returns [(label, path), ...]."""
    pattern = os.path.join(BENCH_DIR, "benchmark_results_*.json")
    files = glob.glob(pattern)
    results = []
    for f in sorted(files):
        basename = os.path.basename(f)
        # Extract label from benchmark_results_{label}.json
        label = basename.replace("benchmark_results_", "").replace(".json", "")
        results.append((label, f))
    # Also check for legacy benchmark_results.json (no label)
    legacy = os.path.join(BENCH_DIR, "benchmark_results.json")
    if os.path.exists(legacy) and legacy not in [f for _, f in results]:
        results.insert(0, ("legacy", legacy))
    return results


def aggregate_by_strategy(results: list[dict]) -> dict:
    """Group results by strategy and compute aggregates."""
    by_strategy = defaultdict(list)
    for r in results:
        by_strategy[r["strategy"]].append(r)

    agg = {}
    for strategy, entries in by_strategy.items():
        n = len(entries)
        passes = sum(1 for e in entries if e["all_pass"])
        agg[strategy] = {
            "n_queries": n,
            "n_pass": passes,
            "accuracy_rate": passes / n if n else 0,
            "avg_accuracy": sum(e["accuracy"] for e in entries) / n if n else 0,
            "avg_latency": sum(e["latency_seconds"] for e in entries) / n if n else 0,
            "min_latency": min(e["latency_seconds"] for e in entries),
            "max_latency": max(e["latency_seconds"] for e in entries),
            "total_latency": sum(e["latency_seconds"] for e in entries),
            "avg_prompt_tokens": sum(e["prompt_tokens"] for e in entries) / n if n else 0,
            "avg_completion_tokens": sum(e["completion_tokens"] for e in entries) / n if n else 0,
            "errors": sum(1 for e in entries if e["error"]),
            "entries": entries,
        }
    return agg


# ---------------------------------------------------------------------------
# Single-run report
# ---------------------------------------------------------------------------

def generate_plots(agg: dict, results: list[dict], output_path: str,
                   model_name: str = "", label: str = ""):
    """Generate a 2x2 figure with comparison plots for a single run."""
    strategies = [s for s in STRATEGY_ORDER if s in agg]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title_suffix = f" [{label}]" if label else ""
    fig.suptitle(
        f"ctxpact Extraction Strategy Benchmark{title_suffix}\n"
        f"Frankenstein (~102k tokens) on {model_name or 'local model'} (16k context)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # --- Plot 1: Accuracy Rate ---
    ax = axes[0, 0]
    acc_rates = [agg[s]["accuracy_rate"] * 100 for s in strategies]
    bars = ax.bar(strategies, acc_rates,
                  color=[STRATEGY_COLORS.get(s, "#999") for s in strategies],
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Accuracy Rate (%)")
    ax.set_title("Accuracy Rate (all keywords found)")
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, acc_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}%", ha="center", va="bottom", fontweight="bold")

    # --- Plot 2: Average Latency ---
    ax = axes[0, 1]
    avg_lats = [agg[s]["avg_latency"] for s in strategies]
    bars = ax.bar(strategies, avg_lats,
                  color=[STRATEGY_COLORS.get(s, "#999") for s in strategies],
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Avg Latency (seconds)")
    ax.set_title("Average Latency per Query")
    for bar, val in zip(bars, avg_lats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=9)

    # --- Plot 3: Average Prompt Tokens ---
    ax = axes[1, 0]
    avg_tokens = [agg[s]["avg_prompt_tokens"] for s in strategies]
    bars = ax.bar(strategies, avg_tokens,
                  color=[STRATEGY_COLORS.get(s, "#999") for s in strategies],
                  edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Avg Prompt Tokens")
    ax.set_title("Average Prompt Tokens (context sent to model)")
    for bar, val in zip(bars, avg_tokens):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    # --- Plot 4: Per-Query Accuracy Heatmap ---
    ax = axes[1, 1]
    query_ids = sorted(set(r["query_id"] for r in results))
    heatmap_data = []
    for s in strategies:
        row = []
        for qid in query_ids:
            entry = next((r for r in results if r["strategy"] == s and r["query_id"] == qid), None)
            row.append(entry["accuracy"] if entry else 0)
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data)
    im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(query_ids)))
    ax.set_xticklabels([f"Q{qid}" for qid in query_ids])
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies)
    ax.set_title("Per-Query Accuracy (green=pass, red=fail)")
    for i in range(len(strategies)):
        for j in range(len(query_ids)):
            val = heatmap_data[i, j]
            text = "P" if val >= 1.0 else ("~" if val > 0 else "F")
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val < 0.5 else "black", fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plots saved to {output_path}")


def generate_markdown_report(
    metadata: dict, agg: dict, results: list[dict], output_path: str,
    label: str = "", plots_filename: str = "benchmark_plots.png",
):
    """Generate a markdown report for a single run."""
    strategies = [s for s in STRATEGY_ORDER if s in agg]

    lines = []
    title_suffix = f" — {label}" if label else ""
    lines.append(f"# ctxpact Extraction Strategy Benchmark Report{title_suffix}")
    lines.append("")
    lines.append(f"**Date:** {metadata.get('timestamp', 'N/A')}")
    lines.append(f"**Model:** {metadata.get('model', 'N/A')}")
    lines.append(f"**Run Label:** {metadata.get('label', label or 'N/A')}")
    lines.append(f"**Test Document:** Frankenstein (pg84.txt)")
    lines.append(f"**Document Size:** {metadata.get('book_chars', 0):,} chars (~{metadata.get('book_est_tokens', 0):,} estimated tokens)")
    lines.append(f"**Model Context Window:** 16,384 tokens")
    lines.append(f"**Strategies Tested:** {len(strategies)}")
    lines.append(f"**Queries:** {metadata.get('num_queries', 0)}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")

    best_acc = max(strategies, key=lambda s: agg[s]["accuracy_rate"])
    best_speed = min(strategies, key=lambda s: agg[s]["avg_latency"])
    best_eff_s = max(strategies, key=lambda s: agg[s]["accuracy_rate"] / max(agg[s]["avg_latency"], 1))
    best_eff_val = agg[best_eff_s]["accuracy_rate"] / max(agg[best_eff_s]["avg_latency"], 1) * 100
    lines.append(f"- **Best Accuracy:** `{best_acc}` — {agg[best_acc]['accuracy_rate']:.0%} of queries fully correct "
                 f"({agg[best_acc]['n_pass']}/{agg[best_acc]['n_queries']}), "
                 f"{'zero' if best_acc in ('header', 'autosearch') else '1+'} LLM extraction calls")
    lines.append(f"- **Fastest:** `{best_speed}` — {agg[best_speed]['avg_latency']:.1f}s avg per query")
    lines.append(f"- **Best Efficiency:** `{best_eff_s}` — {best_eff_val:.1f} accuracy/latency ratio")
    ranked_strats = sorted(strategies, key=lambda s: (-agg[s]["accuracy_rate"], agg[s]["avg_latency"]))
    rec = ranked_strats[0] if ranked_strats else best_acc
    lines.append(f"- **Recommended Default:** `{rec}` — see analysis below")
    lines.append("")

    # Summary Table
    lines.append("## Strategy Comparison")
    lines.append("")
    lines.append("| Strategy | Description | Accuracy | Avg Latency | Avg Prompt Tokens | Errors |")
    lines.append("|----------|-------------|----------|-------------|-------------------|--------|")
    for s in strategies:
        a = agg[s]
        lines.append(
            f"| `{s}` | {STRATEGY_DESC.get(s, '')} | "
            f"{a['n_pass']}/{a['n_queries']} ({a['accuracy_rate']:.0%}) | "
            f"{a['avg_latency']:.1f}s | "
            f"{a['avg_prompt_tokens']:.0f} | "
            f"{a['errors']} |"
        )
    lines.append("")

    # Latency Breakdown
    lines.append("## Latency Breakdown")
    lines.append("")
    lines.append("| Strategy | Min | Avg | Max | Total |")
    lines.append("|----------|-----|-----|-----|-------|")
    for s in strategies:
        a = agg[s]
        lines.append(
            f"| `{s}` | {a['min_latency']:.1f}s | "
            f"{a['avg_latency']:.1f}s | {a['max_latency']:.1f}s | "
            f"{a['total_latency']:.0f}s |"
        )
    lines.append("")

    # Per-Query Results
    lines.append("## Per-Query Results")
    lines.append("")
    query_ids = sorted(set(r["query_id"] for r in results))
    for qid in query_ids:
        q_entries = [r for r in results if r["query_id"] == qid]
        if not q_entries:
            continue
        first = q_entries[0]
        lines.append(f"### Q{qid}: {first['query']}")
        lines.append(f"*Type: {first['query_type']}*")
        lines.append("")
        lines.append("| Strategy | Pass | Accuracy | Latency | Prompt Tokens | Response Preview |")
        lines.append("|----------|------|----------|---------|---------------|-----------------|")
        for s in strategies:
            entry = next((r for r in q_entries if r["strategy"] == s), None)
            if entry:
                status = "PASS" if entry["all_pass"] else "FAIL"
                preview = entry["response"][:80].replace("|", "/").replace("\n", " ")
                lines.append(
                    f"| `{s}` | {status} | {entry['accuracy']:.0%} | "
                    f"{entry['latency_seconds']:.1f}s | "
                    f"{entry['prompt_tokens']} | {preview}... |"
                )
            else:
                lines.append(f"| `{s}` | N/A | - | - | - | - |")
        lines.append("")

    # Analysis
    lines.append("## Analysis")
    lines.append("")
    lines.append("### Accuracy")
    lines.append("")
    for s in strategies:
        a = agg[s]
        failed_qs = [e for e in a["entries"] if not e["all_pass"]]
        if failed_qs:
            failed_ids = ", ".join(f"Q{e['query_id']}" for e in failed_qs)
            lines.append(f"- **{s}**: {a['accuracy_rate']:.0%} — failed on {failed_ids}")
        else:
            lines.append(f"- **{s}**: {a['accuracy_rate']:.0%} — all queries passed")
    lines.append("")

    lines.append("### Latency vs Accuracy Trade-off")
    lines.append("")
    lines.append("| Strategy | LLM Extraction Calls | Latency | Accuracy | Efficiency |")
    lines.append("|----------|---------------------|---------|----------|------------|")
    for s in strategies:
        a = agg[s]
        eff = a["accuracy_rate"] / max(a["avg_latency"], 1) * 100
        lines.append(
            f"| `{s}` | {LLM_CALLS.get(s, '?')} | {a['avg_latency']:.1f}s | "
            f"{a['accuracy_rate']:.0%} | {eff:.1f} |"
        )
    lines.append("")
    lines.append("*Efficiency = Accuracy% / Avg Latency (higher is better)*")
    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    ranked = sorted(strategies, key=lambda s: (-agg[s]["accuracy_rate"], agg[s]["avg_latency"]))
    best = ranked[0] if ranked else "header"
    fastest = min(strategies, key=lambda s: agg[s]["avg_latency"])
    lines.append(f"1. **Recommended default: `{best}`** — Highest accuracy ({agg[best]['accuracy_rate']:.0%}), "
                 f"{agg[best]['avg_latency']:.1f}s avg latency.")
    if fastest != best:
        lines.append(f"2. **Fastest: `{fastest}`** — {agg[fastest]['avg_latency']:.1f}s avg, "
                     f"{agg[fastest]['accuracy_rate']:.0%} accuracy.")
    lines.append("3. **Model capability is the bottleneck** — All strategies deliver similar prompt token counts. "
                 "Upgrading the model would likely improve all strategies proportionally.")
    lines.append("")

    lines.append("## Plots")
    lines.append("")
    lines.append(f"![Benchmark Plots]({plots_filename})")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by ctxpact benchmark suite*")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report saved to {output_path}")


# ---------------------------------------------------------------------------
# Multi-run comparison
# ---------------------------------------------------------------------------

def generate_comparison(runs: list[tuple[str, dict, dict, list[dict]]]):
    """Generate comparison report and plots for multiple labeled runs.

    Args:
        runs: list of (label, metadata, agg, results)
    """
    if len(runs) < 2:
        print("Need at least 2 runs for comparison. Skipping.")
        return

    # --- Comparison Plot ---
    labels = [r[0] for r in runs]
    all_strategies = []
    for _, _, agg, _ in runs:
        for s in STRATEGY_ORDER:
            if s in agg and s not in all_strategies:
                all_strategies.append(s)

    n_runs = len(runs)
    n_strats = len(all_strategies)
    x = np.arange(n_strats)
    width = 0.8 / n_runs
    run_colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("ctxpact Benchmark Comparison — Standard vs Jinja Tool Calling",
                 fontsize=14, fontweight="bold")

    # Accuracy comparison
    ax = axes[0]
    for i, (label, _, agg, _) in enumerate(runs):
        vals = [agg[s]["accuracy_rate"] * 100 if s in agg else 0 for s in all_strategies]
        offset = (i - n_runs / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=run_colors[i % len(run_colors)], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Accuracy Rate (%)")
    ax.set_title("Accuracy by Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(all_strategies, rotation=30, ha="right")
    ax.set_ylim(0, 110)
    ax.legend()

    # Latency comparison
    ax = axes[1]
    for i, (label, _, agg, _) in enumerate(runs):
        vals = [agg[s]["avg_latency"] if s in agg else 0 for s in all_strategies]
        offset = (i - n_runs / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=run_colors[i % len(run_colors)], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.0f}s", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Avg Latency (seconds)")
    ax.set_title("Latency by Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(all_strategies, rotation=30, ha="right")
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(BENCH_DIR, "benchmark_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plots saved to {plot_path}")

    # --- Comparison Markdown ---
    lines = []
    lines.append("# ctxpact Benchmark Comparison Report")
    lines.append("")
    lines.append(f"**Runs Compared:** {', '.join(labels)}")
    lines.append(f"**Generated:** {runs[0][1].get('timestamp', 'N/A')}")
    lines.append("")

    # Summary table: strategy × run accuracy
    lines.append("## Accuracy Comparison")
    lines.append("")
    header = "| Strategy | " + " | ".join(labels) + " | Delta |"
    sep = "|----------|" + "|".join(["--------"] * len(labels)) + "|-------|"
    lines.append(header)
    lines.append(sep)
    for s in all_strategies:
        row = f"| `{s}` |"
        rates = []
        for _, _, agg, _ in runs:
            if s in agg:
                rate = agg[s]["accuracy_rate"]
                rates.append(rate)
                row += f" {agg[s]['n_pass']}/{agg[s]['n_queries']} ({rate:.0%}) |"
            else:
                rates.append(0)
                row += " N/A |"
        if len(rates) >= 2:
            delta = (rates[-1] - rates[0]) * 100
            sign = "+" if delta > 0 else ""
            row += f" {sign}{delta:.0f}pp |"
        else:
            row += " - |"
        lines.append(row)
    lines.append("")

    # Latency table
    lines.append("## Latency Comparison")
    lines.append("")
    header = "| Strategy | " + " | ".join(labels) + " | Delta |"
    lines.append(header)
    lines.append(sep)
    for s in all_strategies:
        row = f"| `{s}` |"
        lats = []
        for _, _, agg, _ in runs:
            if s in agg:
                lat = agg[s]["avg_latency"]
                lats.append(lat)
                row += f" {lat:.1f}s |"
            else:
                lats.append(0)
                row += " N/A |"
        if len(lats) >= 2 and lats[0] > 0:
            delta = lats[-1] - lats[0]
            pct = delta / lats[0] * 100
            sign = "+" if delta > 0 else ""
            row += f" {sign}{delta:.1f}s ({sign}{pct:.0f}%) |"
        else:
            row += " - |"
        lines.append(row)
    lines.append("")

    lines.append("## Plots")
    lines.append("")
    lines.append("![Comparison](benchmark_comparison.png)")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by ctxpact benchmark suite*")

    report_path = os.path.join(BENCH_DIR, "benchmark_comparison.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Comparison report saved to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_single(label: str, path: str):
    """Process a single result file."""
    data = load_results(path)
    metadata = data["metadata"]
    results = data["results"]
    if not results:
        print(f"No results in {path}")
        return None

    agg = aggregate_by_strategy(results)
    print(f"[{label}] Loaded {len(results)} results across {len(agg)} strategies")

    plots_file = f"benchmark_plots_{label}.png"
    generate_plots(agg, results, os.path.join(BENCH_DIR, plots_file),
                   model_name=metadata.get("model", ""), label=label)
    generate_markdown_report(metadata, agg, results,
                             os.path.join(BENCH_DIR, f"benchmark_report_{label}.md"),
                             label=label, plots_filename=plots_file)
    return (label, metadata, agg, results)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark reports")
    parser.add_argument("--label", default=None,
                        help="Process a specific label only")
    parser.add_argument("--compare", action="store_true",
                        help="Only generate comparison (skip individual reports)")
    args = parser.parse_args()

    result_files = find_result_files()
    if not result_files:
        print("No benchmark result files found in ~/ctxpact-bench/")
        return

    if args.label:
        result_files = [(l, p) for l, p in result_files if l == args.label]
        if not result_files:
            print(f"No results found for label '{args.label}'")
            return

    runs = []
    for label, path in result_files:
        if not args.compare:
            run = process_single(label, path)
            if run:
                runs.append(run)
        else:
            data = load_results(path)
            agg = aggregate_by_strategy(data["results"])
            runs.append((label, data["metadata"], agg, data["results"]))

    # Generate comparison if multiple runs
    if len(runs) >= 2:
        generate_comparison(runs)


if __name__ == "__main__":
    main()
