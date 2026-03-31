#!/usr/bin/env python3
"""
Comprehensive benchmark summary for ctxpact-bench.

Reads all benchmark_results_*.json files and produces a grouped report
by model, strategy, and version iteration.
"""

import json
import glob
import os
from collections import defaultdict


BENCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_all_results(bench_dir):
    """Load all benchmark JSON files and return a list of (filename, metadata, results)."""
    pattern = os.path.join(bench_dir, "benchmark_results*.json")
    files = sorted(glob.glob(pattern))
    
    all_data = []
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  WARNING: Could not read {fname}: {e}")
            continue
        
        meta = data.get("metadata", {})
        results = data.get("results", [])
        all_data.append((fname, meta, results))
    
    return all_data


def classify_label(label):
    """Classify a label into a broader category for grouping."""
    if not label:
        return "unknown"
    label_lower = label.lower()
    if "readagent-v25" in label_lower:
        return "readagent-v25"
    elif "readagent" in label_lower:
        return "readagent-older"
    elif "agentic" in label_lower:
        return "agentic"
    elif "subagent" in label_lower:
        return "subagent"
    elif "rlm" in label_lower:
        return "rlm"
    elif "qwen35" in label_lower:
        return "qwen35"
    else:
        return "other"


def normalize_model(model_str):
    """Normalize model names for grouping."""
    if not model_str:
        return "unknown"
    m = model_str.strip()
    if "LFM2" in m or "lfm2" in m.lower():
        return "LFM2-8B-A1B-Q8_0"
    if "Qwen3.5" in m or "qwen3.5" in m.lower() or "qwen35" in m.lower():
        return "Qwen3.5-9B-Q8_0"
    return m


def build_summary(all_data):
    """
    Build a nested summary:
      model -> strategy -> list of run records
    Each run record = { label, fname, queries: [{query_id, all_pass, accuracy, latency}] }
    """
    summary = defaultdict(lambda: defaultdict(list))
    
    for fname, meta, results in all_data:
        model = normalize_model(meta.get("model", ""))
        label = meta.get("label", fname)
        timestamp = meta.get("timestamp", "")
        
        # Group results by strategy within this file
        strat_groups = defaultdict(list)
        for r in results:
            strat = r.get("strategy", "unknown")
            strat_groups[strat].append(r)
        
        for strat, queries in strat_groups.items():
            run = {
                "label": label,
                "fname": fname,
                "timestamp": timestamp,
                "queries": []
            }
            for q in queries:
                run["queries"].append({
                    "query_id": q.get("query_id"),
                    "all_pass": q.get("all_pass", False),
                    "accuracy": q.get("accuracy", 0.0),
                    "latency_seconds": q.get("latency_seconds", 0.0),
                    "error": q.get("error"),
                })
            summary[model][strat].append(run)
    
    return summary


def print_model_strategy_table(summary):
    """Print a summary table for each model showing per-strategy aggregate stats."""
    print("=" * 120)
    print("BENCHMARK SUMMARY: Model x Strategy Aggregates (all files)")
    print("=" * 120)
    
    for model in sorted(summary.keys()):
        strats = summary[model]
        print(f"\n{'~' * 120}")
        print(f"MODEL: {model}")
        print(f"{'~' * 120}")
        print(f"  {'Strategy':<15} {'Runs':>5} {'Queries':>8} {'Pass':>6} {'Pass%':>7} "
              f"{'AvgAcc':>7} {'AvgLat':>8} {'MinLat':>8} {'MaxLat':>8}")
        print(f"  {'-'*15} {'-'*5} {'-'*8} {'-'*6} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
        
        for strat in sorted(strats.keys()):
            runs = strats[strat]
            total_queries = 0
            total_pass = 0
            total_acc = 0.0
            all_latencies = []
            
            for run in runs:
                for q in run["queries"]:
                    total_queries += 1
                    if q["all_pass"]:
                        total_pass += 1
                    total_acc += (q["accuracy"] or 0.0)
                    lat = q["latency_seconds"]
                    if lat and lat > 0:
                        all_latencies.append(lat)
            
            pass_pct = (total_pass / total_queries * 100) if total_queries else 0
            avg_acc = (total_acc / total_queries) if total_queries else 0
            avg_lat = (sum(all_latencies) / len(all_latencies)) if all_latencies else 0
            min_lat = min(all_latencies) if all_latencies else 0
            max_lat = max(all_latencies) if all_latencies else 0
            
            print(f"  {strat:<15} {len(runs):>5} {total_queries:>8} {total_pass:>6} "
                  f"{pass_pct:>6.1f}% {avg_acc:>6.2f}  {avg_lat:>7.1f}s {min_lat:>7.1f}s {max_lat:>7.1f}s")


def print_per_query_breakdown(summary, strategies=None, models=None):
    """Print per-query pass rate for selected strategies/models."""
    print("\n" + "=" * 120)
    print("PER-QUERY PASS RATES (by model + strategy)")
    print("=" * 120)
    
    for model in sorted(summary.keys()):
        if models and model not in models:
            continue
        strats = summary[model]
        for strat in sorted(strats.keys()):
            if strategies and strat not in strategies:
                continue
            runs = strats[strat]
            
            query_stats = defaultdict(lambda: {"pass": 0, "total": 0, "acc_sum": 0.0})
            for run in runs:
                for q in run["queries"]:
                    qid = q["query_id"]
                    query_stats[qid]["total"] += 1
                    if q["all_pass"]:
                        query_stats[qid]["pass"] += 1
                    query_stats[qid]["acc_sum"] += (q["accuracy"] or 0.0)
            
            if not query_stats:
                continue
            
            total_pass = sum(v["pass"] for v in query_stats.values())
            total_q = sum(v["total"] for v in query_stats.values())
            
            print(f"\n  {model} / {strat}  ({len(runs)} runs, {total_pass}/{total_q} total pass)")
            print(f"    {'QID':>4} {'Pass':>6} {'Total':>6} {'Rate':>7} {'AvgAcc':>7}")
            print(f"    {'-'*4} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")
            for qid in sorted(query_stats.keys()):
                s = query_stats[qid]
                rate = s["pass"] / s["total"] * 100 if s["total"] else 0
                avg_a = s["acc_sum"] / s["total"] if s["total"] else 0
                print(f"    {qid:>4} {s['pass']:>6} {s['total']:>6} {rate:>6.1f}% {avg_a*100:>6.1f}%")


def print_readagent_v25_detail(all_data):
    """Detailed view of readagent-v25* runs (latest iterations)."""
    print("\n" + "=" * 120)
    print("READAGENT v25 ITERATION DETAIL (latest readagent runs)")
    print("=" * 120)
    
    v25_runs = []
    for fname, meta, results in all_data:
        label = meta.get("label", "")
        if "readagent-v25" not in label.lower() and "readagent-v25" not in fname.lower():
            continue
        model = normalize_model(meta.get("model", ""))
        
        passes = sum(1 for r in results if r.get("all_pass"))
        total = len(results)
        avg_acc = sum(r.get("accuracy", 0) or 0 for r in results) / total if total else 0
        lats = [r.get("latency_seconds", 0) for r in results if r.get("latency_seconds")]
        avg_lat = sum(lats) / len(lats) if lats else 0
        
        q_pattern = ""
        for r in sorted(results, key=lambda x: x.get("query_id", 0)):
            q_pattern += "P" if r.get("all_pass") else "."
        
        v25_runs.append({
            "label": label,
            "model": model,
            "passes": passes,
            "total": total,
            "avg_acc": avg_acc,
            "avg_lat": avg_lat,
            "q_pattern": q_pattern,
            "timestamp": meta.get("timestamp", ""),
        })
    
    v25_runs.sort(key=lambda x: (x["model"], x["label"]))
    
    current_model = None
    print(f"\n  {'Label':<35} {'Model':<22} {'Pass':>4}/{'':<4} {'Rate':>6} "
          f"{'AvgAcc':>7} {'AvgLat':>8} {'Q1-Q8':>10}  {'Timestamp'}")
    print(f"  {'-'*35} {'-'*22} {'-'*9} {'-'*6} {'-'*7} {'-'*8} {'-'*10}  {'-'*19}")
    
    for run in v25_runs:
        if run["model"] != current_model:
            current_model = run["model"]
            print(f"\n  --- {current_model} ---")
        
        rate = run["passes"] / run["total"] * 100 if run["total"] else 0
        print(f"  {run['label']:<35} {run['model']:<22} {run['passes']:>4}/{run['total']:<4} "
              f"{rate:>5.0f}% {run['avg_acc']*100:>6.1f}% {run['avg_lat']:>7.1f}s "
              f"{run['q_pattern']:>10}  {run['timestamp']}")


def print_qwen35_detail(all_data):
    """Detailed view of all Qwen3.5 runs."""
    print("\n" + "=" * 120)
    print("QWEN3.5-9B RUNS DETAIL")
    print("=" * 120)
    
    qwen_runs = []
    for fname, meta, results in all_data:
        model = normalize_model(meta.get("model", ""))
        if "Qwen" not in model:
            continue
        label = meta.get("label", fname)
        
        strat_groups = defaultdict(list)
        for r in results:
            strat_groups[r.get("strategy", "unknown")].append(r)
        
        for strat, queries in strat_groups.items():
            passes = sum(1 for q in queries if q.get("all_pass"))
            total = len(queries)
            avg_acc = sum(q.get("accuracy", 0) or 0 for q in queries) / total if total else 0
            lats = [q.get("latency_seconds", 0) for q in queries if q.get("latency_seconds")]
            avg_lat = sum(lats) / len(lats) if lats else 0
            
            q_pattern = ""
            for q in sorted(queries, key=lambda x: x.get("query_id", 0)):
                q_pattern += "P" if q.get("all_pass") else "."
            
            qwen_runs.append({
                "label": label,
                "strategy": strat,
                "passes": passes,
                "total": total,
                "avg_acc": avg_acc,
                "avg_lat": avg_lat,
                "q_pattern": q_pattern,
                "timestamp": meta.get("timestamp", ""),
            })
    
    qwen_runs.sort(key=lambda x: (x["strategy"], x["label"]))
    
    print(f"\n  {'Label':<35} {'Strategy':<15} {'Pass':>4}/{'':<4} {'Rate':>6} "
          f"{'AvgAcc':>7} {'AvgLat':>8} {'Q1-Q8':>10}  {'Timestamp'}")
    print(f"  {'-'*35} {'-'*15} {'-'*9} {'-'*6} {'-'*7} {'-'*8} {'-'*10}  {'-'*19}")
    
    for run in qwen_runs:
        rate = run["passes"] / run["total"] * 100 if run["total"] else 0
        print(f"  {run['label']:<35} {run['strategy']:<15} {run['passes']:>4}/{run['total']:<4} "
              f"{rate:>5.0f}% {run['avg_acc']*100:>6.1f}% {run['avg_lat']:>7.1f}s "
              f"{run['q_pattern']:>10}  {run['timestamp']}")


def print_best_runs(summary, top_n=15):
    """Print the top N runs by pass rate across all models/strategies."""
    print("\n" + "=" * 120)
    print(f"TOP {top_n} BEST INDIVIDUAL RUNS (by pass count, then avg accuracy)")
    print("=" * 120)
    
    all_runs = []
    for model, strats in summary.items():
        for strat, runs in strats.items():
            for run in runs:
                passes = sum(1 for q in run["queries"] if q["all_pass"])
                total = len(run["queries"])
                avg_acc = sum(q["accuracy"] or 0 for q in run["queries"]) / total if total else 0
                lats = [q["latency_seconds"] for q in run["queries"] if q["latency_seconds"] and q["latency_seconds"] > 0]
                avg_lat = sum(lats) / len(lats) if lats else 0
                
                all_runs.append({
                    "model": model,
                    "strategy": strat,
                    "label": run["label"],
                    "passes": passes,
                    "total": total,
                    "avg_acc": avg_acc,
                    "avg_lat": avg_lat,
                    "timestamp": run["timestamp"],
                })
    
    all_runs.sort(key=lambda x: (-x["passes"], -x["avg_acc"], x["avg_lat"]))
    
    print(f"\n  {'#':>3} {'Label':<35} {'Model':<22} {'Strategy':<15} {'Pass':>4}/{'':<4} {'Rate':>6} "
          f"{'AvgAcc':>7} {'AvgLat':>8}")
    print(f"  {'-'*3} {'-'*35} {'-'*22} {'-'*15} {'-'*9} {'-'*6} {'-'*7} {'-'*8}")
    
    for i, run in enumerate(all_runs[:top_n], 1):
        rate = run["passes"] / run["total"] * 100 if run["total"] else 0
        print(f"  {i:>3} {run['label']:<35} {run['model']:<22} {run['strategy']:<15} "
              f"{run['passes']:>4}/{run['total']:<4} {rate:>5.0f}% {run['avg_acc']*100:>6.1f}% {run['avg_lat']:>7.1f}s")


def print_model_comparison(summary):
    """Side-by-side comparison of models for the same strategies."""
    print("\n" + "=" * 120)
    print("MODEL COMPARISON: LFM2 vs Qwen3.5 (same strategies)")
    print("=" * 120)
    
    models = sorted(summary.keys())
    all_strats = set()
    for model in models:
        all_strats.update(summary[model].keys())
    
    # Build a table
    header = f"  {'Strategy':<15}"
    for model in models:
        short = model[:25]
        header += f" | {short:^32}"
    print(f"\n{header}")
    
    sub = f"  {'':15}"
    for _ in models:
        sub += f" | {'Runs':>5} {'Pass%':>7} {'AvgAcc':>7} {'AvgLat':>8}"
    print(sub)
    
    sep = f"  {'-'*15}"
    for _ in models:
        sep += f" | {'-'*5} {'-'*7} {'-'*7} {'-'*8}"
    print(sep)
    
    for strat in sorted(all_strats):
        line = f"  {strat:<15}"
        for model in models:
            runs = summary[model].get(strat, [])
            if not runs:
                line += f" | {'--':>5} {'--':>7} {'--':>7} {'--':>8}"
                continue
            total_q = 0
            total_p = 0
            total_acc = 0.0
            all_lat = []
            for run in runs:
                for q in run["queries"]:
                    total_q += 1
                    if q["all_pass"]:
                        total_p += 1
                    total_acc += (q["accuracy"] or 0.0)
                    if q["latency_seconds"] and q["latency_seconds"] > 0:
                        all_lat.append(q["latency_seconds"])
            
            pp = total_p / total_q * 100 if total_q else 0
            aa = total_acc / total_q * 100 if total_q else 0
            al = sum(all_lat) / len(all_lat) if all_lat else 0
            
            line += f" | {len(runs):>5} {pp:>6.1f}% {aa:>6.1f}% {al:>7.1f}s"
        print(line)


def print_file_inventory(all_data):
    """Summary of all files loaded."""
    print("=" * 120)
    print(f"FILE INVENTORY: {len(all_data)} files loaded")
    print("=" * 120)
    
    category_counts = defaultdict(int)
    model_counts = defaultdict(int)
    
    for fname, meta, results in all_data:
        label = meta.get("label", fname)
        cat = classify_label(label)
        category_counts[cat] += 1
        model = normalize_model(meta.get("model", ""))
        model_counts[model] += 1
    
    print("\n  By category:")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:<20} {cnt:>4} files")
    
    print("\n  By model:")
    for model, cnt in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"    {model:<30} {cnt:>4} files")


def main():
    print("Loading benchmark results from:", BENCH_DIR)
    all_data = load_all_results(BENCH_DIR)
    print(f"Loaded {len(all_data)} result files.\n")
    
    summary = build_summary(all_data)
    
    # 1. File inventory
    print_file_inventory(all_data)
    
    # 2. Model x Strategy aggregates
    print_model_strategy_table(summary)
    
    # 3. Model comparison
    print_model_comparison(summary)
    
    # 4. Top runs
    print_best_runs(summary, top_n=15)
    
    # 5. readagent-v25 detail
    print_readagent_v25_detail(all_data)
    
    # 6. Qwen3.5 detail
    print_qwen35_detail(all_data)
    
    # 7. Per-query breakdown for key strategies
    print_per_query_breakdown(summary, 
                              strategies=["readagent", "agentic", "subagent", "embed", "header"],
                              models=None)


if __name__ == "__main__":
    main()
