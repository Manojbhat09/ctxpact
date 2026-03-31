#!/usr/bin/env python3
"""Benchmark all ctxpact extraction strategies against Frankenstein.

Restarts ctxpact with each strategy, runs 8 test queries, and collects metrics.
Saves results to bench/results/benchmark_results_{label}.json.

Usage:
  python benchmark_strategies.py                          # default label="standard"
  python benchmark_strategies.py --label jinja-toolcall   # with native tool calling
  python benchmark_strategies.py --strategies header rlm_v2 toolcall  # subset
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

CTXPACT_URL = "http://localhost:8000"
CTXPACT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
BENCH_DIR = os.path.join(CTXPACT_DIR, "bench")
BOOK_PATH = os.environ.get("BENCH_BOOK", os.path.join(BENCH_DIR, "data", "frankenstein.txt"))
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.5-9B-Q8_0.gguf")

ALL_STRATEGIES = ["header", "autosearch", "rlm", "rlm_v2", "rlm_v3", "toolcall", "chunking",
                  "embed", "compress", "adaptive", "icl", "rlm_v4", "rlm_v5", "rlm_v6", "agentic", "subagent"]

# Each query: (question, type, list of keyword groups)
# A keyword group is a list of alternatives — at least one must match.
# All groups must pass for the query to be "correct".
QUERIES = [
    {
        "id": 1,
        "query": "How many chapters are there in this book? List all chapter numbers.",
        "type": "structural",
        "keyword_groups": [["24"]],
        "max_tokens": 1024,
    },
    {
        "id": 2,
        "query": "Who is Justine Moritz and what happens to her?",
        "type": "character",
        "keyword_groups": [
            ["justine"],
            ["trial", "executed", "hanged", "condemned", "guilty"],
        ],
        "max_tokens": 512,
    },
    {
        "id": 3,
        "query": "What does the creature ask Frankenstein to create for him?",
        "type": "plot",
        "keyword_groups": [
            ["companion", "mate", "female", "wife", "partner"],
        ],
        "max_tokens": 512,
    },
    {
        "id": 4,
        "query": "Where does Clerval get murdered?",
        "type": "detail",
        "keyword_groups": [
            ["clerval"],
            ["ireland", "irish"],
        ],
        "max_tokens": 512,
    },
    {
        "id": 5,
        "query": "What is the name of the ship captain who rescues Frankenstein from the ice?",
        "type": "character",
        "keyword_groups": [
            ["walton"],
        ],
        "max_tokens": 512,
    },
    {
        "id": 6,
        "query": "How does the novel end? What happens to the creature?",
        "type": "plot",
        "keyword_groups": [
            ["arctic", "ice", "north", "pole"],
            ["death", "dies", "fire", "funeral", "perish", "destroy"],
        ],
        "max_tokens": 512,
    },
    {
        "id": 7,
        "query": "What does Victor study at the university of Ingolstadt?",
        "type": "detail",
        "keyword_groups": [
            ["chemistry", "natural philosophy", "science", "philosophy", "anatomy"],
        ],
        "max_tokens": 512,
    },
    {
        "id": 8,
        "query": "Who teaches the creature to read and speak? How does it learn?",
        "type": "detail",
        "keyword_groups": [
            ["felix", "de lacey", "safie", "delacey", "cottage"],
        ],
        "max_tokens": 512,
    },
]


def kill_ctxpact():
    """Kill any running ctxpact processes."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "ctxpact.server"],
            capture_output=True, text=True
        )
        for pid in result.stdout.strip().split("\n"):
            if pid:
                os.kill(int(pid), signal.SIGTERM)
    except Exception:
        pass
    time.sleep(2)


def start_ctxpact(strategy: str, log_path: str) -> subprocess.Popen:
    """Start ctxpact with a specific strategy."""
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "ctxpact.server",
                "--config", "config.yaml",
                "--local",
                "--strategy", strategy,
            ],
            cwd=CTXPACT_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    return proc


def wait_for_health(timeout: int = 30) -> bool:
    """Wait for ctxpact health endpoint to respond."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{CTXPACT_URL}/health")
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read().decode())
            if data.get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def api_call(data: dict, timeout: int = 600) -> dict:
    """Make a chat completion API call."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{CTXPACT_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read().decode())


def check_keywords(response: str, keyword_groups: list[list[str]]) -> dict:
    """Check if response contains required keywords.

    Returns dict with group results and overall pass/fail.
    """
    response_lower = response.lower()
    group_results = []
    for group in keyword_groups:
        found = [kw for kw in group if kw.lower() in response_lower]
        group_results.append({
            "alternatives": group,
            "found": found,
            "pass": len(found) > 0,
        })
    return {
        "groups": group_results,
        "all_pass": all(g["pass"] for g in group_results),
        "score": sum(1 for g in group_results if g["pass"]) / len(group_results),
    }


def run_benchmark(label: str = "standard", strategies: list[str] | None = None):
    """Run the full benchmark.

    Args:
        label: Run label (e.g., "standard", "jinja-toolcall") — used in output filenames.
        strategies: List of strategies to test. None = all.
    """
    strategies = strategies or ALL_STRATEGIES

    # Load book
    with open(BOOK_PATH) as f:
        book_text = f.read()

    print(f"Run label: {label}")
    print(f"Book: {len(book_text):,} chars (~{len(book_text) // 4:,} est tokens)")
    print(f"Strategies: {strategies}")
    print(f"Queries: {len(QUERIES)}")
    print("=" * 70)

    all_results = {
        "metadata": {
            "book": BOOK_PATH,
            "book_chars": len(book_text),
            "book_est_tokens": len(book_text) // 4,
            "model": MODEL,
            "label": label,
            "num_strategies": len(strategies),
            "num_queries": len(QUERIES),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": [],
    }

    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"STRATEGY: {strategy}")
        print(f"{'='*70}")

        # Kill existing, start fresh
        kill_ctxpact()
        log_dir = os.path.join(BENCH_DIR, "logs", label)
        os.makedirs(log_dir, exist_ok=True)
        log_path = f"{log_dir}/ctxpact_bench_{strategy}.log"
        proc = start_ctxpact(strategy, log_path)

        if not wait_for_health():
            print(f"  ERROR: ctxpact failed to start with strategy={strategy}")
            proc.terminate()
            continue

        print(f"  ctxpact started (PID {proc.pid})")

        strategy_results = []
        for q in QUERIES:
            query = q["query"]
            full_input = f"{query}\n\nHere is the full text of the book:\n\n{book_text}"

            print(f"\n  Q{q['id']}: {query}")
            start = time.time()

            try:
                result = api_call({
                    "model": MODEL,
                    "messages": [{"role": "user", "content": full_input}],
                    "max_tokens": q["max_tokens"],
                    "temperature": 0.1,
                })
                elapsed = time.time() - start
                response = result["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
                kw_check = check_keywords(response, q["keyword_groups"])

                entry = {
                    "strategy": strategy,
                    "query_id": q["id"],
                    "query": query,
                    "query_type": q["type"],
                    "response": response,
                    "response_length": len(response),
                    "latency_seconds": round(elapsed, 1),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "keyword_check": kw_check,
                    "accuracy": kw_check["score"],
                    "all_pass": kw_check["all_pass"],
                    "error": None,
                }

                status = "PASS" if kw_check["all_pass"] else "FAIL"
                print(
                    f"     [{status}] {elapsed:.1f}s, "
                    f"{usage.get('prompt_tokens', 0)} prompt tokens, "
                    f"accuracy={kw_check['score']:.0%}"
                )
                print(f"     Response: {response[:150]}...")

            except Exception as e:
                elapsed = time.time() - start
                entry = {
                    "strategy": strategy,
                    "query_id": q["id"],
                    "query": query,
                    "query_type": q["type"],
                    "response": "",
                    "response_length": 0,
                    "latency_seconds": round(elapsed, 1),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "keyword_check": {"groups": [], "all_pass": False, "score": 0},
                    "accuracy": 0,
                    "all_pass": False,
                    "error": str(e),
                }
                print(f"     [ERROR] {elapsed:.1f}s: {e}")

            strategy_results.append(entry)
            all_results["results"].append(entry)

        # Strategy summary
        passes = sum(1 for r in strategy_results if r["all_pass"])
        avg_latency = sum(r["latency_seconds"] for r in strategy_results) / len(strategy_results)
        avg_accuracy = sum(r["accuracy"] for r in strategy_results) / len(strategy_results)
        avg_prompt = sum(r["prompt_tokens"] for r in strategy_results) / len(strategy_results)
        print(f"\n  --- {strategy} Summary ---")
        print(f"  Accuracy: {passes}/{len(strategy_results)} queries passed ({avg_accuracy:.0%} avg)")
        print(f"  Avg Latency: {avg_latency:.1f}s")
        print(f"  Avg Prompt Tokens: {avg_prompt:.0f}")

        # Stop ctxpact
        proc.terminate()
        proc.wait()

    # Save results
    results_dir = os.path.join(BENCH_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"benchmark_results_{label}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*70}")
    print(f"Results saved to {output_path}")
    print(f"Strategy logs at {log_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ctxpact extraction strategies")
    parser.add_argument("--label", default="standard",
                        help="Run label for output files (e.g., 'standard', 'jinja-toolcall')")
    parser.add_argument("--strategies", nargs="+", choices=ALL_STRATEGIES, default=None,
                        help="Strategies to test (default: all)")
    args = parser.parse_args()
    run_benchmark(label=args.label, strategies=args.strategies)
