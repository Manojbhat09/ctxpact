#!/usr/bin/env python3
"""Benchmark ctxpact agentic strategy on LoCoMo-MC10 dataset.

LoCoMo-MC10: Multi-session conversation QA with 10-choice multiple choice.
1,986 questions across 5 types: single_hop, multi_hop, temporal_reasoning,
open_domain, adversarial.

Usage:
  python benchmark_locomo.py                      # 20-question sample
  python benchmark_locomo.py --n 100              # 100 questions
  python benchmark_locomo.py --n 0                # all 1,986 questions
  python benchmark_locomo.py --types multi_hop temporal_reasoning  # specific types
"""

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

CTXPACT_URL = "http://localhost:8000"
CTXPACT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.environ.get(
    "LOCOMO_DATA",
    os.path.join(BENCH_DIR, "data", "locomo-mc10", "data", "locomo_mc10.json"),
)
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.5-9B-Q8_0.gguf")
STRATEGY = os.environ.get("BENCH_STRATEGY", "readagent")


def load_dataset(
    path: str,
    n: int = 20,
    types: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load LoCoMo-MC10 entries, optionally sampling a subset."""
    with open(path) as f:
        entries = [json.loads(line) for line in f]

    if types:
        entries = [e for e in entries if e["question_type"] in types]

    if n > 0 and n < len(entries):
        # Stratified sample by question type
        from collections import defaultdict

        by_type = defaultdict(list)
        for e in entries:
            by_type[e["question_type"]].append(e)

        rng = random.Random(seed)
        sampled = []
        per_type = max(1, n // len(by_type))
        for qtype, items in sorted(by_type.items()):
            k = min(per_type, len(items))
            sampled.extend(rng.sample(items, k))

        # Fill remaining quota
        remaining = n - len(sampled)
        if remaining > 0:
            pool = [e for e in entries if e not in sampled]
            sampled.extend(rng.sample(pool, min(remaining, len(pool))))

        entries = sampled[:n]

    return entries


def format_conversation(entry: dict) -> str:
    """Format conversation sessions into a single text block."""
    parts = []
    for i, session in enumerate(entry["haystack_sessions"]):
        dt = entry["haystack_session_datetimes"][i] if i < len(
            entry["haystack_session_datetimes"]
        ) else ""
        parts.append(f"--- Session {i + 1} ({dt}) ---")
        for msg in session:
            parts.append(msg["content"])
    return "\n\n".join(parts)


def format_question(entry: dict) -> str:
    """Format question + choices for the model."""
    lines = [
        f"Question: {entry['question']}",
        "",
        "Choose the correct answer (respond with ONLY the choice number):",
    ]
    for i, choice in enumerate(entry["choices"]):
        lines.append(f"{i + 1}. {choice}")
    return "\n".join(lines)


def parse_choice(response: str, num_choices: int = 10) -> int | None:
    """Extract the chosen number from model response.

    Returns 0-indexed choice or None if unparseable.
    """
    text = response.strip()

    # Try to find a standalone number 1-10
    # First line often has just the number
    first_line = text.split("\n")[0].strip().rstrip(".")
    m = re.match(r"^(\d{1,2})\.?\s*$", first_line)
    if m:
        num = int(m.group(1))
        if 1 <= num <= num_choices:
            return num - 1

    # Look for "The answer is X" or "correct answer is X"
    m = re.search(r"(?:answer|choice)\s+(?:is\s+)?(\d{1,2})", text, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        if 1 <= num <= num_choices:
            return num - 1

    # Look for any number 1-10 at the start
    m = re.match(r"(\d{1,2})", text)
    if m:
        num = int(m.group(1))
        if 1 <= num <= num_choices:
            return num - 1

    # Try matching the answer text in the response
    return None


def kill_ctxpact():
    """Kill any running ctxpact server."""
    subprocess.run(
        ["pkill", "-f", "ctxpact.server"],
        capture_output=True,
    )
    time.sleep(1)


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
    """Wait for ctxpact health endpoint."""
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


def run_benchmark(
    n: int = 20,
    types: list[str] | None = None,
    label: str = "locomo-agentic",
    strategy: str = STRATEGY,
    model: str = MODEL,
):
    entries = load_dataset(DATASET_PATH, n=n, types=types)

    print(f"LoCoMo-MC10 Benchmark")
    print(f"Label: {label}")
    print(f"Model: {model}")
    print(f"Questions: {len(entries)}")
    from collections import Counter
    type_counts = Counter(e["question_type"] for e in entries)
    print(f"Types: {dict(type_counts)}")
    print(f"Strategy: {strategy}")
    print("=" * 70)

    # Start ctxpact
    kill_ctxpact()
    log_dir = os.path.join(BENCH_DIR, "logs", label)
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/ctxpact_bench_{strategy}.log"
    proc = start_ctxpact(strategy, log_path)

    if not wait_for_health():
        print("ERROR: ctxpact failed to start")
        proc.terminate()
        return

    print(f"ctxpact started (PID {proc.pid})")

    results = []
    correct = 0
    total = 0
    type_stats: dict[str, dict] = {}

    for idx, entry in enumerate(entries):
        qid = entry["question_id"]
        qtype = entry["question_type"]
        question_text = entry["question"]
        correct_idx = entry["correct_choice_index"]
        answer = entry["answer"]

        # Build the prompt
        conv_text = format_conversation(entry)
        q_text = format_question(entry)
        full_input = (
            f"{q_text}\n\n"
            f"Here is the full conversation history:\n\n{conv_text}"
        )

        conv_chars = len(conv_text)
        conv_tokens_est = conv_chars // 4

        print(f"\n  [{idx + 1}/{len(entries)}] {qid} ({qtype})")
        print(f"    Q: {question_text[:80]}...")
        print(f"    Context: ~{conv_tokens_est:,} tokens")

        start = time.time()
        try:
            result = api_call({
                "model": model,
                "messages": [{"role": "user", "content": full_input}],
                "max_tokens": 256,
                "temperature": 0.1,
            })
            elapsed = time.time() - start
            response = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})

            predicted = parse_choice(response, entry["num_choices"])
            is_correct = predicted == correct_idx

            if is_correct:
                correct += 1
            total += 1

            # Track per-type stats
            if qtype not in type_stats:
                type_stats[qtype] = {"correct": 0, "total": 0}
            type_stats[qtype]["total"] += 1
            if is_correct:
                type_stats[qtype]["correct"] += 1

            status = "CORRECT" if is_correct else "WRONG"
            pred_str = (
                f"#{predicted + 1} ({entry['choices'][predicted]})"
                if predicted is not None
                else "UNPARSED"
            )
            print(
                f"    [{status}] {elapsed:.1f}s | "
                f"Predicted: {pred_str} | "
                f"Correct: #{correct_idx + 1} ({answer})"
            )
            if not is_correct:
                print(f"    Response: {response[:150]}...")

            results.append({
                "question_id": qid,
                "question_type": qtype,
                "question": question_text,
                "answer": answer,
                "correct_choice_index": correct_idx,
                "predicted_choice_index": predicted,
                "is_correct": is_correct,
                "response": response[:500],
                "latency_seconds": elapsed,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "conv_chars": conv_chars,
                "conv_tokens_est": conv_tokens_est,
                "error": None,
            })

        except Exception as e:
            elapsed = time.time() - start
            total += 1
            print(f"    [ERROR] {elapsed:.1f}s: {e}")
            results.append({
                "question_id": qid,
                "question_type": qtype,
                "question": question_text,
                "answer": answer,
                "correct_choice_index": correct_idx,
                "predicted_choice_index": None,
                "is_correct": False,
                "response": "",
                "latency_seconds": elapsed,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "conv_chars": len(conv_text),
                "conv_tokens_est": len(conv_text) // 4,
                "error": str(e),
            })

    # Summary
    print(f"\n{'=' * 70}")
    print(f"LoCoMo-MC10 — {strategy} Summary")
    print(f"{'=' * 70}")
    print(f"Overall: {correct}/{total} correct ({correct / total * 100:.1f}%)")
    print(f"Random baseline: 10% (1/10 choices)")
    print(f"\nPer-type accuracy:")
    for qtype in sorted(type_stats):
        s = type_stats[qtype]
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  {qtype:30s}: {s['correct']}/{s['total']} ({pct:.0f}%)")

    avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
    avg_tokens = sum(r["prompt_tokens"] for r in results) / len(results)
    print(f"\nAvg latency: {avg_latency:.1f}s")
    print(f"Avg prompt tokens: {avg_tokens:.0f}")

    # Save results
    output = {
        "metadata": {
            "dataset": "LoCoMo-MC10",
            "dataset_path": DATASET_PATH,
            "model": model,
            "strategy": strategy,
            "label": label,
            "num_questions": len(entries),
            "num_correct": correct,
            "accuracy": correct / total if total else 0,
            "random_baseline": 0.1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "per_type": {
            qtype: {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": s["correct"] / s["total"] if s["total"] else 0,
            }
            for qtype, s in sorted(type_stats.items())
        },
        "results": results,
    }

    results_dir = os.path.join(BENCH_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"benchmark_results_{label}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Logs at {log_dir}/")

    # Cleanup
    kill_ctxpact()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark ctxpact on LoCoMo-MC10"
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of questions (0 = all). Default: 20",
    )
    parser.add_argument(
        "--types", nargs="+",
        help="Filter by question types (e.g., multi_hop temporal_reasoning)",
    )
    parser.add_argument(
        "--label", default=None,
        help="Run label for output files (default: locomo-{strategy}-{model_short})",
    )
    parser.add_argument(
        "--strategy", default=None,
        help="Extraction strategy (overrides BENCH_STRATEGY env var)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name (overrides BENCH_MODEL env var)",
    )
    args = parser.parse_args()

    strategy = args.strategy or STRATEGY
    model = args.model or MODEL

    label = args.label or f"locomo-{strategy}-qwen35"
    run_benchmark(n=args.n, types=args.types, label=label,
                  strategy=strategy, model=model)
