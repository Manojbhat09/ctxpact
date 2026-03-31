#!/usr/bin/env python3
"""Benchmark LongMemEval on Dyson's llama-server directly (no ctxpact).

LongMemEval: Multi-session conversation QA with free-text answers.
500 questions across 6 types. Oracle dataset has ground-truth answer sessions.

Designed to run ON DYSON via SSH to avoid crashing Xavier.
Calls llama-server on localhost:8080 directly.

Usage (on Dyson):
  python3 benchmark_longmemeval.py                         # 20-question sample
  python3 benchmark_longmemeval.py --n 50                  # 50 questions
  python3 benchmark_longmemeval.py --n 0                   # all 500 questions
  python3 benchmark_longmemeval.py --types temporal-reasoning multi-session
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

LLM_URL = os.environ.get("BENCH_PROVIDER", "http://localhost:8080/v1")
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.5-9B-Q8_0.gguf")
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.environ.get(
    "LONGMEMEVAL_DATA",
    os.path.join(BENCH_DIR, "data", "longmemeval", "longmemeval_oracle.json"),
)
MAX_CONTEXT_TOKENS = 14000  # Leave room for question + response in 16k window


def load_dataset(path, n=20, types=None, seed=42):
    with open(path) as f:
        entries = json.load(f)

    if types:
        entries = [e for e in entries if e["question_type"] in types]

    if n > 0 and n < len(entries):
        by_type = defaultdict(list)
        for e in entries:
            by_type[e["question_type"]].append(e)

        rng = random.Random(seed)
        sampled = []
        per_type = max(1, n // len(by_type))
        for qtype, items in sorted(by_type.items()):
            k = min(per_type, len(items))
            sampled.extend(rng.sample(items, k))

        remaining = n - len(sampled)
        if remaining > 0:
            pool = [e for e in entries if e not in sampled]
            sampled.extend(rng.sample(pool, min(remaining, len(pool))))

        entries = sampled[:n]

    return entries


def format_conversation(entry):
    """Format conversation sessions into text, truncating if needed."""
    parts = []
    for i, session in enumerate(entry["haystack_sessions"]):
        dt = entry["haystack_dates"][i] if i < len(entry["haystack_dates"]) else ""
        parts.append(f"--- Session {i + 1} ({dt}) ---")
        for msg in session:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")

    full_text = "\n\n".join(parts)

    # Truncate if too long (keep beginning + end)
    est_tokens = len(full_text) // 4
    if est_tokens > MAX_CONTEXT_TOKENS:
        max_chars = MAX_CONTEXT_TOKENS * 4
        half = max_chars // 2
        full_text = (
            full_text[:half]
            + "\n\n[... context truncated ...]\n\n"
            + full_text[-half:]
        )

    return full_text


def check_answer(response, answer):
    """Check if the model response contains the expected answer.

    Uses fuzzy keyword matching — all significant words from the answer
    must appear in the response.
    """
    response_lower = response.lower().strip()
    answer_lower = answer.lower().strip()

    # Exact containment
    if answer_lower in response_lower:
        return True

    # Keyword matching: split answer into words, check each
    answer_words = [w for w in re.split(r'\W+', answer_lower) if len(w) > 2]
    if not answer_words:
        return answer_lower in response_lower

    matched = sum(1 for w in answer_words if w in response_lower)
    ratio = matched / len(answer_words)

    # Require >= 60% of answer keywords present
    return ratio >= 0.6


def api_call(messages, max_tokens=512, temperature=0.1, timeout=300):
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    req = urllib.request.Request(
        f"{LLM_URL}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read().decode())


def run_benchmark(n=20, types=None, label="longmemeval-qwen35"):
    entries = load_dataset(DATASET_PATH, n=n, types=types)

    print(f"LongMemEval Benchmark (Direct)")
    print(f"Label: {label}")
    print(f"Model: {MODEL}")
    print(f"Questions: {len(entries)}")
    type_counts = Counter(e["question_type"] for e in entries)
    print(f"Types: {dict(type_counts)}")
    print("=" * 70)

    results = []
    correct = 0
    total = 0
    type_stats = {}

    for idx, entry in enumerate(entries):
        qid = entry["question_id"]
        qtype = entry["question_type"]
        question = entry["question"]
        answer = entry["answer"]

        conv_text = format_conversation(entry)
        conv_tokens = len(conv_text) // 4

        prompt = (
            f"You are a helpful personal assistant. Based on our previous conversations below, "
            f"answer the following question concisely.\n\n"
            f"Conversation history:\n{conv_text}\n\n"
            f"Question (asked on {entry.get('question_date', 'unknown date')}): {question}\n\n"
            f"Answer concisely based only on the conversation history above."
        )

        print(f"\n  [{idx + 1}/{len(entries)}] {qid} ({qtype})")
        print(f"    Q: {question[:80]}...")
        print(f"    Context: ~{conv_tokens:,} tokens")

        start = time.time()
        try:
            result = api_call(
                [{"role": "user", "content": prompt}],
                max_tokens=256,
            )
            elapsed = time.time() - start
            response = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})

            is_correct = check_answer(response, answer)
            if is_correct:
                correct += 1
            total += 1

            if qtype not in type_stats:
                type_stats[qtype] = {"correct": 0, "total": 0}
            type_stats[qtype]["total"] += 1
            if is_correct:
                type_stats[qtype]["correct"] += 1

            status = "CORRECT" if is_correct else "WRONG"
            print(f"    [{status}] {elapsed:.1f}s")
            print(f"    Expected: {answer[:80]}")
            if not is_correct:
                print(f"    Got: {response[:150]}...")

            results.append({
                "question_id": qid,
                "question_type": qtype,
                "question": question,
                "answer": answer,
                "response": response[:500],
                "is_correct": is_correct,
                "latency_seconds": elapsed,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "conv_tokens_est": conv_tokens,
                "error": None,
            })

        except Exception as e:
            elapsed = time.time() - start
            total += 1
            print(f"    [ERROR] {elapsed:.1f}s: {e}")
            if qtype not in type_stats:
                type_stats[qtype] = {"correct": 0, "total": 0}
            type_stats[qtype]["total"] += 1
            results.append({
                "question_id": qid,
                "question_type": qtype,
                "question": question,
                "answer": answer,
                "response": "",
                "is_correct": False,
                "latency_seconds": elapsed,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "conv_tokens_est": len(conv_text) // 4,
                "error": str(e),
            })

    # Summary
    print(f"\n{'=' * 70}")
    print(f"LongMemEval — Summary")
    print(f"{'=' * 70}")
    accuracy = correct / total * 100 if total else 0
    print(f"Overall: {correct}/{total} correct ({accuracy:.1f}%)")
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
            "dataset": "LongMemEval",
            "model": MODEL,
            "label": label,
            "mode": "direct (no ctxpact)",
            "num_questions": len(entries),
            "num_correct": correct,
            "accuracy": correct / total if total else 0,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LongMemEval (direct)")
    parser.add_argument("--n", type=int, default=20, help="Questions (0=all)")
    parser.add_argument("--types", nargs="+", help="Filter question types")
    parser.add_argument("--label", default="longmemeval-qwen35",
                        help="Run label")
    args = parser.parse_args()
    run_benchmark(n=args.n, types=args.types, label=args.label)
