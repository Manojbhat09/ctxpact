#!/usr/bin/env python3
"""Direct benchmark: run ReadAgentExtractor on all 8 Frankenstein queries."""

import asyncio
import json
import logging
import os
import re
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ctxpact.compaction.rlm_extractor import ReadAgentExtractor
from ctxpact.compaction.book import ConversationBook

logging.basicConfig(level=logging.INFO, format="%(message)s")

BOOK_PATH = os.environ.get("BENCH_BOOK", os.path.join(BENCH_DIR, "data", "frankenstein.txt"))
PROVIDER_URL = os.environ.get("BENCH_PROVIDER", "http://localhost:8080/v1")
MODEL = os.environ.get("BENCH_MODEL", "Qwen3.5-9B-Q8_0.gguf")
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_BUDGET = 12000
LABEL = sys.argv[1] if len(sys.argv) > 1 else "readagent-v1"

QUERIES = [
    {
        "id": 1,
        "query": "How many chapters are there in this book? List all chapter numbers.",
        "type": "structural",
        "keyword_groups": [["24"]],
    },
    {
        "id": 2,
        "query": "Who is Justine Moritz and what happens to her?",
        "type": "character",
        "keyword_groups": [["justine"], ["trial", "executed", "hanged", "condemned", "guilty"]],
    },
    {
        "id": 3,
        "query": "What does the creature ask Frankenstein to create for him?",
        "type": "plot",
        "keyword_groups": [["companion", "mate", "female", "wife", "partner"]],
    },
    {
        "id": 4,
        "query": "Where does Clerval get murdered?",
        "type": "detail",
        "keyword_groups": [["clerval"], ["ireland", "irish"]],
    },
    {
        "id": 5,
        "query": "What is the name of the ship captain who rescues Frankenstein from the ice?",
        "type": "character",
        "keyword_groups": [["walton"]],
    },
    {
        "id": 6,
        "query": "How does the novel end? What happens to the creature?",
        "type": "plot",
        "keyword_groups": [["arctic", "ice", "north", "pole"], ["death", "dies", "fire", "funeral", "perish", "destroy"]],
    },
    {
        "id": 7,
        "query": "What does Victor study at the university of Ingolstadt?",
        "type": "detail",
        "keyword_groups": [["chemistry", "natural philosophy", "science", "philosophy", "anatomy"]],
    },
    {
        "id": 8,
        "query": "Who teaches the creature to read and speak? How does it learn?",
        "type": "detail",
        "keyword_groups": [["felix", "de lacey", "safie", "delacey", "cottage"]],
    },
]


def check_keywords(response: str, keyword_groups: list[list[str]]) -> dict:
    response_lower = response.lower()
    groups = []
    for alts in keyword_groups:
        found = [k for k in alts if k in response_lower]
        groups.append({
            "alternatives": alts,
            "found": found,
            "pass": len(found) > 0,
        })
    all_pass = all(g["pass"] for g in groups)
    score = sum(1 for g in groups if g["pass"]) / len(groups) if groups else 0
    return {"groups": groups, "all_pass": all_pass, "score": score}


async def run_query(extractor, book, q):
    query = q["query"]
    t0 = time.time()

    try:
        result = await extractor.extract(book, query, TOKEN_BUDGET)
        elapsed = time.time() - t0

        # Send result + query to LLM for answer
        import httpx
        messages = [
            {"role": "system", "content": (
                "You are a reading comprehension assistant. "
                "Answer questions using ONLY the information from "
                "the provided text. Do not use your prior knowledge "
                "about the book. If the text mentions a specific "
                "location, name, or detail, use that exact "
                "information in your answer."
            )},
            {"role": "user", "content": result},
        ]
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{PROVIDER_URL}/chat/completions",
                json={
                    "model": MODEL,
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        response_text = data["choices"][0]["message"]["content"]
        total_elapsed = time.time() - t0

        kw_check = check_keywords(response_text, q["keyword_groups"])

        return {
            "strategy": "readagent",
            "query_id": q["id"],
            "query": query,
            "query_type": q["type"],
            "response": response_text,
            "response_length": len(response_text),
            "latency_seconds": round(total_elapsed, 1),
            "keyword_check": kw_check,
            "accuracy": kw_check["score"],
            "all_pass": kw_check["all_pass"],
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "strategy": "readagent",
            "query_id": q["id"],
            "query": query,
            "query_type": q["type"],
            "response": "",
            "response_length": 0,
            "latency_seconds": round(elapsed, 1),
            "keyword_check": {"groups": [], "all_pass": False, "score": 0},
            "accuracy": 0.0,
            "all_pass": False,
            "error": str(e),
        }


async def main():
    print("Loading book...")
    with open(BOOK_PATH) as f:
        book_text = f.read()

    book = ConversationBook()
    book.build_from_messages([{"role": "user", "content": book_text}])
    print(f"Book: {book.section_count} sections, {book.total_tokens} tokens")

    extractor = ReadAgentExtractor(
        provider_url=PROVIDER_URL,
        model=MODEL,
    )

    print(f"\nLabel: {LABEL}")
    print(f"Queries: {len(QUERIES)}")
    print("=" * 70)

    results = []
    total_score = 0

    for q in QUERIES:
        print(f"\n  Q{q['id']}: {q['query']}")
        result = await run_query(extractor, book, q)
        results.append(result)

        status = "PASS" if result["all_pass"] else "FAIL"
        total_score += result["accuracy"]
        print(f"    [{status}] score={result['accuracy']:.1f} time={result['latency_seconds']:.1f}s")
        print(f"    Response: {result['response'][:150]}...")
        if not result["all_pass"]:
            for g in result["keyword_check"]["groups"]:
                if not g["pass"]:
                    print(f"    MISSING: {g['alternatives']}")

    avg_score = total_score / len(QUERIES)
    pass_count = sum(1 for r in results if r["all_pass"])
    print(f"\n{'='*70}")
    print(f"TOTAL: {pass_count}/{len(QUERIES)} passed, avg score={avg_score:.2f}")
    print(f"{'='*70}")

    # Save results
    output = {
        "metadata": {
            "book": BOOK_PATH,
            "book_chars": len(book_text),
            "book_est_tokens": len(book_text) // 4,
            "model": MODEL,
            "label": LABEL,
            "num_strategies": 1,
            "num_queries": len(QUERIES),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results,
    }
    results_dir = os.path.join(BENCH_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    outfile = os.path.join(results_dir, f"benchmark_results_{LABEL}.json")
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    asyncio.run(main())
