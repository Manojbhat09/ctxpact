#!/usr/bin/env python3
"""
Stress test: Push conversations past the 70% threshold to trigger compaction.

This simulates a multi-turn conversation that grows past 22.4k tokens
(70% of 32k) to validate the two-stage compaction pipeline.

Usage:
    python scripts/stress_test_compaction.py [--url http://localhost:8000]

What it does:
    1. Sends an initial system + user message
    2. Continues the conversation with follow-up questions
    3. Monitors token growth via X-Session-ID and /v1/sessions endpoint
    4. Reports when compaction fires and the before/after token counts
"""

import argparse
import json
import sys
import time
import urllib.request


def api_call(url: str, endpoint: str, method: str = "GET",
             data: dict | None = None, headers: dict | None = None) -> dict:
    """Make an HTTP request and return JSON response."""
    full_url = f"{url.rstrip('/')}{endpoint}"
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(full_url, data=body, headers=req_headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode()[:200]}")
        raise
    except Exception as e:
        print(f"  Connection error: {e}")
        raise


def get_session_info(url: str, session_id: str) -> dict | None:
    """Fetch session state from ctxpact debug endpoint."""
    try:
        return api_call(url, f"/v1/sessions/{session_id}")
    except Exception:
        return None


def run_stress_test(url: str, max_turns: int = 20, verbose: bool = True):
    """Run a multi-turn conversation designed to trigger compaction."""

    session_id = f"stress-test-{int(time.time())}"
    print(f"\n{'='*70}")
    print(f"  ctxpact Compaction Stress Test")
    print(f"  Target: {url}")
    print(f"  Session: {session_id}")
    print(f"  Max turns: {max_turns}")
    print(f"  Compaction threshold: 70% of 32k = ~22,400 tokens")
    print(f"{'='*70}\n")

    # Conversation messages — designed to generate verbose responses
    # that fill up the 32k context quickly
    prompts = [
        # Turn 1: Big initial context
        "Write a detailed Python class for a REST API client with authentication, "
        "retry logic, rate limiting, connection pooling, and comprehensive error handling. "
        "Include docstrings and type hints for every method.",

        # Turn 2: Expand it
        "Now add comprehensive unit tests for that class using pytest. "
        "Test every method including edge cases, mock the HTTP calls, "
        "and test the retry logic with different failure scenarios.",

        # Turn 3: More context
        "Add a caching layer to the API client. Implement both in-memory "
        "and Redis-based caching with configurable TTL, cache invalidation, "
        "and cache key generation from request parameters.",

        # Turn 4: Even more
        "Now create a CLI tool using argparse that uses this API client. "
        "Support all CRUD operations, multiple output formats (JSON, table, CSV), "
        "configuration files, and verbose logging.",

        # Turn 5: Push past threshold
        "Refactor everything into a proper Python package with setup.py, "
        "requirements.txt, Dockerfile, CI/CD config, and comprehensive README. "
        "Show the complete directory structure and every file.",

        # Turns 6+: Keep pushing to see multiple compactions
        "Add async support to the entire API client using aiohttp. "
        "Show the full async version of every method with proper connection management.",

        "Write integration tests that test the full stack against a mock server. "
        "Include load testing scenarios and performance benchmarks.",

        "Create a monitoring dashboard for the API client using FastAPI + htmx. "
        "Track request counts, latencies, error rates, and cache hit ratios.",

        "Add OpenTelemetry tracing and Prometheus metrics to the API client. "
        "Include distributed tracing across async calls.",

        "Write comprehensive API documentation in OpenAPI format with examples "
        "for every endpoint, error response, and authentication flow.",
    ]

    messages = [
        {"role": "system", "content": (
            "You are a senior Python developer. Provide complete, production-ready code "
            "with full implementations (not stubs). Include all imports, type hints, "
            "docstrings, and error handling. Be thorough and verbose."
        )},
    ]

    compaction_events_seen = 0

    for turn in range(min(max_turns, len(prompts))):
        prompt = prompts[turn]
        messages.append({"role": "user", "content": prompt})

        print(f"Turn {turn + 1}/{max_turns}: Sending ({len(prompt)} chars)...")
        start = time.time()

        try:
            response = api_call(
                url, "/v1/chat/completions",
                method="POST",
                data={
                    "model": "default",
                    "messages": messages,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                },
                headers={"X-Session-ID": session_id},
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            break

        elapsed = time.time() - start
        choice = response.get("choices", [{}])[0]
        assistant_content = choice.get("message", {}).get("content", "")
        usage = response.get("usage", {})
        ctxpact_meta = response.get("_ctxpact", {})

        # Add assistant response to our local messages
        messages.append({"role": "assistant", "content": assistant_content})

        # Check for compaction
        current_compactions = ctxpact_meta.get("compaction_events", 0)
        new_compaction = current_compactions > compaction_events_seen

        provider = ctxpact_meta.get("provider", "?")
        msg_count = ctxpact_meta.get("message_count", "?")

        print(f"  Response: {len(assistant_content)} chars in {elapsed:.1f}s")
        print(f"  Provider: {provider} | Messages in session: {msg_count}")
        print(f"  Usage: prompt={usage.get('prompt_tokens', '?')} "
              f"completion={usage.get('completion_tokens', '?')}")

        if new_compaction:
            compaction_events_seen = current_compactions
            print(f"  🔄 COMPACTION #{current_compactions} TRIGGERED!")

            # Fetch detailed session info
            session_info = get_session_info(url, session_id)
            if session_info and session_info.get("compaction_events"):
                latest = session_info["compaction_events"][-1]
                print(f"     Trigger: {latest.get('trigger', '?')}")
                print(f"     Tokens: {latest.get('tokens_before', '?')} → "
                      f"{latest.get('tokens_after', '?')}")
                print(f"     Stage: {latest.get('stage', '?')}")

        print()

    # Final summary
    print(f"\n{'='*70}")
    print("  STRESS TEST RESULTS")
    print(f"{'='*70}")
    print(f"  Turns completed: {min(max_turns, len(prompts))}")
    print(f"  Compaction events: {compaction_events_seen}")

    session_info = get_session_info(url, session_id)
    if session_info:
        print(f"  Final message count: {session_info.get('message_count', '?')}")
        print(f"  Total input tokens: {session_info.get('total_input_tokens', '?'):,}")
        print(f"  Total output tokens: {session_info.get('total_output_tokens', '?'):,}")
        print(f"  User turns: {session_info.get('user_turn_count', '?')}")

        events = session_info.get("compaction_events", [])
        if events:
            print(f"\n  Compaction Events:")
            for i, evt in enumerate(events, 1):
                print(f"    #{i}: {evt.get('tokens_before', '?')} → "
                      f"{evt.get('tokens_after', '?')} tokens "
                      f"(stage: {evt.get('stage', '?')})")

    print(f"{'='*70}\n")

    if compaction_events_seen > 0:
        print("✅ Compaction is working! The pipeline triggered successfully.")
    else:
        print("⚠️  No compaction triggered. Try increasing max_turns or lowering "
              "the token_ratio threshold in config.yaml.")

    return compaction_events_seen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ctxpact compaction stress test")
    parser.add_argument("--url", default="http://localhost:8000", help="ctxpact URL")
    parser.add_argument("--turns", type=int, default=10, help="Max conversation turns")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    try:
        # Quick health check first
        health = api_call(args.url, "/health")
        providers = health.get("providers", [])
        available = [p for p in providers if p.get("is_available")]
        print(f"ctxpact is healthy — {len(available)}/{len(providers)} providers available")
    except Exception:
        print(f"Cannot reach ctxpact at {args.url}")
        print("Make sure ctxpact is running: make run")
        sys.exit(1)

    run_stress_test(args.url, max_turns=args.turns, verbose=args.verbose)
