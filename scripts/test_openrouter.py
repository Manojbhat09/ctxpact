#!/usr/bin/env python3
"""
ctxpact OpenRouter Integration Test Suite

Run this on your M4 to test:
  1. OpenRouter API connectivity with your key
  2. Summarization quality benchmark across free models
  3. Real summarization with conversation compaction
  4. Circuit breaker failover from local vLLM-MLX to OpenRouter
  5. Full E2E lifecycle (local primary → OOM → OpenRouter fallback → compaction)

Usage:
    python scripts/test_openrouter.py                    # All tests
    python scripts/test_openrouter.py --test 1           # Specific test
    python scripts/test_openrouter.py --test 1 2         # Multiple tests
    python scripts/test_openrouter.py --model qwen/qwen3-coder:free  # Override model

Requires: pip install httpx pydantic tiktoken pyyaml
"""

import sys
import os
import json
import asyncio
import time
import argparse
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

from ctxpact.compaction.tokens import count_tokens, count_message_tokens, count_messages_tokens
from ctxpact.compaction.pruner import DynamicContextPruner
from ctxpact.compaction.detector import SequenceDetector
from ctxpact.compaction.engine import CompactionEngine
from ctxpact.compaction.prompts import build_compaction_prompt, format_messages_for_summary
from ctxpact.compaction.summarizer import Summarizer
from ctxpact.routing.circuit_breaker import CircuitBreaker, CircuitState
from ctxpact.routing.client import LLMClient, BACKEND_DOWN_ERRORS, BackendError
from ctxpact.routing.router import ProviderRouter
from ctxpact.routing.health import HealthChecker
from ctxpact.config import (
    CtxpactConfig, ProviderConfig, CompactionConfig, TriggersConfig,
    DcpConfig, SummarizeConfig, CircuitBreakerConfig, HealthCheckConfig,
)

# =========================================================================
# Config — loaded from config.yaml or defaults
# =========================================================================
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    if os.path.exists(config_path):
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f)
        # Find OpenRouter provider
        for p in data.get("providers", []):
            if "openrouter" in p.get("url", ""):
                return p["api_key"], p["url"], p["model"]
    # Fallback
    return (
        os.environ.get("OPENROUTER_API_KEY", ""),
        "https://openrouter.ai/api/v1",
        "qwen/qwen3-coder:free",
    )

API_KEY, BASE_URL, DEFAULT_MODEL = load_config()

FREE_MODELS = [
    "deepseek/deepseek-r1-0528:free",
    "qwen/qwen3-coder:free",
    "tngtech/deepseek-r1t-chimera:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
]

# =========================================================================
# Test infrastructure
# =========================================================================
passed = 0
failed = 0
errors_list = []
results = {}

async def call_openrouter(model: str, messages: list, max_tokens: int = 1024, timeout: float = 120.0) -> dict:
    """Call OpenRouter API."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://ctxpact.dev",
                    "X-Title": "ctxpact-test",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
            )
            if response.status_code >= 400:
                return {"error": f"HTTP {response.status_code}", "detail": response.text[:500]}
            return response.json()
        except Exception as e:
            return {"error": str(e)}

async def run_test(fn, name):
    global passed, failed
    start = time.time()
    try:
        await fn()
        elapsed = time.time() - start
        passed += 1
        results[name] = {"status": "PASS", "time": elapsed}
        print(f"  ✓ {name} ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - start
        failed += 1
        errors_list.append((name, e))
        results[name] = {"status": "FAIL", "time": elapsed, "error": str(e)}
        print(f"  ✗ {name} ({elapsed:.1f}s)")
        traceback.print_exc()
    print()


# =========================================================================
# TEST 1: OpenRouter connectivity
# =========================================================================
async def test_1_connectivity():
    """Verify OpenRouter API key works and free models are available."""
    print("    Testing API key...")
    assert API_KEY, "No API key found. Set OPENROUTER_API_KEY or configure config.yaml"

    working_models = []
    for model in FREE_MODELS:
        print(f"    Trying {model}...")
        result = await call_openrouter(
            model=model,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=50,
            timeout=30.0,
        )
        if "error" in result:
            print(f"      ✗ {result['error'][:100]}")
        else:
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            model_used = result.get("model", "?")
            print(f"      ✓ Response from {model_used}: {content[:60]}")
            working_models.append(model)

    print(f"\n    Working models: {len(working_models)}/{len(FREE_MODELS)}")
    results["working_models"] = working_models
    assert len(working_models) > 0, "No free models are available"


# =========================================================================
# TEST 2: Summarization quality benchmark
# =========================================================================
async def test_2_summarization_quality():
    """Benchmark free models for <summary> tag output quality."""
    conversation = [
        {"role": "user", "content": "Read server.py and tell me what's wrong"},
        {"role": "assistant", "content": "Let me read that file.",
         "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "server.py"}'}}]},
        {"role": "tool", "content": "from flask import Flask\napp = Flask(__name__)\n\n@app.route('/api/users')\ndef get_users():\n    db = get_db()\n    return db.query('SELECT * FROM users')  # SQL injection risk\n\n@app.route('/api/users', methods=['POST'])\ndef create_user():\n    data = request.json\n    db.execute(f\"INSERT INTO users VALUES ('{data['name']}')\")\n    return {'status': 'ok'}"},
        {"role": "assistant", "content": "I found two SQL injection vulnerabilities in server.py:\n1. Line 7: `db.query('SELECT * FROM users')` - while this one is safe (no user input), the pattern is risky\n2. Line 12: `db.execute(f\"INSERT INTO users VALUES ('{data['name']}')\")`  - this is a critical SQL injection.\n\nFix: Use parameterized queries."},
        {"role": "user", "content": "Fix both issues and add input validation"},
        {"role": "assistant", "content": "Writing the fix.",
         "tool_calls": [{"function": {"name": "write_file", "arguments": json.dumps({"path": "server.py", "content": "from flask import Flask, request, jsonify\nfrom werkzeug.exceptions import BadRequest\nimport re\n\napp = Flask(__name__)\n\ndef validate_name(name: str) -> str:\n    if not name or len(name) > 100:\n        raise BadRequest('Name must be 1-100 chars')\n    return name.strip()\n\n@app.route('/api/users')\ndef get_users():\n    db = get_db()\n    return jsonify(db.execute('SELECT * FROM users').fetchall())\n\n@app.route('/api/users', methods=['POST'])\ndef create_user():\n    data = request.get_json(force=True)\n    name = validate_name(data.get('name', ''))\n    db = get_db()\n    db.execute('INSERT INTO users (name) VALUES (?)', (name,))\n    db.commit()\n    return jsonify({'status': 'created'}), 201\n"})}}]},
        {"role": "tool", "content": "File written successfully."},
        {"role": "user", "content": "Run the tests"},
        {"role": "assistant", "content": "Running pytest..."},
        {"role": "tool", "content": "2 passed, 1 failed — test_create_user fails because get_db() isn't initialized."},
        {"role": "assistant", "content": "The test_create_user test fails because get_db() isn't initialized in the test fixture."},
    ]

    summary_input = format_messages_for_summary(conversation)
    summary_prompt = build_compaction_prompt(summary_input, code_line_limit=50)
    input_tokens = count_messages_tokens(conversation)

    print(f"    Input: {len(conversation)} messages, {input_tokens} tokens")
    print()

    model_results = {}
    for model in FREE_MODELS:
        print(f"    Testing {model}...")
        start = time.time()
        result = await call_openrouter(model=model, messages=summary_prompt, max_tokens=2000, timeout=120.0)
        elapsed = time.time() - start

        if "error" in result:
            print(f"      ✗ Error: {result['error'][:100]}")
            model_results[model] = {"status": "error", "error": result["error"][:200]}
            continue

        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        has_summary_tags = "<summary>" in content and "</summary>" in content
        has_file_paths = "server.py" in content
        has_error_info = "injection" in content.lower()
        has_fix_info = "parameterized" in content.lower() or "validate" in content.lower()
        has_status = "test" in content.lower() or "fail" in content.lower()

        quality = sum([has_summary_tags, has_file_paths, has_error_info, has_fix_info, has_status])
        summary_tokens = count_tokens(content)

        model_results[model] = {
            "status": "ok", "time": elapsed, "tokens": summary_tokens, "quality": quality,
            "tags": has_summary_tags, "paths": has_file_paths, "errors": has_error_info,
            "fixes": has_fix_info, "test_status": has_status, "content": content,
        }

        compression = (1 - summary_tokens / input_tokens) * 100 if input_tokens > 0 else 0
        print(f"      ✓ {elapsed:.1f}s, {summary_tokens} tokens ({compression:.0f}% compression), quality={quality}/5")
        print(f"        Tags:{'✓' if has_summary_tags else '✗'}  "
              f"Paths:{'✓' if has_file_paths else '✗'}  "
              f"Errors:{'✓' if has_error_info else '✗'}  "
              f"Fixes:{'✓' if has_fix_info else '✗'}  "
              f"Status:{'✓' if has_status else '✗'}")

    working = {k: v for k, v in model_results.items() if v["status"] == "ok"}
    if working:
        best = max(working.items(), key=lambda x: (x[1]["quality"], -x[1]["time"]))
        print(f"\n    🏆 BEST MODEL: {best[0]} (quality={best[1]['quality']}/5, {best[1]['time']:.1f}s)")
        print(f"\n    Summary preview:")
        for line in best[1]["content"].split("\n")[:10]:
            print(f"    | {line}")
        results["best_summarization_model"] = best[0]
        results["model_benchmark"] = model_results
    else:
        print("\n    ⚠ No models produced usable output")

    assert len(working) > 0, "At least one model should work for summarization"


# =========================================================================
# TEST 3: Real summarization via OpenRouter
# =========================================================================
async def test_3_real_summarization():
    """Actually send a conversation to OpenRouter for summarization and validate quality."""
    messages_to_compact = [
        {"role": "user", "content": "Help me fix the login bug in auth.py"},
        {"role": "assistant", "content": "I'll look at auth.py. The issue is likely in the token validation."},
        {"role": "user", "content": "The error is 'jwt.ExpiredSignatureError' after 1 hour"},
        {"role": "assistant", "content": "Found it. The token expiry is set to 3600 seconds (1 hour) but there's no refresh mechanism. Here's the fix:\n\n```python\ndef create_token(user_id: str) -> str:\n    payload = {\n        'sub': user_id,\n        'exp': datetime.utcnow() + timedelta(hours=24),\n        'iat': datetime.utcnow(),\n    }\n    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')\n\ndef refresh_token(token: str) -> str:\n    try:\n        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'], options={'verify_exp': False})\n        if datetime.utcnow() - datetime.fromtimestamp(payload['iat']) > timedelta(days=7):\n            raise ValueError('Token too old to refresh')\n        return create_token(payload['sub'])\n    except jwt.InvalidTokenError:\n        raise ValueError('Invalid token')\n```"},
        {"role": "user", "content": "Tests pass now. What about the CORS issue on /api/users?"},
        {"role": "assistant", "content": "The CORS issue is because Flask-CORS isn't configured. Add:\n```python\nfrom flask_cors import CORS\nCORS(app, resources={r'/api/*': {'origins': ['http://localhost:3000']}})\n```"},
    ]

    original_tokens = count_messages_tokens(messages_to_compact)
    summary_text = format_messages_for_summary(messages_to_compact)
    prompt_messages = build_compaction_prompt(summary_text, code_line_limit=50)

    print(f"    Input: {len(messages_to_compact)} messages, {original_tokens} tokens")

    model = DEFAULT_MODEL
    result = await call_openrouter(model=model, messages=prompt_messages, max_tokens=1500, timeout=120.0)

    if "error" in result:
        # Try fallback
        model = "deepseek/deepseek-r1-0528:free"
        print(f"    First model failed, trying {model}...")
        result = await call_openrouter(model=model, messages=prompt_messages, max_tokens=1500, timeout=120.0)

    assert "error" not in result, f"OpenRouter error: {result.get('error')}"

    summary = result["choices"][0]["message"]["content"]
    summary_tokens = count_tokens(summary)

    print(f"    Model: {model}")
    print(f"    Summary: {summary_tokens} tokens ({(1 - summary_tokens/original_tokens)*100:.0f}% compression)")
    print(f"    ---")
    for line in summary.split("\n")[:12]:
        print(f"    | {line}")
    print(f"    ---")

    # Verify reconstruction
    reconstructed = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": summary},
        {"role": "user", "content": "Now help me add rate limiting to the API"},
    ]
    recon_tokens = count_messages_tokens(reconstructed)
    print(f"    Reconstruction: {original_tokens} → {recon_tokens} tokens")

    # Quality checks
    checks = {
        "mentions_auth": "auth" in summary.lower() or "token" in summary.lower(),
        "mentions_cors": "cors" in summary.lower(),
        "mentions_files": "auth.py" in summary,
        "concise": summary_tokens < original_tokens,
    }
    passed_checks = sum(checks.values())
    for check, val in checks.items():
        print(f"    {'✓' if val else '✗'} {check}")

    assert passed_checks >= 2, f"Summary quality too low: {passed_checks}/{len(checks)}"


# =========================================================================
# TEST 4: Circuit breaker failover to real OpenRouter
# =========================================================================
async def test_4_circuit_breaker_failover():
    """
    Start a mock primary that always OOMs, verify failover to real OpenRouter.
    This tests the actual production failover path.
    """
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class OOMHandler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            self.send_error(503, "OOM")
        def do_POST(self):
            self.send_error(503, "CUDA out of memory")

    server = HTTPServer(("127.0.0.1", 0), OOMHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        providers = [
            ProviderConfig(
                name="local-dead", url=f"http://127.0.0.1:{port}/v1",
                model="dead-model", max_context=32000, priority=1, api_key="dummy",
                health_check=HealthCheckConfig(endpoint="/v1/models", interval_seconds=60),
            ),
            ProviderConfig(
                name="openrouter-fallback", url="https://openrouter.ai/api/v1",
                model=DEFAULT_MODEL, max_context=128000, priority=2,
                api_key=API_KEY,
                health_check=HealthCheckConfig(endpoint="/models", interval_seconds=60),
            ),
        ]

        router = ProviderRouter(providers, CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=30))

        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2+2? Answer in one word."},
        ]

        # First request — should fail on primary, fail again, then route to OpenRouter
        print("    Sending request (primary is dead, should failover to OpenRouter)...")
        response, provider = await router.chat_completion(messages=messages, max_tokens=50)
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"    → Served by: {provider.name}")
        print(f"    → Response: {content[:100]}")

        # Verify it went to OpenRouter
        assert provider.name == "openrouter-fallback", f"Expected OpenRouter, got {provider.name}"

        # Check circuit state
        primary_state = router.breakers["local-dead"].state
        print(f"    → Primary circuit: {primary_state.value}")
        assert primary_state in (CircuitState.OPEN, CircuitState.HALF_OPEN), \
            f"Primary should be OPEN/HALF_OPEN, got {primary_state.value}"

        # Second request — should go directly to OpenRouter
        print("    Sending follow-up request...")
        response2, provider2 = await router.chat_completion(
            messages=[{"role": "user", "content": "Capital of France? One word."}],
            max_tokens=50,
        )
        content2 = response2.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"    → Served by: {provider2.name}: {content2[:80]}")

        print("    ✓ Circuit breaker failover to OpenRouter confirmed!")

    finally:
        server.shutdown()


# =========================================================================
# TEST 5: Full E2E lifecycle with real OpenRouter
# =========================================================================
async def test_5_full_e2e():
    """
    Complete lifecycle:
    1. Mock primary serves initial small requests
    2. Primary OOMs when context grows
    3. Circuit breaker opens → OpenRouter takes over
    4. Compaction fires to reduce context
    """
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class SmartVLLMHandler(BaseHTTPRequestHandler):
        request_count = 0
        def log_message(self, *a): pass

        def do_GET(self):
            if "/models" in self.path:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": "nanbeige"}]}).encode())
            else:
                self.send_error(404)

        def do_POST(self):
            SmartVLLMHandler.request_count += 1
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            messages = body.get("messages", [])
            total_chars = sum(len(json.dumps(m)) for m in messages)
            # OOM if context is large (simulates 32k limit)
            if total_chars > 8000:
                self.send_error(503, f"OOM: context too large ({total_chars} chars)")
                return

            turn = SmartVLLMHandler.request_count
            content = f"Response from local Nanbeige (turn {turn}). " + "Processing your request. " * 20
            resp = {
                "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "model": "nanbeige",
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())

    server = HTTPServer(("127.0.0.1", 0), SmartVLLMHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        providers = [
            ProviderConfig(
                name="local-nanbeige", url=f"http://127.0.0.1:{port}/v1",
                model="nanbeige", max_context=32000, priority=1, api_key="dummy",
                health_check=HealthCheckConfig(endpoint="/v1/models", interval_seconds=60),
            ),
            ProviderConfig(
                name="openrouter-cloud", url="https://openrouter.ai/api/v1",
                model=DEFAULT_MODEL, max_context=128000, priority=2,
                api_key=API_KEY,
                health_check=HealthCheckConfig(endpoint="/models", interval_seconds=60),
            ),
        ]

        router = ProviderRouter(providers, CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=30))
        engine = CompactionEngine(CompactionConfig(
            triggers=TriggersConfig(token_ratio=0.50),
            stage1_dcp=DcpConfig(enabled=True),
            stage2_summarize=SummarizeConfig(retention_window=4, eviction_window=0.40),
        ))

        messages = [{"role": "system", "content": "You are a coding assistant."}]
        providers_used = []
        compactions = 0

        print("    Running multi-turn conversation...")
        for turn in range(1, 15):
            # Growing prompts
            prompt = f"Turn {turn}: Implement feature_{turn} with comprehensive error handling, " \
                     f"input validation, logging, and retry logic. " * (2 + turn)
            messages.append({"role": "user", "content": prompt})

            tokens = count_messages_tokens(messages)

            # Check compaction
            should, reason = engine.should_compact(messages, max_context=32000)
            if should:
                compactions += 1
                prune_result = engine.pruner.prune(messages)
                classification = engine.detector.classify(prune_result.messages)
                # Use a mock summary for local compaction
                messages = (
                    classification.system_messages
                    + [{"role": "user", "content": f"<summary>Summary of turns 1-{turn}: built features 1-{turn}</summary>"}]
                    + classification.retention_window
                )
                tokens = count_messages_tokens(messages)
                print(f"      Turn {turn}: ⚡ Compacted to {tokens} tokens")

            try:
                response, provider = await router.chat_completion(messages=messages, max_tokens=200)
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                messages.append({"role": "assistant", "content": content[:500]})
                providers_used.append(provider.name)
                if turn % 3 == 0 or provider.name != (providers_used[-2] if len(providers_used) > 1 else providers_used[0]):
                    print(f"      Turn {turn}: {tokens} tokens → {provider.name}")
            except RuntimeError as e:
                print(f"      Turn {turn}: All failed: {e}")
                break

        local_count = providers_used.count("local-nanbeige")
        cloud_count = providers_used.count("openrouter-cloud")
        print(f"\n    LIFECYCLE RESULTS:")
        print(f"    Turns completed: {len(providers_used)}")
        print(f"    Local Nanbeige served: {local_count}")
        print(f"    OpenRouter served: {cloud_count}")
        print(f"    Compactions: {compactions}")
        print(f"    Final tokens: {count_messages_tokens(messages)}")

        assert local_count > 0, "Local should have served some requests"
        # We expect cloud to take over at some point
        if cloud_count > 0:
            print("    ✓ Full lifecycle confirmed: local → OOM → cloud failover")
        else:
            print("    ⚠ Primary never OOMed (context may not have grown enough)")

    finally:
        server.shutdown()


# =========================================================================
# Runner
# =========================================================================
async def main():
    parser = argparse.ArgumentParser(description="ctxpact OpenRouter Integration Tests")
    parser.add_argument("--test", nargs="*", type=int, help="Specific test numbers to run (1-5)")
    parser.add_argument("--model", type=str, help="Override default model")
    args = parser.parse_args()

    global DEFAULT_MODEL
    if args.model:
        DEFAULT_MODEL = args.model

    if not API_KEY:
        print("ERROR: No OpenRouter API key found.")
        print("Either set OPENROUTER_API_KEY env var or configure config.yaml")
        sys.exit(1)

    print("=" * 70)
    print("  ctxpact OpenRouter Integration Test Suite")
    print(f"  API: {BASE_URL}")
    print(f"  Default model: {DEFAULT_MODEL}")
    print(f"  Key: {API_KEY[:15]}...{API_KEY[-4:]}")
    print("=" * 70)
    print()

    all_tests = [
        (1, "OpenRouter API connectivity", test_1_connectivity),
        (2, "Summarization quality benchmark", test_2_summarization_quality),
        (3, "Real summarization via OpenRouter", test_3_real_summarization),
        (4, "Circuit breaker failover to OpenRouter", test_4_circuit_breaker_failover),
        (5, "Full E2E lifecycle (local → OOM → OpenRouter)", test_5_full_e2e),
    ]

    selected = args.test if args.test else [t[0] for t in all_tests]

    for num, name, fn in all_tests:
        if num in selected:
            await run_test(fn, f"TEST {num}: {name}")

    # Report
    print("=" * 70)
    print("  FINAL REPORT")
    print("=" * 70)
    for name, result in results.items():
        if isinstance(result, dict) and "status" in result:
            s = "✓" if result["status"] == "PASS" else "✗"
            print(f"  {s} {name} [{result.get('time', 0):.1f}s]")

    if "best_summarization_model" in results:
        print(f"\n  Recommended summarization model: {results['best_summarization_model']}")

    print()
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED ✓")
    else:
        print(f"  {passed} passed, {failed} FAILED")
        for name, err in errors_list:
            print(f"    FAIL: {name}: {err}")

    print("=" * 70)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
