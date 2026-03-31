#!/usr/bin/env bash
# ============================================================
# Quick Validation: Test Nanbeige4.1-3B summarization with Forge
#
# This script validates whether your local model produces
# good-enough summaries for context compaction.
#
# Prerequisites:
#   1. vLLM-MLX running on localhost:8080
#   2. npm install -g @antinomyhq/forge
# ============================================================

set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8080/v1}"
MODEL="${MODEL:-Nanbeige/Nanbeige4.1-3B}"

echo "=== ctxpact: Forge Validation Script ==="
echo ""

# Step 1: Check if vLLM-MLX is running
echo "1. Checking vLLM-MLX at ${VLLM_URL}..."
if curl -sf "${VLLM_URL}/models" > /dev/null 2>&1; then
    echo "   ✅ vLLM-MLX is running"
else
    echo "   ❌ vLLM-MLX not reachable at ${VLLM_URL}"
    echo "   Start it with: vllm serve ${MODEL} --host 0.0.0.0 --port 8080"
    exit 1
fi

# Step 2: Test basic completion
echo ""
echo "2. Testing basic chat completion..."
RESPONSE=$(curl -sf "${VLLM_URL}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one sentence.\"}],
        \"max_tokens\": 50
    }" 2>&1)

if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print('   ✅ Response:', d['choices'][0]['message']['content'][:100])" 2>/dev/null; then
    :
else
    echo "   ❌ Failed to get completion"
    echo "   Raw response: $RESPONSE"
    exit 1
fi

# Step 3: Test summarization capability (the critical question!)
echo ""
echo "3. Testing summarization quality (the key question for compaction)..."
SUMMARY_RESPONSE=$(curl -sf "${VLLM_URL}/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"${MODEL}"'",
        "messages": [
            {"role": "system", "content": "You are a context compaction assistant. Summarize conversations concisely."},
            {"role": "user", "content": "Summarize this conversation into a structured summary:\n\n[USER]: Help me debug why my Python FastAPI app crashes on startup\n[ASSISTANT]: Lets check the traceback. Can you share the error?\n[USER]: It says ModuleNotFoundError: No module named uvicorn\n[ASSISTANT]: You need to install uvicorn. Run: pip install uvicorn[standard]\n[USER]: That fixed it! But now I get port 8000 already in use\n[ASSISTANT]: Another process is using port 8000. Run: lsof -i :8000 to find it, then kill it or use a different port with --port 8001\n[USER]: Found it was an old Docker container. Killed it, works now.\n\nWrap in <summary></summary> tags."}
        ],
        "max_tokens": 500,
        "temperature": 0.1
    }' 2>&1)

echo ""
echo "   Summary output:"
echo "$SUMMARY_RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
text = d['choices'][0]['message']['content']
print('   ---')
print(text)
print('   ---')
tokens = d.get('usage', {})
print(f'   Tokens: {tokens.get(\"prompt_tokens\", \"?\")} in, {tokens.get(\"completion_tokens\", \"?\")} out')

# Quality checks
has_summary_tags = '<summary>' in text.lower()
has_structure = any(w in text.lower() for w in ['task', 'step', 'issue', 'resolved', 'fix'])
is_concise = len(text) < 1500

print()
print(f'   Quality checks:')
print(f'   - Has <summary> tags: {\"✅\" if has_summary_tags else \"⚠️ Missing\"}')
print(f'   - Has structure: {\"✅\" if has_structure else \"⚠️ Weak\"}')
print(f'   - Concise (<1500 chars): {\"✅\" if is_concise else \"⚠️ Too long\"}')

if has_summary_tags and has_structure and is_concise:
    print()
    print('   🎉 Model produces good summaries — ready for ctxpact!')
elif has_structure:
    print()
    print('   ⚠️  Decent but not perfect. Consider using a cloud fallback for summarization.')
else:
    print()
    print('   ❌  Poor summary quality. Use stage2_summarize.model to set a cloud model for summaries.')
" 2>/dev/null || echo "   ❌ Failed to parse summary response"

# Step 4: Check Forge (optional)
echo ""
echo "4. Checking Forge installation..."
if command -v forge &> /dev/null; then
    echo "   ✅ Forge is installed ($(forge --version 2>/dev/null || echo 'version unknown'))"
    echo ""
    echo "   To test with Forge:"
    echo "   export OPENAI_URL=${VLLM_URL}"
    echo "   export OPENAI_API_KEY=dummy"
    echo "   forge"
else
    echo "   ⚠️  Forge not installed"
    echo "   Install: npm install -g @antinomyhq/forge"
fi

echo ""
echo "=== Validation complete ==="
