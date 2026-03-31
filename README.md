# ctxpact

**Resilient context compaction proxy for local LLM inference.**

ctxpact is a lightweight, OpenAI-compatible proxy that handles oversized inputs. When a 110k-token document hits a 16k-token model, ctxpact extracts the most relevant ~12k tokens to answer accurately — achieving **100% on our reading comprehension benchmark** with the right model+strategy.

Drop it in front of any llama.cpp / Ollama / vLLM / vLLM-mlx server. Works with any agentic framework ( OpenClaw, Hermes, etc.) that speaks the OpenAI API.

## Why ctxpact?

Local LLMs have small context windows. Real-world agentic use cases send huge payloads — full codebases, long documents, multi-session conversation histories. Current options: truncate (lose data), chunk+summarize (slow, lossy), or cloud APIs (expensive, not private).

ctxpact sits between your agent and your local LLM:

```
Agent (OpenClaw, Hermes, etc.)
  │
  ▼
ctxpact proxy (localhost:8000)       ◄── OpenAI-compatible, drop-in
  │
  ├── Stage 1: DCP — dedup tool calls, strip stale writes, truncate errors
  ├── Stage 2: Summarize — evict old context, keep recent turns
  ├── Stage 3: Extract — 16 strategies to pull relevant content
  │
  ▼
Local LLM (llama-server / Ollama / vLLM)
```

**No API keys. No cloud. Everything runs on your hardware.**

### Design Alternatives Considered

Three possible architectures :

```
Design A: Standalone Proxy (chosen)          Design B: LiteLLM Plugin
┌──────────┐                                 ┌──────────┐
│  Agent   │                                 │  Agent   │
└────┬─────┘                                 └────┬─────┘
     │                                            │
     ▼                                            ▼
┌──────────────┐                             ┌──────────────┐
│   ctxpact    │ ◄── Full control            │   LiteLLM    │
│   FastAPI    │     over compaction          │   + plugin   │ ◄── Callback hooks only
│   ~11k LOC   │     lifecycle               │   ~200 LOC   │     limited lifecycle control
└────┬─────────┘                             └────┬─────────┘
     │                                            │
     ▼                                            ▼
┌──────────┐                                 ┌──────────┐
│   LLM    │                                 │   LLM    │
└──────────┘                                 └──────────┘


Design C: Sidecar
┌──────────┐    ┌───────────┐
│  Agent   │───▶│  ctxpact  │ ◄── Separate process
└────┬─────┘    │  sidecar  │     IPC overhead
     │          └─────┬─────┘
     ▼                │
┌──────────┐          │
│   LLM    │◀─────────┘
└──────────┘
```

| | A: Standalone Proxy | B: LiteLLM Plugin | C: Sidecar |
|---|---|---|---|
| **Control** | Full (own pipeline) | Callback hooks only | Full |
| **Multi-stage compaction** | Yes (DCP → summarize → extract) | No (single hook point) | Yes |
| **Mid-pipeline LLM calls** | Yes (readagent, rlm) | No | Yes |
| **Dependencies** | FastAPI + httpx | LiteLLM (~50 deps) | FastAPI + IPC |
| **Complexity** | ~11k LOC | ~200 LOC plugin | ~13k LOC |
| **Ship time** | 1-2 weeks | 1-2 days | 2-3 weeks |

**We chose A** — the breakthrough strategies (`readagent`, `rlm`) need mid-pipeline LLM calls that LiteLLM's callback system can't support. Full lifecycle control was worth the extra code.

## Quick Start

```bash
# Install
git clone https://github.com/user/ctxpact && cd ctxpact
pip install -e .

# Start your LLM backend (example: llama-server)
llama-server -m Qwen3.5-9B-Q8_0.gguf \
  --host 0.0.0.0 --port 8080 --ctx-size 16384 --jinja -ngl 99

# Start ctxpact
python -m ctxpact.server --config config.yaml --local --strategy readagent

# Use it — same API as your LLM, just different port
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-9B-Q8_0.gguf", "messages": [{"role": "user", "content": "...your 100k token message..."}]}'
```

## Features

- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/models`, streaming support
- **16 extraction strategies** — from zero-LLM-call heuristics to dual-LLM retrieval pipelines
- **3-stage compaction pipeline** — DCP pruning → summarization → intelligent extraction
- **Provider failover** — circuit breaker + health monitoring across multiple backends
- **Session tracking** — per-conversation state with compaction audit trail
- **GOG context isolation** — optional graph-of-graphs for codebase-aware compaction
- **~11k lines of Python** — lightweight, no heavy frameworks

![Arch Diagram](bench/CTXPACTARCH.drawio.svg)

## Benchmark Results

Tested on Frankenstein (110k tokens → 12k budget, 8 reading comprehension questions) and LoCoMo-MC10 (multi-session conversation QA, 10-choice, 20 questions):

| Configuration | Frankenstein | LoCoMo-MC10 | Combined |
|---------------|-------------|-------------|----------|
| **readagent + Qwen3.5-9B** | **8/8 (100%)** | **15/20 (75%)** | **87.5%** |
| rlm + Qwen3.5-9B | 8/8 (100%) | 12/20 (60%) | 80.0% |
| embed + Qwen3.5-9B | 7/8 (87.5%) | 14/20 (70%) | 78.8% |
| agentic + LFM2-8B-A1B | 6.2/8 avg (78%) | 5/20 (25%) | 51.3% |
| Random baseline (LoCoMo) | — | 2/20 (10%) | — |

**`readagent` and `rlm` are the breakthrough strategies.** Both achieve **100% on Frankenstein** — deterministic, repeatable perfect scores (readagent verified across 3 consecutive runs with 0% variance). They are the only strategies that solve Q4 ("Where does Clerval get murdered?"), the hardest query requiring both precise retrieval of a sparse signal AND faithful in-context reading over parametric knowledge. readagent leads on LoCoMo-MC10 (75% vs rlm's 60%), making it the recommended default for cross-domain use.

**Model choice matters more than strategy choice.** Switching from LFM2 to Qwen3.5 improved every strategy by +25-50pp. NR-MMLU (reading comprehension) is the best predictor of context engineering performance. See [BENCHMARKS.md](BENCHMARKS.md) for full analysis with visualizations.

## Extraction Strategies

### Zero LLM Calls (Pure Retrieval)

| Strategy | Description |
|----------|-------------|
| `header` | Section previews (600 chars each) + recent full sections |
| `autosearch` | Heuristic keyword extraction → word-level grep → assemble |
| `embed` | ChromaDB embedding retrieval (all-MiniLM-L6-v2) |
| `compress` | LLMLingua token-level compression (GPT-2 on CPU) |
| `adaptive` | Query router: structural → header, detail → embed |
| `icl` | In-context learning: select turns by similarity + recency decay |

### Single LLM Call (Search Term Generation)

| Strategy | Description |
|----------|-------------|
| `rlm` | LLM generates search terms → IDF-weighted grep → multi-pass assembly |
| `rlm_v2` | Word-level matching, IDF ranking, no summary bloat |
| `rlm_v4` | Runs embed+header+rlm_v2 in parallel, picks best by quality score |

### Dual LLM Calls (Best Accuracy)

| Strategy | Description |
|----------|-------------|
| `readagent` | Embed + BM25 + RRF fusion → dual LLM term expansion → position-aware excerpting |
| `agentic` | readagent + proper noun weighting (3x) + first/last match snippets |

### Multi-Turn (N LLM Calls)

| Strategy | Description |
|----------|-------------|
| `rlm_v3` | DSPy: model writes Python to search the book (sandboxed Deno) |
| `rlm_v5` | Enhanced tools: stats, search, regex, read, done |
| `toolcall` | Multi-turn tool-calling loop: search/read/done |
| `chunking` | Map-reduce: summarize each chunk, answer from summaries |
| `subagent` | Hierarchical query decomposition |



## Visualizations

### Accuracy by Strategy — Side-by-Side Model Comparison

```
Score         ██ = LFM2-8B     ░░ = Qwen3.5-9B
(out of 8)
  8 |                    ░░          ░░
    |                    ░░          ░░
  7 |          ░░        ░░          ░░
    |          ░░        ░░          ░░
  6 |  ░░      ░░    ██  ░░    ██    ░░
    |  ░░      ░░    ██  ░░    ██    ░░
  5 |  ░░  ██  ░░    ██  ░░    ██    ░░
    |  ░░  ██  ░░    ██  ░░    ██    ░░
  4 |  ██  ██  ██    ██  ░░    ██    ░░
    |  ██  ██  ██    ██  ░░    ██    ░░
  3 |  ██  ██  ██    ██  ░░    ██    ░░
  2 |  ██  ██  ██    ██  ░░    ██    ░░
  1 |  ██  ██  ██    ██  ░░    ██    ░░
    +---------------------------------------------
      header  embed   rlm   agentic  readagent
                       ↑               ↑
                     8/8!             8/8!
```

### Per-Query Pass Heatmap — LFM2 vs Qwen3.5

```
                 LFM2-8B-A1B                          Qwen3.5-9B
          hdr aut emb icl adt rlm rv2 rv4       hdr aut emb icl adt rlm rv2 rv4 rda
    Q1  |  o   o   o   o   o   o   o   o  |    |  o   o   o   o   o   o   o   o   o  |
    Q2  |  o   o   o   o   o   o   ~   ~  |    |  o   o   o   o   o   o   o   o   o  |
    Q3  |  o   o   o   o   o   o   o   o  |    |  o   o   o   o   o   o   o   o   o  |
    Q4  |  ~   ~   ~   ~   ~   ~   x   ~  |    |  x   x   x   x   x   o   x   x   o  |
    Q5  |  x   x   x   x   o   x   o   o  |    |  o   o   o   o   o   o   o   o   o  |
    Q6  |  x   ~   ~   ~   ~   ~   x   ~  |    |  x   o   o   o   x   o   o   o   o  |
    Q7  |  o   o   o   o   o   o   o   o  |    |  o   o   o   o   o   o   o   o   o  |
    Q8  |  x   x   x   x   x   o   x   x  |    |  o   o   o   o   o   o   o   o   o  |
        +----------------------------------+    +--------------------------------------+
         o = PASS   ~ = partial (0.5)   x = FAIL          rda = readagent
```

### Accuracy vs Latency — Pareto Frontier

```
Score
(out of 8)
  8 |                                     * readagent+Q35   * rlm+Q35
    |
  7 |            * embed+Q35   * autosearch+Q35  * rlm_v2+Q35
    |              * icl+Q35     * rlm_v4+Q35
  6 |  * header+Q35                                    * agentic+LFM2 (avg)
    |  * adaptive+Q35         * adaptive+LFM2
    |  * rlm_v4+LFM2
  5 |  * embed+LFM2  * rlm+LFM2
    |  * icl+LFM2   * autosearch+LFM2  * toolcall+LFM2
    |  * header+LFM2  * rlm_v3+LFM2
  4 |  * rlm_v2+LFM2  * rlm_v5+LFM2
    |
  3 |
    |
  2 |
    |                                                         * chunking+LFM2
  1 |
    +----+--------+--------+--------+--------+--------+--------+------
        20s      40s      60s      80s     100s     120s     140s    Latency

    PARETO FRONTIER: header+Q35 (6/8,91s) → embed+Q35 (7/8,91s) → rlm+Q35 (8/8,110s)
```

### Model Accuracy Distribution

```
    LFM2-8B-A1B (12 strategies)            Qwen3.5-9B (8 strategies)
    ┌─────────────────────────┐            ┌─────────────────────────┐
    │                         │            │                         │
8/8 │                         │        8/8 │ ██ (2: rlm, readagent)  │
    │                         │            │                         │
7/8 │                         │        7/8 │ ██████████ (5: embed,   │
    │                         │            │   icl, auto, rv2, rv4)  │
6/8 │ ████ (2: rlm, adaptive) │        6/8 │ ████ (2: header,       │
    │                         │            │   adaptive)             │
5/8 │ ████████████ (5: embed, │        5/8 │                         │
    │   icl, auto, rv4, tool) │            │                         │
4/8 │ ████████ (4: hdr, rv2,  │        4/8 │                         │
    │   rv3, rv5)             │            │                         │
    │                         │            │                         │
1/8 │ ██ (1: chunking)        │            │                         │
    └─────────────────────────┘            └─────────────────────────┘
    Mean: 5.06  Median: 5.0               Mean: 6.88  Median: 7.0
    SD:   0.55  Range: 1.5-6.0            SD:   0.64  Range: 6.0-8.0
```

### LLM Calls vs Accuracy — Diminishing Returns

```
Avg Score
(out of 8)
  8 |                    * (Qwen3.5)
    |                   /
  7 |              *   /
    |             / \ /
  6 |        *   /   *-------- Qwen3.5 (projected, N-call unreliable)
    |       / \ /   /
  5 |  *   /   *   /
    |   \ /       /
  4 |    *-------*------------ LFM2
    |
  3 |
    +----+-------+-------+-------
        0       1       2       N       LLM Extraction Calls

    ─── LFM2:   4.9 → 5.3 → 6.2 → 4.7   (N-call = worse than 0-call!)
    ─── Qwen3.5: 6.6 → 7.3 → 8.0 → N/A   (diminishing but still positive)
```

### Q4 (Ireland) — The Litmus Test

```
    "Where does Clerval get murdered?"   Correct answer: Ireland

    LFM2 responses (all 12 strategies):
    ┌────────────────────────────────────────────────────────┐
    │  "Geneva, Switzerland"  ████████████████████  (8x)     │
    │  "a forest near Geneva" ██████  (3x)                   │
    │  "Dover, England"       ██  (1x)                       │
    │  "Ireland"                                     (0x)    │
    └────────────────────────────────────────────────────────┘
    → LFM2 ALWAYS defaults to parametric knowledge (Switzerland)

    Qwen3.5 responses (8 strategies):
    ┌────────────────────────────────────────────────────────┐
    │  "Geneva, Switzerland"  ██████████  (4x)               │
    │  "Dover, England"       ████  (2x)                     │
    │  "Ireland"              ████  (2x: rlm, readagent)     │
    └────────────────────────────────────────────────────────┘
    → Qwen3.5 answers correctly WHEN the right section is retrieved

    Diagnosis: LFM2 fails at MODEL level. Qwen3.5 fails at RETRIEVAL level.
```

### Stability: Score Variance Across Runs

```
    LFM2 Agentic (4 runs)               Qwen3.5 ReadAgent (3 runs)
    Score: 6.5  8.0  5.0  5.5           Score: 8.0  8.0  8.0
           ─────────────────                   ─────────────
    8  |        *                        8  |   *     *     *
    7  |                                 7  |
    6  |   *                             6  |
    5  |             *    *              5  |
    4  |                                 4  |
       +---+----+----+----+                +---+-----+-----+
          r1   r2   r3   r4                  r1    r2    r3

    Mean: 6.25    SD: 1.31               Mean: 8.00    SD: 0.00
    Range: 5.0-8.0 (3.0 spread)          Range: 8.0-8.0 (0 spread)
    CV: 21.0%                            CV: 0.0%
```

### LoCoMo-MC10 — Cross-Domain Results

```
Accuracy (%)
  80 |                                        ██
     |                                  ██    ██
  70 |                            ██    ██    ██
     |                            ██    ██    ██
  60 |                      ██    ██    ██    ██
     |                      ██    ██    ██    ██
  50 |                      ██    ██    ██    ██
     |                      ██    ██    ██    ██
  40 |                      ██    ██    ██    ██
     |                      ██    ██    ██    ██
  30 |                      ██    ██    ██    ██
     |                ██    ██    ██    ██    ██
  20 |                ██    ██    ██    ██    ██
     |                ██    ██    ██    ██    ██
  10 |  ░░░░░░░░░░    ██    ██    ██    ██    ██
     +--------+-------+------+------+------+------
             random   LFM2  rlm   embed  read-
             10%     agent  Q35   Q35    agent
                      25%   60%   70%    Q35
                                         75%
```

### LoCoMo-MC10 — Per-Type Heatmap

```
                  LFM2    Qwen3.5-9B
                agentic   rlm  embed readagent
  adversarial  |   x   |  ~    o      o    |    Qwen3.5: 67-75% vs LFM2: 0%
  multi_hop    |   x   |  x    o      ~    |    Hardest for all strategies
  open_domain  |   ~   |  o    o      o    |    100% across all Qwen3.5 strategies
  single_hop   |   o   |  o    ~      o    |    Mixed results
  temporal     |   x   |  ~    ~      ~    |    50% ceiling for all
               +-------+--------------------+
  o = 75-100%   ~ = 50%   x = 0-25%
```



## Configuration

```yaml
# config.yaml
server:
  host: "0.0.0.0"
  port: 8000

providers:
  - name: "local"
    url: "http://localhost:8080/v1"     # Your llama-server
    model: "Qwen3.5-9B-Q8_0.gguf"
    api_key: "dummy"
    max_context: 16384
    priority: 1
    timeout_seconds: 180

circuit_breaker:
  failure_threshold: 3
  recovery_timeout_seconds: 30

compaction:
  enabled: true
  triggers:
    token_ratio: 0.70                   # Compact when input > 70% of context

  stage1_dcp:                           # Dynamic Context Pruning
    dedup_tool_calls: true
    strip_superseded_writes: true
    truncate_errors: true

  stage2_summarize:                     # LLM-based summarization
    max_summary_tokens: 2000
    retention_window: 6

  oversized:
    strategy: "readagent"               # Which extraction strategy to use
```

## Architecture

```
ctxpact/ (11,326 lines)
├── ctxpact/
│   ├── server.py              # FastAPI server, 3 endpoints, streaming
│   ├── config.py              # Pydantic config models
│   ├── compaction/
│   │   ├── engine.py          # 3-stage compaction orchestrator
│   │   ├── pruner.py          # Stage 1: DCP (dedup, strip, truncate)
│   │   ├── summarizer.py      # Stage 2: LLM summarization
│   │   ├── rlm_extractor.py   # Stage 3: 16 extraction strategies (5,736 lines)
│   │   ├── book.py            # ConversationBook + BookSection
│   │   ├── tokens.py          # Tiktoken-based token counting
│   │   ├── chunker.py         # Map-reduce chunking fallback
│   │   └── prompts.py         # LLM prompt templates
│   ├── routing/
│   │   ├── router.py          # Provider selection + failover
│   │   ├── client.py          # OpenAI-compatible async HTTP client
│   │   ├── circuit_breaker.py # Closed → Open → Half-Open state machine
│   │   └── health.py          # Async health monitoring
│   ├── isolation/             # Optional GOG context isolation
│   │   ├── isolator.py        # Graph-of-Graphs orchestrator
│   │   ├── graph_builder.py   # Import dependency graph construction
│   │   ├── seed_finder.py     # Extract identifiers from prompts
│   │   └── ...parsers         # Python AST + TypeScript tree-sitter
│   └── session/
│       ├── models.py          # Session, Message, CompactionEvent
│       └── store.py           # Memory + SQLite backends
├── tests/                     # 5 test files, 660 lines
├── bench/                     # Benchmark suite
│   ├── benchmark_strategies.py    # Frankenstein all-strategy benchmark
│   ├── benchmark_readagent_direct.py  # Direct ReadAgent benchmark
│   ├── benchmark_locomo.py        # LoCoMo-MC10 conversation QA
│   ├── benchmark_longmemeval.py   # LongMemEval free-text QA
│   ├── benchmark_report.py        # Plot/report generator
│   ├── data/frankenstein.txt      # Test document (Project Gutenberg #84)
│   └── results/                   # Raw JSON evidence (12 key runs)
├── config.yaml                # Example configuration
├── pyproject.toml             # Dependencies + metadata
├── Dockerfile                 # Python 3.11 slim image
└── Makefile                   # dev tasks: test, lint, run, docker
```

## API

ctxpact exposes three endpoints:

```
GET  /health                    # Provider status + compaction config
GET  /v1/models                 # OpenAI-compatible model listing
POST /v1/chat/completions       # Main endpoint (streaming supported)
```

Session tracking via `X-Session-ID` header. If not provided, auto-generated.

## Research Background

ctxpact implements ideas from several papers:

- **ReadAgent** (Lee et al., ICML 2024) — Gist memory for global context; our `readagent` strategy uses gist-inspired structure overviews
- **RLM** (Zhang et al., arXiv:2512.24601) — Context as an interactive environment; our `rlm_v3` and `rlm_v5` strategies use programmatic exploration
- **SRLM** (Alizadeh et al., arXiv:2603.15653) — Multi-candidate selection without LLM judge; our `rlm_v4` runs parallel strategies and picks best by quality score
- **LLMLingua** (Jiang et al.) — Token-level compression; our `compress` strategy uses GPT-2 for query-aware pruning

## Dependencies

**Core:**
```
fastapi, uvicorn, httpx, sse-starlette
pydantic, pyyaml, tiktoken, aiosqlite
```

**Strategy-specific (optional):**
```
chromadb, sentence-transformers    # embed, icl, readagent strategies
llmlingua, nltk                    # compress strategy
dspy                               # rlm_v3 strategy
```

## Development

```bash
make test          # Run pytest
make lint          # Run ruff
make run           # Start server with default config
make docker-build  # Build container
```

## Benchmarking

The `bench/` directory contains a full benchmark suite for evaluating extraction strategies.

```bash
# Frankenstein benchmark — all strategies, 8 reading comprehension questions
# Requires ctxpact server + llama-server running
python bench/benchmark_strategies.py --label my-run

# Direct ReadAgent benchmark (no ctxpact server needed, calls llama-server directly)
python bench/benchmark_readagent_direct.py readagent-test

# LoCoMo-MC10 — multi-session conversation QA (10-choice, 20 questions)
python bench/benchmark_locomo.py --strategy readagent --n 20

# LongMemEval — free-text conversation QA (500 questions)
python bench/benchmark_longmemeval.py --n 20

# Generate plots from results
python bench/benchmark_report.py
```

Environment variables: `BENCH_MODEL`, `BENCH_BOOK`, `BENCH_PROVIDER`, `LOCOMO_DATA`, `LONGMEMEVAL_DATA`.

Results are saved to `bench/results/`. Pre-computed results from our evaluation (12 key runs across 2 models) are included. See [BENCHMARKS.md](BENCHMARKS.md) for full analysis with visualizations.

## Model Recommendations

From our benchmark of 331 GGUF models on Mac Mini M4 16GB:

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| **Qwen3.5-9B (Q8_0)** | 11 tok/s | 8/8 Frankenstein, 75% LoCoMo | **Best accuracy** |
| LFM2-8B-A1B (Q8_0) | 50 tok/s | 6.2/8 Frankenstein, 25% LoCoMo | Fast inference |

**Key insight:** NR-MMLU (reading comprehension) is the best predictor of context engineering performance. Qwen3.5's 65% NR-MMLU vs LFM2's 47% directly maps to accuracy improvements.

## License

MIT

## Acknowledgments

Built for the local LLM community. Tested on Mac Mini M4 16GB with llama.cpp. If you're running small models on constrained hardware and hitting context limits, ctxpact is for you.
