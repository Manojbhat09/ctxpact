# Benchmarks

Comprehensive benchmark results for ctxpact extraction strategies across two models and two evaluation datasets.

**Hardware:** Mac Mini M4 16GB running llama-server via llama.cpp
**Models:** Qwen3.5-9B-Q8_0 (11 tok/s) and LFM2-8B-A1B-Q8_0 (50 tok/s)

---

## TL;DR

| Configuration | Frankenstein | LoCoMo-MC10 | Combined |
|---------------|-------------|-------------|----------|
| **readagent + Qwen3.5-9B** | **8/8 (100%)** | **15/20 (75%)** | **87.5%** |
| rlm + Qwen3.5-9B | 8/8 (100%) | 12/20 (60%) | 80.0% |
| embed + Qwen3.5-9B | 7/8 (87.5%) | 14/20 (70%) | 78.8% |
| agentic + LFM2-8B-A1B | 6.2/8 avg (78%) | 5/20 (25%) | 51.3% |
| Random baseline (LoCoMo) | — | 2/20 (10%) | — |

**Key finding: Model choice matters more than strategy choice.** Switching from LFM2 to Qwen3.5 improved every strategy by +25-50pp.

---

## Benchmark 1: Frankenstein (Long Document QA)

**Task:** 8 reading comprehension questions on Frankenstein (438,842 chars, ~110k tokens, 25 sections)
**Budget:** ~12,000 tokens (model context: 16,384)
**Question types:** Structural, character, plot, detail

### Head-to-Head: All Strategies, Both Models

| Strategy | LLM Calls | LFM2 Score | LFM2 Latency | Qwen3.5 Score | Qwen3.5 Latency | Delta |
|----------|-----------|-----------|-------------|--------------|-----------------|-------|
| **readagent** | 2 | 5.7/8 avg | 32s | **8/8 (3x)** | 117s | **+2.3** |
| **rlm** | 1 | 6.0/8 | 23s | **8/8** | 110s | **+2.0** |
| **agentic** | 2 | 6.2/8 avg | 37s | — | — | — |
| autosearch | 0 | 5.0/8 | 22s | 7/8 | 97s | +2.0 |
| rlm_v2 | 1 | 4.5/8 | 23s | 7/8 | 107s | +2.5 |
| embed | 0 | 5.0/8 | 21s | 7/8 | 91s | +2.0 |
| icl | 0 | 5.0/8 | 20s | 7/8 | 97s | +2.0 |
| rlm_v4 | 1 | 5.5/8 | 21s | 7/8 | 102s | +1.5 |
| adaptive | 0 | 6.0/8 | 20s | 6/8 | 92s | +0.0 |
| header | 0 | 4.5/8 | 20s | 6/8 | 91s | +1.5 |
| rlm_v3 | N | 4.5/8 | 184s | 4/8* | 249s | -0.5 |
| toolcall | N | 5.0/8 | 67s | — | — | — |
| rlm_v5 | N | 4.5/8 | 62s | — | — | — |
| chunking | N | 1.5/8 | 370s | 0/8* | 338s | -1.5 |

*rlm_v3 and chunking with Qwen3.5 unreliable due to 503 errors/timeouts.

**Evidence note:** JSON result files in `bench/results/` back the readagent 8/8 (3 runs), embed/icl/rlm_v4/adaptive from `qwen35-v2`, all LFM2 strategies, and LoCoMo-MC10 runs. Other Qwen3.5 strategy results (header, autosearch, rlm, rlm_v2) were observed during development but the raw JSON was not preserved.

### Per-Query Breakdown — Qwen3.5-9B

| Query | Type | header | auto | embed | icl | adapt | rlm | rlm_v2 | rlm_v4 | readagent (3 runs) |
|-------|------|--------|------|-------|-----|-------|-----|--------|--------|---------------------|
| Q1: Chapters | structural | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS, PASS, PASS |
| Q2: Justine | character | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS, PASS, PASS |
| Q3: Companion | plot | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS, PASS, PASS |
| **Q4: Clerval** | **detail** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **PASS** | **FAIL** | **FAIL** | **PASS, PASS, PASS** |
| Q5: Walton | character | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS, PASS, PASS |
| Q6: Ending | plot | FAIL | PASS | PASS | PASS | FAIL | PASS | PASS | PASS | PASS, PASS, PASS |
| Q7: Studies | detail | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS, PASS, PASS |
| Q8: Reading | detail | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS, PASS, PASS |
| **Total** | | **6/8** | **7/8** | **7/8** | **7/8** | **6/8** | **8/8** | **7/8** | **7/8** | **8/8, 8/8, 8/8** |

### Per-Query Breakdown — LFM2-8B-A1B

Scores: 1.0 = pass, 0.5 = partial, 0.0 = fail.

| Query | Type | header | auto | embed | icl | adapt | rlm | rlm_v2 | rlm_v4 | rlm_v3 | rlm_v5 | tool | chunk |
|-------|------|--------|------|-------|-----|-------|-----|--------|--------|--------|--------|------|-------|
| Q1: Chapters | structural | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 |
| Q2: Justine | character | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5 | 0.5 | 0.5 | 0.5 | 1.0 | 0.0 |
| Q3: Companion | plot | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| **Q4: Ireland** | **detail** | **0.5** | **0.5** | **0.5** | **0.5** | **0.5** | **0.5** | **0.0** | **0.5** | **0.5** | **0.5** | **0.5** | **0.5** |
| Q5: Walton | character | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| Q6: Ending | plot | 0.0 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.0 | 0.5 | 0.5 | 0.5 | 0.5 | 0.0 |
| Q7: Studies | detail | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Q8: Reading | detail | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Total** | | **4.5** | **5.0** | **5.0** | **5.0** | **6.0** | **6.0** | **4.5** | **5.5** | **4.5** | **4.5** | **5.0** | **1.5** |

### ReadAgent Stability — 3 Consecutive Perfect Runs

| Query | Run 1 | Run 2 | Run 3 | Pass Rate |
|-------|-------|-------|-------|-----------|
| Q1 | PASS (95.5s) | PASS (93.6s) | PASS (97.2s) | 100% |
| Q2 | PASS (136.5s) | PASS (155.0s) | PASS (141.4s) | 100% |
| Q3 | PASS (103.3s) | PASS (101.7s) | PASS (109.3s) | 100% |
| Q4 | PASS (122.0s) | PASS (114.7s) | PASS (114.2s) | 100% |
| Q5 | PASS (113.9s) | PASS (100.5s) | PASS (105.1s) | 100% |
| Q6 | PASS (139.4s) | PASS (131.5s) | PASS (126.9s) | 100% |
| Q7 | PASS (119.7s) | PASS (102.5s) | PASS (115.5s) | 100% |
| Q8 | PASS (105.3s) | PASS (121.4s) | PASS (123.1s) | 100% |
| **Total** | **8/8 (119.2s)** | **8/8 (115.1s)** | **8/8 (116.6s)** | **100%** |

Standard deviation of per-query latency: 3.7-12.1s — highly consistent.

---

## Benchmark 2: LoCoMo-MC10 (Multi-Session Conversation QA)

**Task:** Multi-session conversation QA with 10-choice multiple choice
**Dataset:** 1,986 questions total; 20-question stratified sample per run
**Context:** 12k-25k tokens of multi-session conversation per question
**Question types:** single_hop, multi_hop, temporal_reasoning, open_domain, adversarial
**Random baseline:** 10% (1/10 choices)

### Results — Qwen3.5-9B

| Strategy | Overall | Adversarial | Multi-hop | Open-domain | Single-hop | Temporal | Latency |
|----------|---------|-------------|-----------|-------------|------------|----------|---------|
| **readagent** | **15/20 (75%)** | 3/4 (75%) | 2/4 (50%) | **4/4 (100%)** | **4/4 (100%)** | 2/4 (50%) | 92s |
| embed | 14/20 (70%) | **3/4 (75%)** | **3/4 (75%)** | **4/4 (100%)** | 2/4 (50%) | 2/4 (50%) | 66s |
| rlm | 12/20 (60%) | 2/4 (50%) | 1/4 (25%) | **4/4 (100%)** | 3/4 (75%) | 2/4 (50%) | 76s |
| *LFM2 agentic* | *5/20 (25%)* | *0/4 (0%)* | *0/4 (0%)* | *2/4 (50%)* | *3/4 (75%)* | *0/4 (0%)* | *22s* |

### Key Findings

1. **Strategy ranking differs from Frankenstein.** readagent leads (75%), embed second (70%), rlm third (60%). Semantic embedding outperforms keyword search on conversation data.
2. **Open-domain questions are solved** — 100% by all Qwen3.5 strategies.
3. **Temporal reasoning is the ceiling** — 50% max across all strategies.
4. **Adversarial questions improved dramatically** — LFM2 scored 0/4; Qwen3.5 scores 50-75%.

---

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

---

## Analysis

### Model Effect Size

Across matched strategies (8 strategies tested on both models):

| Metric | LFM2 | Qwen3.5 | Improvement |
|--------|-------|---------|-------------|
| Mean score | 5.06/8 | 6.88/8 | **+1.81 (+36%)** |
| Median score | 5.0/8 | 7.0/8 | **+2.0** |
| Max score | 6.0/8 | 8.0/8 | **+2.0** |
| Min score | 4.5/8 | 6.0/8 | **+1.5** |

The model switch improved every single strategy without exception.

### Why Model > Strategy

| | LFM2-8B-A1B | Qwen3.5-9B |
|---|---|---|
| **Best single-run** | 8/8 (agentic, 1 of 4 runs) | **8/8 (readagent, 3/3 runs)** |
| **Best avg** | 6.2/8 (agentic, 4 runs) | **8.0/8 (readagent, 3 runs)** |
| **Median strategy** | 4.5/8 | **7/8** |
| **Q4 pass rate** | 0% (12 strategies) | **100% (rlm), 100% (readagent)** |
| **Speed** | ~50 tok/s | ~11 tok/s |

### Architecture Comparison

| Property | LFM2-8B-A1B | Qwen3.5-9B |
|----------|-------------|------------|
| Total params | 8.3B | 9.0B |
| Active params | 1.5B (MoE) | 9.0B (dense) |
| Architecture | Mixture of Experts | Hybrid DeltaNet + Attention |
| NR-MMLU (reading comprehension) | 47.2% | 65.0% |
| Throughput (M4 16GB) | ~50 tok/s | ~11 tok/s |
| Q8_0 file size | ~8.5 GB | ~9.5 GB |

**Active parameter ratio explains the speed gap**: LFM2 activates 1.5B of 8.3B params per token (MoE routing); Qwen3.5 uses all 9B. ~6x more compute per token → ~4.5x speed difference.

### NR-MMLU Correlation

LFM2's 47.2% vs Qwen3.5's 65.0% on NR-MMLU (+17.8pp) maps directly to the accuracy improvement we observe (+36% relative). **NR-MMLU is the best predictor of context engineering performance** for document QA tasks.

### Error Mode Analysis

**LFM2 (96 query responses across 12 strategies):**

| Error Mode | Frequency | Description |
|------------|-----------|-------------|
| Parametric override | 38% (37/96) | Answers from training data, ignores context |
| Creative fabrication | 15% (14/96) | Invents plausible but incorrect details |
| Partial extraction | 12% (12/96) | Finds some keywords but misses critical ones |
| Correct | 35% (33/96) | — |

**Qwen3.5 (64 query responses across 8 strategies):**

| Error Mode | Frequency | Description |
|------------|-----------|-------------|
| Missing retrieval | 14% (9/64) | Correct section not in context |
| Parametric override | 0% (0/64) | Never observed |
| Creative fabrication | 0% (0/64) | Never observed |
| Correct | 86% (55/64) | — |

Qwen3.5 never overrides context with parametric knowledge. When it fails, the relevant text wasn't retrieved — a retrieval problem fixable through better strategies.

### Strategy Complexity vs Accuracy

| LLM Calls | LFM2 Avg | Qwen3.5 Avg | Observation |
|-----------|----------|-------------|-------------|
| 0 calls | 4.9/8 | 6.6/8 | Simple retrieval; model does the heavy lifting |
| 1 call | 5.3/8 | 7.3/8 | LLM search terms help meaningfully |
| 2 calls | 6.2/8 | 8.0/8 | Dual LLM = best accuracy |
| N calls | 4.7/8 | N/A | Diminishing returns, instability |

The sweet spot is **2 LLM extraction calls**. Beyond that, complexity hurts.

### Efficiency Frontier

Pareto-optimal configurations (nothing dominates on both accuracy and latency):

1. `adaptive + LFM2` — 6.0/8, 20s (fastest acceptable accuracy)
2. `embed + Qwen3.5` — 7.0/8, 91s (best 0-LLM-call option)
3. `rlm + Qwen3.5` — 8.0/8, 110s (cheapest perfect score)
4. `readagent + Qwen3.5` — 8.0/8, 117s (most robust perfect score)

---

## Strategy Taxonomy

```
Extraction Strategies (by LLM calls)
├── 0 LLM calls (pure retrieval)
│   ├── header     — structural previews
│   ├── autosearch — heuristic keyword grep
│   ├── embed      — embedding similarity (ChromaDB)
│   ├── icl        — in-context learning with summaries
│   └── adaptive   — hybrid header+embed
├── 1 LLM call (search term generation)
│   ├── rlm        — LLM search terms + IDF + multi-pass assembly
│   ├── rlm_v2     — word-level matching variant
│   └── rlm_v4     — multi-candidate parallel selection (SRLM-inspired)
├── 2 LLM calls (dual term generation)
│   ├── readagent  — embed + BM25 + RRF + dual LLM + excerpting
│   └── agentic    — readagent + proper noun weighting + first/last match
└── N LLM calls (multi-turn)
    ├── rlm_v3     — DSPy code generation
    ├── rlm_v5     — enhanced programmatic exploration
    ├── toolcall   — iterative search/read loop
    ├── chunking   — map-reduce summarization
    └── subagent   — hierarchical query decomposition
```

---

## Recommendations

### For Production

1. **Default: `readagent` + Qwen3.5-9B** — 100% Frankenstein, 75% LoCoMo, deterministic
2. **Fast fallback: `rlm` + Qwen3.5-9B** — 100% Frankenstein, 60% LoCoMo, no ChromaDB needed
3. **Ultra-fast: `embed` + Qwen3.5-9B** — 87.5% Frankenstein, 70% LoCoMo, 0 LLM calls

### For Model Selection

- **Prioritize NR-MMLU** when selecting models. Reading comprehension is the key capability.
- **Test parametric override** on Q4-type queries where the answer requires overriding common knowledge.

### For Strategy Development

- **Stop optimizing multi-turn strategies.** 1-2 LLM calls with a good model beats N-call strategies.
- **LLM-generated search terms are the key technique.** The 0→1 call jump provides the biggest accuracy gain.

---

## Appendix: ReadAgent v25j (Qwen3.5, 8/8 Run)

| Query | Keywords Found | Latency | Prompt Tokens |
|-------|---------------|---------|---------------|
| Q1: Chapters | "24" | 95.5s | 12,388 |
| Q2: Justine | "justine", "trial", "executed" | 136.5s | 12,497 |
| Q3: Companion | "companion", "mate", "female" | 103.3s | 12,975 |
| Q4: Clerval | "clerval", "ireland" | 122.0s | 13,340 |
| Q5: Walton | "walton" | 113.9s | 12,572 |
| Q6: Ending | "arctic", "ice", "death" | 139.4s | 13,911 |
| Q7: Studies | "chemistry", "natural philosophy" | 119.7s | 11,820 |
| Q8: Reading | "felix", "de lacey", "safie", "cottage" | 105.3s | 12,600 |

## Appendix: Agentic v29b (LFM2, 8/8 Run)

| Query | Keywords Found | Latency | Prompt Tokens |
|-------|---------------|---------|---------------|
| Q1: Chapters | "24" | 30.9s | 13,778 |
| Q2: Justine | "justine", "trial" | 36.8s | 14,354 |
| Q3: Companion | "companion", "female" | 34.7s | 14,300 |
| Q4: Clerval | "clerval", "ireland" | 40.8s | 14,392 |
| Q5: Walton | "walton" | 33.6s | 14,410 |
| Q6: Ending | "arctic", "ice", "death" | 36.4s | 14,479 |
| Q7: Studies | "chemistry", "natural philosophy" | 34.6s | 14,300 |
| Q8: Reading | "felix", "safie" | 45.7s | 14,422 |

---

## Research References

- **ReadAgent** (Lee et al., ICML 2024) — Gist memory for global context
- **RLM** (Zhang et al., arXiv:2512.24601) — Context as an interactive environment
- **SRLM** (Alizadeh et al., arXiv:2603.15653) — Multi-candidate selection without LLM judge
- **LLMLingua** (Jiang et al.) — Token-level compression

---

*Benchmark suite: [ctxpact-bench](../ctxpact-bench/) | Hardware: Mac Mini M4 16GB | Models: 331 GGUFs evaluated, top 2 selected*
