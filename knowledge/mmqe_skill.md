# Skill: Multi-Teacher Dataset Distillation (MTDD)

## Purpose

Distill a large-scale, high-quality labeled dataset by aggregating scores from multiple API-based teacher models, then use that dataset to train a compact, fast student model that matches or exceeds teacher quality.

**When to apply this skill:**
- You need to label millions of samples across many dimensions
- No single teacher model is best at all dimensions or sub-groups (e.g., languages)
- You want a small, fast student model for production deployment
- You have a human-annotated gold corpus for teacher calibration

**When NOT to apply:**
- Single-dimension binary classification (use a single teacher + threshold)
- Dataset is small enough for human annotation (< 50K samples)
- No variation in teacher specialization across sub-groups

---

## Core Concept

> Different teacher models specialize in different sub-groups and dimensions. A confidence matrix indexed by **(sub-group × dimension × teacher)** captures who is best at what, and a softmax-weighted aggregation (CSA) combines their scores proportionally to their measured competence.

### The Confidence Matrix

The central data structure of MTDD. For each (sub-group, dimension) pair, record how well each teacher agrees with human annotations:

```
C[language][model][dimension] = Spearman ρ (or agreement rate)
```

This matrix is built once from a human-annotated benchmark corpus, then used to weight all future teacher score aggregation.

### CSA: Confidence-based Score Aggregation

Given $N$ teachers, each producing a score $s_i$ for dimension $d$, with confidence weights $\rho_i = C[\text{group}][\text{model}_i][d]$:

$$w_i = \frac{\exp(\rho_i / \tau)}{\sum_{j=1}^{N} \exp(\rho_j / \tau)}, \qquad \text{agg\_score}[d] = \sum_{i=1}^{N} s_i \cdot w_i$$

Where:
- $\tau$: temperature (default 0.05). Lower → sharper weighting toward best teacher
- $w_i \in [0, 1], \sum w_i = 1$: softmax-normalized confidence weight
- Higher $\rho_i$ → exponentially higher influence on the aggregated score

**Aggregation rules by dimension type:**

| Dimension Type | Rule | Example |
|---|---|---|
| Ordinal (1-5 scale) | Softmax-weighted average, round to integer | Quality scores |
| Binary (yes/no) | Majority vote | Toxicity flag |
| Continuous | Softmax-weighted average, keep float | Sentiment score |

---

## Four-Stage Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Stage 1    │    │  Stage 2     │    │  Stage 3    │    │  Stage 4     │
│  Teacher    │───▶│  Multi-Teach │───▶│  Student    │───▶│  Evaluation  │
│  Selection  │    │  Scoring +   │    │  Training   │    │  + Ablation  │
│  & Benchmark│    │  CSA Aggreg. │    │  (Full SFT) │    │              │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
     │                   │                   │                   │
  Gold corpus       Confidence         Trained student     ρ, F1, GAL
  + teacher          matrix +          model               results
  scoring            MTDD dataset
```

---

### Stage 1: Teacher Selection & Benchmarking

**Goal**: Select 2-4 teacher models and build the confidence matrix.

#### Step 1.1 — Prepare a Gold Corpus

Collect a human-annotated benchmark dataset that:
- Covers all target sub-groups (languages, domains, demographics)
- Uses the same rubric that teachers will apply
- Has sufficient per-group sample size (≥ 100 per sub-group for stable ρ)
- Is annotated by domain experts with inter-annotator agreement measured

#### Step 1.2 — Define the Multi-Dimensional Rubric

Design a structured rubric with:
- **Ordinal dimensions** (e.g., 1-5 Likert scale): capture quality facets
- **Binary dimensions** (e.g., yes/no): hard gates or binary classifiers
- **Hierarchical grouping**: organize dimensions by capability level

> **Design tip**: Keep the rubric prompt under 1,500 chars. Verbose prompts (> 5K chars) yield negligible quality improvement but cost more per call.

#### Step 1.3 — Benchmark Teachers on Gold Corpus

For each candidate teacher:
1. Score the entire gold corpus using the rubric prompt
2. Compute per-(sub-group, dimension) Spearman ρ against human scores
3. Compute per-(sub-group, dimension) Cohen's κ (LWK, QWK) for agreement
4. Record cost ($/M tokens), latency (s/request), and throughput (req/s)

**Selection criteria**:
- Rank correlation ρ ≥ 0.3 on most dimensions
- Acceptable cost for scoring millions of samples
- Complementary specialization (not all strong in the same dimensions)

#### Step 1.4 — Build the Confidence Matrix

```python
confidence_matrix = {}
for language in gold_corpus_languages:
    for model in teacher_models:
        for dimension in rubric_dimensions:
            rho = spearman(model_scores[model][language][dimension],
                          human_scores[language][dimension])
            confidence_matrix[language][model][dimension] = rho
```

Save as JSON — this file is the single source of truth for all subsequent aggregation.

---

### Stage 2: Multi-Teacher Scoring + CSA Aggregation

**Goal**: Score the raw dataset with all teachers, then aggregate into one MTDD dataset.

#### Step 2.1 — Prepare Raw Dataset

Curate a large, diverse raw dataset:
- Multiple data sources (web crawls, domain corpora, encyclopedias)
- Balanced across sub-groups (avoid dominant languages/categories)
- Deduplicated by text hash (MD5/SHA256)
- Filtered for minimum length and basic validity

#### Step 2.2 — Score with Each Teacher (Parallel)

Run inference for each teacher independently. This is embarrassingly parallel — all teachers can run simultaneously:

```bash
# Run all teachers in parallel
python score.py --teacher teacher-a --data raw/ --output scores_a.jsonl &
python score.py --teacher teacher-b --data raw/ --output scores_b.jsonl &
python score.py --teacher teacher-c --data raw/ --output scores_c.jsonl &
wait
```

**Inference best practices**:
- Use async HTTP with concurrency limits (e.g., 32 concurrent workers)
- Implement retry with exponential backoff (3 retries, 1-2-4s delays)
- Disable reasoning/thinking mode for throughput (10× faster)
- Low temperature (0.1) for deterministic outputs
- Resume support: track completed samples by hash, skip on restart

**JSON extraction from LLM outputs** (critical for reliability):

```python
def extract_json(llm_output: str) -> Optional[dict]:
    # 1. Strip thinking/reasoning tags
    output = re.sub(r'<thinking>.*?</thinking>', '', llm_output, flags=re.DOTALL)

    # 2. Try JSON code fence first
    m = re.search(r'```(?:json)?\s*\n?(.*?)```', output, re.DOTALL)
    if m:
        try: return json.loads(m.group(1).strip())
        except: pass

    # 3. Fallback: find JSON object by known key
    m = re.search(r'\{[^{}]*"<first_dim_key>"[^{}]*\}', output, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except: pass

    return None
```

#### Step 2.3 — Aggregate Scores with CSA

Load all teacher scores, join by sample hash, and apply CSA:

```python
def aggregate(sample_scores, confidence_matrix, sub_group, dimension):
    """
    Args:
        sample_scores: {model_name: score} for one sample, one dimension
        confidence_matrix: pre-computed ρ for (sub_group, model, dimension)
        sub_group: language/category of this sample
        dimension: rubric dimension name
    """
    models = [m for m in sample_scores if sample_scores[m] is not None]
    if not models:
        return None

    rho_values = [confidence_matrix.get(sub_group, {}).get(m, {}).get(dimension, 0.0)
                  for m in models]

    # Softmax weighting (τ = 0.05)
    tau = 0.05
    max_rho = max(rho_values)
    exp_vals = [math.exp((r - max_rho) / tau) for r in rho_values]
    sum_exp = sum(exp_vals)
    weights = [e / sum_exp for e in exp_vals]

    # Weighted average
    agg = sum(sample_scores[m] * w for m, w in zip(models, weights))
    return round(agg)  # round to integer for ordinal scales
```

**Fallback strategy**: If a sub-group is not in the confidence matrix (unseen language), use model-level average confidence as weight.

#### Step 2.4 — MTDD Output Format

Each aggregated sample should contain:

```json
{
  "text": "<original text>",
  "text_hash": "d41d8cd98f00b204e9800998ecf8427e",
  "sub_group": "Arabic",
  "domain": "encyclopedia",
  "scores": {
    "dim_1": 4,
    "dim_2": 3,
    "dim_binary": "yes"
  },
  "teacher_count": 3,
  "teachers_used": ["Teacher-A", "Teacher-B", "Teacher-C"]
}
```

---

### Stage 3: Student Model Training

**Goal**: Fine-tune a compact model on the MTDD dataset.

#### Training Configuration

| Parameter | Recommendation | Rationale |
|---|---|---|
| Backbone | Pre-trained instruction model (3-8B params) | Good instruction-following base |
| Method | Full SFT | MMQE gains +13pp Spearman ρ over the base model |
| Epochs | 3 | Convergence without overfitting |
| Input | Text + rubric prompt (~1K chars) | Compact prompt sufficient |
| Output | JSON with dimension scores | ~100 tokens per sample |
| Max input | 4096 tokens | Covers most documents |
| Max output | 256 tokens | Comfortable for compact JSON |

#### Training Data Format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "<rubric prompt with dimension definitions>"
    },
    {
      "role": "user",
      "content": "<document text>"
    },
    {
      "role": "assistant",
      "content": "{\"dim_1\": 4, \"dim_2\": 3, \"dim_binary\": \"yes\"}"
    }
  ]
}
```

#### Key Decision: Prompt Language

**Always use English for the rubric prompt**, even for non-English data:
- Reduces dimension name mismatches by ~7pp
- LLMs follow English instructions more reliably
- 1K-char prompt is sufficient — no need for verbose descriptions
- Test: compare English vs. native-language prompts on a subset

---

### Stage 4: Evaluation

#### Metrics

| Metric | Purpose | How |
|---|---|---|
| Spearman ρ | Rank correlation with humans | Per-dimension, per-sub-group, overall sum score |
| Cohen's κ (LWK/QWK) | Agreement with humans | Weighted for ordinal scales |
| F1@P-threshold | Classification at cutoff | Precision/Recall/F1 at P20 threshold + binary gate |
| Throughput | Production readiness | req/s, tokens/s, $/M tokens |
| GAL ρ | Generalization to unseen sub-groups | Evaluate on held-out sub-groups |

#### Generalization Across Sub-groups (GAL)

Always evaluate on sub-groups **not seen during training**:
- Hold out a subset of sub-groups from the MTDD training data
- Evaluate student model on those held-out groups
- Report ρ separately for seen vs. unseen sub-groups

This measures whether the student learned transferable quality concepts vs. memorizing sub-group-specific patterns.

---

## Design Decisions & Rationale

### Why Multi-Teacher (Not Single)?

| Advantage | Evidence |
|---|---|
| Complementary strengths | Teachers specialize in different sub-groups/dimensions |
| Robustness | Reduces single-model systematic bias |
| Cost efficiency | Expensive models only influence where they add most value |
| Ablation result | +5.0% QWK, +2.5% ρ vs. best single teacher |

### Why Softmax Weighting (Not Simple Average)?

- **Simple average**: All teachers equal → noisy teacher dilutes signal
- **Linear ρ weighting**: $w_i = \rho_i / \sum \rho_j$ → moderate teacher discrimination
- **Softmax (τ=0.05)**: $w_i \propto \exp(\rho_i / \tau)$ → sharply favors best teacher per (group, dim)

Softmax with low τ amplifies the best teacher's influence while still allowing weaker teachers to contribute. The temperature τ controls the sharpness:
- τ → ∞: uniform weights (simple average)
- τ → 0: winner-take-all (best teacher only)
- τ = 0.05: empirically optimal for 3-teacher setups

### Why Compact Rubric Prompts?

- 1K-char vs. 8K-char rubric → negligible ρ difference
- Shorter prompts = lower cost, faster inference, higher throughput
- Key insight: LLMs already understand quality concepts; prompt just calibrates scale

---

## Case Study: MMQE (Multilingual Quality Estimation)

This skill was validated in the MMQE project — building a multilingual web content quality estimator.

### Project-Specific Configuration

| Element | MTDD General | MMQE Specific |
|---|---|---|
| Sub-groups | Languages, domains, demographics | 21 languages |
| Rubric | Multi-dimensional, ordinal + binary | 7D-QR (7 dims, 4 levels) |
| Gold corpus | Human-annotated benchmark | QECA: 14,706 samples, 38 languages, 40+ annotators |
| Teachers | 2-4 API models | Gemini-3.7-Flash, Qwen3.8-Flash, DeepSeek-V4-Flash |
| Student | 3-8B instruction model | Qwen3-4B-Instruct-2507 |
| Dataset size | Millions | 1.69M samples |

### 7D-QR Rubric

| Level | Dimension | Type | Scale |
|---|---|---|---|
| L1 Safety | Text Toxicity | Binary | yes (non-toxic) / no (toxic) |
| L2 Lexical | Text Fluency | Ordinal | 1-5 |
| L2 Lexical | Text Cleanliness | Ordinal | 1-5 |
| L3 Information | Information Richness | Ordinal | 1-5 |
| L4 Educational | Knowledge Value | Ordinal | 1-5 |
| L4 Educational | Reasoning Value | Ordinal | 1-5 |
| L4 Educational | Artistic & Cultural Value | Ordinal | 1-5 |

### Results

| Metric | Best Selected Teacher (Gemini-3.7-Flash) | MMQE Student (Qwen3-4B) | Δ |
|---|---|---|---|
| Spearman ρ | 0.680 | **0.743** | +6.3pp |
| Throughput | 0.83 Ktok/s | **120+ Ktok/s** | ~140× |
| Cost | $4.50/M tok | ~$0.17/M tok | ~1/26 |

Note: "Best Selected Teacher" = best among the three teachers chosen for distillation. Claude-Sonnet-5 scores higher overall (ρ = 0.692, Table 4 of the paper) but was not selected as a teacher due to cost/throughput trade-offs; MMQE still exceeds it by +5.1pp.

### Confidence Matrix Snapshot (Arabic)

| Dimension | Gemini-3.7-Flash | Qwen3.8-Flash | DeepSeek-V4-Flash |
|---|---|---|---|
| Text Fluency | 0.80 | 0.55 | 0.59 |
| Text Cleanliness | 0.79 | 0.62 | 0.58 |
| Info. Richness | 0.65 | 0.48 | 0.52 |
| Knowledge Value | 0.60 | 0.45 | 0.51 |
| Toxicity (agreement) | 0.95 | 0.91 | 0.89 |

→ Gemini-3.7-Flash dominates Arabic, so CSA assigns it ~90% weight for Arabic dimensions.

---

## Reference Implementation

### File Structure

```
project/
├── config/                    # Per-run configurations
│   └── run_config.json
├── stage1_benchmark/            # Teacher scoring on gold corpus
│   ├── score_gold.py          # Score gold corpus with each teacher
│   └── build_confidence.py    # Compute confidence matrix
├── stage2_mtdd/                 # Multi-teacher dataset construction
│   ├── score_raw.py           # Score raw data with each teacher
│   └── csa_aggregate.py       # CSA aggregation
├── stage3_training/             # Student model training
│   └── train_sft.py           # Full SFT training
├── stage4_evaluation/           # Evaluation
│   ├── evaluate.py            # Main evaluation
│   └── gal_study.py           # Generalization across sub-groups
├── lib/                         # Shared utilities
│   ├── config_loader.py
│   └── json_extractor.py
└── knowledge/                   # This document
    └── mmqe_skill.md
```

### Configuration Template

```json
{
  "data_root": "/path/to/data/run_001",
  "gold_corpus": "/path/to/gold_corpus.jsonl",
  "raw_data_dir": "/path/to/raw_data/",
  "teachers": [
    {
      "name": "Teacher-A",
      "model_id": "teacher-a-flash",
      "api_url": "https://api.provider.com/v1/chat",
      "concurrent_workers": 32,
      "max_retries": 3,
      "request_timeout": 60
    }
  ],
  "student": {
    "base_model": "instruct-model-4b",
    "epochs": 3,
    "max_input_len": 4096,
    "max_output_len": 256
  },
  "rubric_prompt_file": "prompts/rubric_en.txt",
  "csa_temperature": 0.05
}
```

### Checklist

- [ ] Gold corpus annotated with inter-annotator agreement measured
- [ ] Rubric prompt designed and validated (≤ 1,500 chars, English)
- [ ] ≥ 2 teacher models benchmarked on gold corpus
- [ ] Confidence matrix computed and saved as JSON
- [ ] Raw dataset curated, deduplicated, balanced across sub-groups
- [ ] All teachers scored the raw dataset (parallel)
- [ ] CSA aggregation applied → MTDD dataset
- [ ] Student model trained with full SFT (3 epochs)
- [ ] Evaluation on gold corpus (Spearman ρ, F1@threshold)
- [ ] GAL study on held-out sub-groups
- [ ] Ablation: single-teacher vs. multi-teacher comparison

---

## Common Pitfalls

| Pitfall                            | Symptom | Fix |
|------------------------------------|---|---|
| Thinking mode enabled              | 10× slower inference, same quality | Disable reasoning/thinking in API payload |
| Verbose rubric prompt (> 5K chars) | Higher cost, no ρ improvement | Compress to ≤ 1,500 chars |
| Non-English rubric prompt          | Dimension name mismatches (+7pp error) | Always use English prompts |
| No resume support                  | Re-scoring after failures wastes $$$ | Track completed samples by hash |
| Too few gold samples per sub-group | Unstable ρ estimates (high variance) | ≥ 100 samples per sub-group |
| Missing teacher scores             | Sample dropped entirely | Fallback to available teachers, re-normalize weights |
| Only in-distribution evaluation    | Overestimate generalization | Always run GAL study on held-out groups |

---

## References

- **MMQE Paper**: Multi-teacher aggregated Multilingual Quality Estimation framework. ./mmqe.tex
- **CSA Algorithm**: Confidence-based Score Aggregation with softmax weighting
- **Code**: https://anonymous.4open.science/r/MMQE-655A/
