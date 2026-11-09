# MMQE — Multilingual Quality Estimation

A reference implementation of **Multi-Teacher Dataset Distillation (MTDD)**:
distill a large-scale multilingual quality dataset by aggregating scores from
multiple API teacher models with **CSA** (Confidence-based Score Aggregation),
then train a compact 4B student model that matches or exceeds teacher quality
at ~140× throughput and ~1/26 the cost.

See [`knowledge/mmqe_skill.md`](knowledge/mmqe_skill.md) for the full method.

## Pipeline

```
Stage 1                 Stage 2                  Stage 3            Stage 4
Teacher benchmark   →   Multi-teacher scoring →  Student SFT    →   Evaluation
(gold corpus)           + CSA aggregation      (Qwen3-4B)          (ρ, QWK, F1, GAL)
     │                        │                      │                   │
confidence_matrix.json   mtdd_dataset.jsonl    student_model/      reports (*.json)
```

## Repository layout

```
MMQE/
├── config/run_config.json       # MMQE run configuration (teachers, dims, CSA τ, student)
├── prompts/rubric_en.txt        # English 7D-QR rubric prompt (<1.5K chars — always English)
├── lib/
│   ├── config_loader.py         # config loading + dotted-path CLI overrides
│   ├── json_extractor.py        # robust JSON extraction from LLM outputs + score coercion
│   ├── llm_client.py            # async OpenAI-compatible client (32 workers, 1-2-4s backoff)
│   ├── scorer.py                # resume-safe batch scoring engine (dedupe by hash)
│   ├── metrics.py               # Spearman ρ, LWK/QWK, agreement, F1@P-threshold
│   └── student_predictor.py     # local student inference (hf / vllm backends)
├── stage1_benchmark/
│   ├── score_gold.py            # score the human gold corpus with each teacher
│   └── build_confidence.py      # C[language][model][dim] → confidence_matrix.json
├── stage2_mtdd/
│   ├── score_raw.py             # score raw data with each teacher (parallel, resume-safe)
│   └── csa_aggregate.py         # softmax-weighted aggregation → MTDD dataset (+ablation modes)
├── stage3_training/
│   ├── prepare_data.py          # MTDD → chat SFT format, GAL holdout split
│   └── train_sft.py             # full-parameter SFT (3 epochs, 4096/256, bf16)
├── stage4_evaluation/
│   ├── evaluate.py              # ρ / QWK / LWK / sum-score ρ / F1@80% vs. human gold
│   ├── gal_study.py             # seen vs. unseen sub-group generalization
│   └── ablation.py              # single-teacher vs. CSA(τ sweep) vs. student
└── knowledge/mmqe_skill.md      # method documentation
```

## Data formats

Gold corpus (JSONL). QECA annotators rate L4 as a single collapsed
"Education Value" dimension, so `human_scores` may carry **either** the three
L4 keys **or** a collapsed `education_value` key — evaluation handles both:

```json
{"text": "...", "sub_group": "Arabic", "domain": "encyclopedia",
 "human_scores": {"text_fluency": 4, "text_cleanliness": 5, "information_richness": 3,
                  "education_value": 4,
                  "text_toxicity": "yes"}}
```

Raw dataset (JSONL): `{"text": "...", "sub_group": "Arabic", "domain": "web"}`
(`text_hash` optional — computed as MD5 when absent).

MTDD output (JSONL):

```json
{"text": "...", "text_hash": "...", "sub_group": "Arabic", "domain": "encyclopedia",
 "scores": {"text_fluency": 4, "...": 3, "text_toxicity": "yes"},
 "teacher_count": 3, "teachers_used": ["Gemini-3.7-Flash", "Qwen3.8-Flash", "DeepSeek-V4-Flash"]}
```

## Quickstart

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=... DASHSCOPE_API_KEY=... DEEPSEEK_API_KEY=...
```

Edit `config/run_config.json` — set `data_root`, `gold_corpus`, `raw_data_dir`,
and the student `base_model`/`output_dir`. Any config field can also be
overridden per-command with `-o key=value` (e.g. `-o csa_temperature=0.1`).

### Stage 1 — Benchmark teachers, build the confidence matrix

```bash
# Score the gold corpus with all teachers (resume-safe)
python -m stage1_benchmark.score_gold --config config/run_config.json

# Build C[language][model][dimension] (Spearman ρ; agreement for binary dims)
python -m stage1_benchmark.build_confidence \
    --config config/run_config.json \
    --scores-dir $DATA_ROOT/gold_scores/ \
    --output $DATA_ROOT/confidence_matrix.json
```

Selection criteria: ρ ≥ 0.3 on most dimensions, complementary specialization,
≥ 100 gold samples per sub-group.

### Stage 2 — Multi-teacher scoring + CSA aggregation

```bash
# Score raw data — one process per teacher, run in parallel:
python -m stage2_mtdd.score_raw --teacher Gemini-3.7-Flash &
python -m stage2_mtdd.score_raw --teacher Qwen3.8-Flash &
python -m stage2_mtdd.score_raw --teacher DeepSeek-V4-Flash &
wait

# Aggregate with CSA (softmax, τ=0.05; ordinal→weighted avg rounded, binary→weighted vote)
python -m stage2_mtdd.csa_aggregate \
    --scores-dir $DATA_ROOT/scores/ \
    --confidence $DATA_ROOT/confidence_matrix.json \
    --raw-dir /path/to/raw_data/ \
    --output $DATA_ROOT/mtdd_dataset.jsonl
```

Ablations: `--mode uniform|linear`, `--single-teacher <Name>`, `--tau <t>`.

### Stage 3 — Student training (full SFT)

```bash
python -m stage3_training.prepare_data --config config/run_config.json \
    --output-dir $DATA_ROOT/sft/        # train / eval / heldout_groups splits

torchrun --nproc_per_node=8 -m stage3_training.train_sft \
    --config config/run_config.json \
    --train-file $DATA_ROOT/sft/train.jsonl
```

Full SFT (not frozen embedder + heads; MMQE gains +13pp ρ over the base
model), 3 epochs, English rubric prompt even for non-English documents,
assistant target = compact JSON (~100 tokens).

### Stage 4 — Evaluation

```bash
# Student vs. gold corpus: per-dim & per-language ρ, QWK/LWK, sum-score ρ, F1@80%
# --target-only reproduces the paper protocol: QECA filtered to the 21 MTDD
# target languages (Sec 5.3, 6.9K samples)
python -m stage4_evaluation.evaluate --config config/run_config.json \
    --model $DATA_ROOT/student_model --backend vllm --target-only \
    --output $DATA_ROOT/eval_student.json \
    --dump-predictions $DATA_ROOT/preds_student.jsonl

# Teacher baseline with identical metrics
python -m stage4_evaluation.evaluate --config config/run_config.json \
    --teacher-scores $DATA_ROOT/gold_scores/gold_scores_Gemini-3.7-Flash.jsonl

# GAL: seen vs. unseen sub-groups (unseen defaults to qeca_languages - languages,
# the 23 paper languages; pooled_rho is the paper's headline number)
python -m stage4_evaluation.gal_study --config config/run_config.json \
    --predictions $DATA_ROOT/preds_student.jsonl \
    --output $DATA_ROOT/gal_report.json

# Full ablation: single teachers vs. CSA(τ sweep) vs. student
python -m stage4_evaluation.ablation --config config/run_config.json \
    --gold-scores-dir $DATA_ROOT/gold_scores/ \
    --confidence $DATA_ROOT/confidence_matrix.json \
    --student-predictions $DATA_ROOT/preds_student.jsonl \
    --output $DATA_ROOT/ablation_report.json
```

## 7D-QR rubric

| Level | Dimension | Type | Scale |
|---|---|---|---|
| L1 Safety | text_toxicity | binary | yes (non-toxic) / no (toxic) |
| L2 Lexical | text_fluency | ordinal | 1–5 |
| L2 Lexical | text_cleanliness | ordinal | 1–5 |
| L3 Information | information_richness | ordinal | 1–5 |
| L4 Educational | knowledge_value | ordinal | 1–5 |
| L4 Educational | reasoning_value | ordinal | 1–5 |
| L4 Educational | artistic_cultural_value | ordinal | 1–5 |

The QECA human corpus collapses the three L4 dimensions into a single
"Education Value" score; model-side Education Value is defined as
`max(knowledge_value, reasoning_value, artistic_cultural_value)`.

## Paper metric definitions (mmqe.tex)

Implemented in `lib/metrics.py` exactly as the paper specifies:

- **Sum score** (§5.2): `text_fluency + text_cleanliness + information_richness
  + Education_Value`, where `Education_Value = max(knowledge_value,
  reasoning_value, artistic_cultural_value)`. Range 4–20. All headline ρ
  numbers (0.743, GAL 0.662) are Spearman ρ on this sum score.
- **F1@80%** (Appendix B): each side (human, model) computes its own sum
  score; "pass" = `sum >= own P20 percentile` AND non-toxic; report
  precision/recall/F1 of model-pass vs human-pass. The P20 cutoff yields an
  ~80% pass rate before the toxicity gate.
- **Distribution stats** (Table 7): mean / var / P20 / P50 / P80 of sum score,
  used for the stricter-vs-lenient judge analysis.

## Expected results (paper reference)

| Metric | Best selected teacher (Gemini-3.7-Flash) | MMQE student (Qwen3-4B) |
|---|---|---|
| Spearman ρ | 0.680 | **0.743** |
| F1@80% | 0.9036 | **0.9087** |
| Throughput | 0.83 Ktok/s | **120+ Ktok/s** |
| Cost | $4.50/M tok | ~$0.17/M tok |
| GAL ρ (23 unseen langs) | — | **0.662** |

## Operational notes

- **Resume safety**: every scoring step tracks completed samples by text hash
  and skips them on restart — failures never re-spend API budget.
- **Throughput**: thinking/reasoning mode is disabled in the API payload
  (10× faster); temperature 0.1; 32 concurrent workers per teacher.
- **JSON reliability**: layered extraction (strip `<thinking>` → code fence →
  key-anchored object → first balanced object) + strict per-dimension coercion.
- **Missing scores**: samples aggregate over available teachers with
  re-normalized weights; unseen sub-groups fall back to model-level average
  confidence.
