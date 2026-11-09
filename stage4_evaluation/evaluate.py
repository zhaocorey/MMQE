"""Stage 4 — Evaluate a scored model against the human gold corpus.

Works for ANY source of predictions (student model or teacher), enabling
the single-teacher vs. multi-teacher vs. student ablation with identical
metrics:

  A) Student/local model:  --model <path> [--backend hf|vllm]
     runs inference on the gold corpus itself.
  B) Teacher scores:       --teacher-scores <gold_scores_Teacher.jsonl>
  C) Pre-computed preds:   --predictions <predictions.jsonl>

Reports (JSON + console table):
  - per-dimension overall: Spearman rho, QWK, LWK (ordinal); agreement,
    F1 with positive="no" i.e. toxic-detection (binary)
  - per-(sub_group, dimension) rho with macro/micro summaries
  - sum-score rho — paper Sec 5.2 definition:
      sum = fluency + cleanliness + richness + Education_Value,
      Education_Value = max(knowledge, reasoning, artistic_cultural)
  - F1@80% — paper Appendix B: per-side P20 percentile of sum score as
    pass threshold + non-toxic gate; P/R/F1 of model-pass vs human-pass
  - score distribution stats (mean/var/P20/P50/P80, cf. paper Table 7)
  - throughput (samples/s, approx Ktok/s) when inference is run here

Usage:
    # Paper protocol (Sec 5.3): filter QECA to the 21 MTDD target languages
    python -m stage4_evaluation.evaluate --config config/run_config.json \
        --gold /path/to/gold_corpus.jsonl \
        --model <data_root>/student_model --target-only \
        --output <data_root>/eval_student.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config_loader import apply_overrides, load_config, load_rubric_prompt
from lib.json_extractor import coerce_scores, extract_json
from lib.metrics import (agreement_rate, binary_f1, cohen_kappa,
                         education_value, f1_at_80pct, percentile,
                         spearman_rho, sum_score, summarize_correlations)
from lib.scorer import iter_jsonl, sample_hash

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stage4.evaluate")


# --------------------------------------------------------------------------- #
# Prediction sources
# --------------------------------------------------------------------------- #

def load_gold(path: str | Path) -> List[Dict[str, Any]]:
    return [s for s in iter_jsonl(path) if s.get("human_scores")]


def predictions_from_model(cfg: Dict[str, Any], model_path: str,
                           gold: List[Dict[str, Any]], backend: str,
                           batch_size: int) -> Tuple[Dict[str, Dict], Dict]:
    from lib.student_predictor import StudentPredictor
    rubric = load_rubric_prompt(cfg)
    student_cfg = cfg.get("student", {})
    inf_cfg = student_cfg.get("inference", {})
    predictor = StudentPredictor(
        model_path, rubric, backend=backend,
        max_input_len=int(student_cfg.get("max_input_len", 4096)),
        max_output_len=int(student_cfg.get("max_output_len", 256)),
        batch_size=batch_size,
        gpu_memory_utilization=float(inf_cfg.get("gpu_memory_utilization", 0.9)),
        temperature=float(inf_cfg.get("temperature", 0.1)))
    texts = [s["text"] for s in gold]
    raws = predictor.predict(texts)

    first_dim = next(iter(cfg["dimensions"]))
    preds: Dict[str, Dict] = {}
    n_fail = 0
    for s, raw in zip(gold, raws):
        obj = extract_json(raw or "", first_key_hint=first_dim)
        scores = coerce_scores(obj, cfg["dimensions"]) if obj else None
        if scores is None:
            n_fail += 1
            continue
        preds[sample_hash(s)] = scores
    logger.info("model predictions: %d ok, %d parse failures",
                len(preds), n_fail)
    stats = dict(predictor.last_stats)
    stats["parse_failures"] = n_fail
    return preds, stats


def predictions_from_scores_file(path: str | Path) -> Dict[str, Dict]:
    preds = {}
    for rec in iter_jsonl(path):
        if rec.get("scores"):
            preds[rec["text_hash"]] = rec["scores"]
    logger.info("loaded %d predictions from %s", len(preds), path)
    return preds


def dump_predictions(preds: Dict[str, Dict], gold: List[Dict[str, Any]],
                     path: Path) -> None:
    meta = {sample_hash(s): s.get("sub_group") for s in gold}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, scores in preds.items():
            f.write(json.dumps({"text_hash": h, "sub_group": meta.get(h),
                                "scores": scores}, ensure_ascii=False) + "\n")
    logger.info("wrote predictions -> %s", path)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_report(cfg: Dict[str, Any],
                   gold: List[Dict[str, Any]],
                   preds: Dict[str, Dict[str, Any]],
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    dims: Dict[str, Dict[str, Any]] = cfg["dimensions"]
    ordinal: List[str] = cfg["ordinal_dimensions"]
    binary: List[str] = cfg["binary_dimensions"]
    scale_labels = list(range(1, 6))

    # Paired samples only
    pairs = [(s, preds[sample_hash(s)]) for s in gold
             if sample_hash(s) in preds]
    logger.info("evaluating %d paired samples (%d gold, %d predictions)",
                len(pairs), len(gold), len(preds))
    if not pairs:
        raise SystemExit("No overlap between gold corpus and predictions")

    per_dim_overall: Dict[str, Any] = {}
    per_dim_per_group: Dict[str, Dict[str, Optional[float]]] = {}
    group_sizes: Dict[str, int] = defaultdict(int)
    for s, _ in pairs:
        group_sizes[s.get("sub_group")] += 1

    for d, spec in dims.items():
        obs = [(p[d], s["human_scores"][d]) for s, p in pairs
               if d in p and d in s["human_scores"]]
        if not obs:
            continue
        pred_vals = [o[0] for o in obs]
        gold_vals = [o[1] for o in obs]

        if d in ordinal:
            overall = {
                "n": len(obs),
                "spearman_rho": _r(spearman_rho(
                    [float(x) for x in pred_vals],
                    [float(x) for x in gold_vals])),
                "qwk": _r(cohen_kappa(gold_vals, pred_vals, "quadratic",
                                      scale_labels)),
                "lwk": _r(cohen_kappa(gold_vals, pred_vals, "linear",
                                      scale_labels)),
                "exact_agreement": _r(agreement_rate(gold_vals, pred_vals)),
            }
            # per-group rho
            by_group: Dict[Any, Tuple[list, list]] = defaultdict(
                lambda: ([], []))
            for (s, p) in pairs:
                if d in p and d in s["human_scores"]:
                    g = s.get("sub_group")
                    by_group[g][0].append(float(p[d]))
                    by_group[g][1].append(float(s["human_scores"][d]))
            per_group_rho = {}
            for g, (pv, gv) in by_group.items():
                per_group_rho[str(g)] = _r(spearman_rho(pv, gv))
            per_dim_per_group[d] = per_group_rho
        else:  # binary
            overall = {
                "n": len(obs),
                "agreement": _r(agreement_rate(gold_vals, pred_vals)),
                # positive="no" -> detecting toxic content
                **{f"toxic_{k}": _r(v)
                   for k, v in binary_f1(gold_vals, pred_vals,
                                         positive="no").items()},
            }
        per_dim_overall[d] = overall

    # Sum-score rho — paper Sec 5.2 definition:
    #   sum = fluency + cleanliness + richness + Education_Value
    #   Education_Value = max(knowledge, reasoning, artistic_cultural)
    # (gold may carry a collapsed "education_value" key — sum_score handles
    #  both forms via the spec in cfg.evaluation.sum_score)
    sum_spec = cfg.get("evaluation", {}).get("sum_score") or _default_sum_spec(cfg)
    tox_key = cfg.get("evaluation", {}).get("toxicity_gate", "text_toxicity")

    # Education Value as its own dimension (paper Fig 5 per-dimension view):
    # model side = max over L4 dims; gold side = collapsed education_value.
    ev_key = sum_spec.get("education_value_key", "education_value")
    ev_obs = []
    for s, p in pairs:
        g_ev = s["human_scores"].get(ev_key)
        p_ev = education_value(p, sum_spec)
        if g_ev is not None and p_ev is not None:
            ev_obs.append((float(p_ev), float(g_ev)))
    if ev_obs:
        pv = [a for a, _ in ev_obs]
        gv = [b for _, b in ev_obs]
        pv_i = [int(round(x)) for x in pv]
        gv_i = [int(round(x)) for x in gv]
        per_dim_overall[ev_key] = {
            "n": len(ev_obs),
            "spearman_rho": _r(spearman_rho(pv, gv)),
            "qwk": _r(cohen_kappa(gv_i, pv_i, "quadratic", scale_labels)),
            "lwk": _r(cohen_kappa(gv_i, pv_i, "linear", scale_labels)),
            "exact_agreement": _r(agreement_rate(gv_i, pv_i)),
            "note": "model=max(knowledge,reasoning,artistic) vs human collapsed L4",
        }

    sum_obs = []
    for s, p in pairs:
        ps = sum_score(p, sum_spec)
        hs = sum_score(s["human_scores"], sum_spec)
        if ps is not None and hs is not None:
            sum_obs.append((ps, hs))
    sum_rho = _r(spearman_rho([a for a, _ in sum_obs],
                              [b for _, b in sum_obs])) if sum_obs else None

    # F1@80% — paper Appendix B: each side computes its own sum score;
    # pass = (sum >= own P20 percentile) AND non-toxic; P/R/F1 of
    # model-pass vs human-pass.
    f1_pct = float(cfg.get("evaluation", {}).get("f1_percentile", 0.2))
    m_sums, h_sums, m_nt, h_nt = [], [], [], []
    for s, p in pairs:
        ps = sum_score(p, sum_spec)
        hs = sum_score(s["human_scores"], sum_spec)
        if ps is None or hs is None:
            continue
        m_sums.append(ps)
        h_sums.append(hs)
        m_nt.append(p.get(tox_key) == "yes")
        h_nt.append(s["human_scores"].get(tox_key) == "yes")
    f1_80 = f1_at_80pct(m_sums, m_nt, h_sums, h_nt, pct=f1_pct)

    # Score distribution stats (paper Table 7: mean/var/P20/P50/P80)
    def dist_stats(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {}
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return {"mean": round(mean, 2), "var": round(var, 2),
                "p20": percentile(vals, 0.2), "p50": percentile(vals, 0.5),
                "p80": percentile(vals, 0.8)}

    report = {
        "n_gold": len(gold),
        "n_predictions": len(preds),
        "n_paired": len(pairs),
        "per_dimension_overall": per_dim_overall,
        "sum_score": {
            "definition": "fluency + cleanliness + richness + "
                          "max(knowledge, reasoning, artistic_cultural)",
            "n": len(sum_obs),
            "spearman_rho": sum_rho,
            "model_distribution": dist_stats([a for a, _ in sum_obs]),
            "human_distribution": dist_stats([b for _, b in sum_obs]),
        },
        "sum_score_rho": sum_rho,  # convenience alias
        "f1_at_80pct": f1_80,
        "per_dimension_per_group_rho": per_dim_per_group,
        "per_dimension_group_summary": {
            d: summarize_correlations(g, {str(k): v for k, v in group_sizes.items()})
            for d, g in per_dim_per_group.items()},
    }
    if extra:
        report["inference_stats"] = extra
    return report


def _default_sum_spec(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {"additive": ["text_fluency", "text_cleanliness",
                         "information_richness"],
            "education_value_key": "education_value",
            "max_of": ["knowledge_value", "reasoning_value",
                       "artistic_cultural_value"]}


def _r(x: Optional[float]) -> Optional[float]:
    return round(x, 4) if x is not None else None


def print_report(report: Dict[str, Any]) -> None:
    print("\n=== Overall per-dimension ===")
    for d, m in report["per_dimension_overall"].items():
        bits = ", ".join(f"{k}={v}" for k, v in m.items())
        print(f"  {d:<26} {bits}")
    ss = report["sum_score"]
    print(f"\nsum-score Spearman rho : {ss['spearman_rho']}  (n={ss['n']})")
    print(f"model sum distribution : {ss['model_distribution']}")
    print(f"human sum distribution : {ss['human_distribution']}")
    print(f"F1@80% (P20+non-toxic) : {report['f1_at_80pct']}")
    print("\n=== Per-dimension group summary (macro/micro rho) ===")
    for d, s in report["per_dimension_group_summary"].items():
        print(f"  {d:<26} {s}")
    if "inference_stats" in report:
        print(f"\ninference: {report['inference_stats']}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/run_config.json")
    ap.add_argument("--gold", help="Gold corpus JSONL (default: config.gold_corpus)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="Student model path (runs inference)")
    src.add_argument("--teacher-scores",
                     help="gold_scores_<teacher>.jsonl from Stage 1")
    src.add_argument("--predictions", help="Pre-computed predictions JSONL")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--languages", nargs="*", default=None,
                    help="Restrict evaluation to these sub-groups")
    ap.add_argument("--target-only", action="store_true",
                    help="Restrict to config.languages — the paper protocol "
                         "(QECA filtered to the 21 MTDD target languages, "
                         "Sec 5.3)")
    ap.add_argument("--output", help="Report JSON path (default: stdout only)")
    ap.add_argument("--dump-predictions",
                    help="Also write predictions JSONL here (reuse in gal_study)")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.override)
    gold_path = args.gold or cfg.get("gold_corpus")
    if not gold_path:
        ap.error("no gold corpus: pass --gold or set gold_corpus in config")
    gold = load_gold(gold_path)

    if args.target_only and args.languages:
        ap.error("--target-only and --languages are mutually exclusive")
    if args.target_only:
        langs = set(cfg.get("languages", []))
    elif args.languages:
        langs = set(args.languages)
    else:
        langs = None
    if langs is not None:
        before = len(gold)
        gold = [s for s in gold if s.get("sub_group") in langs]
        logger.info("language filter: %d -> %d gold samples (%d languages)",
                    before, len(gold), len(langs))
        if not gold:
            ap.error("language filter removed all gold samples — check "
                     "sub_group naming against the config language lists")

    extra: Optional[Dict[str, Any]] = None
    if args.model:
        preds, extra = predictions_from_model(cfg, args.model, gold,
                                              args.backend, args.batch_size)
    elif args.teacher_scores:
        preds = predictions_from_scores_file(args.teacher_scores)
    else:
        preds = predictions_from_scores_file(args.predictions)

    if args.dump_predictions:
        dump_predictions(preds, gold, Path(args.dump_predictions))

    report = compute_report(cfg, gold, preds, extra=extra)
    print_report(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("wrote report -> %s", out)


if __name__ == "__main__":
    main()
