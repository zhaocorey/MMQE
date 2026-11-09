"""Stage 4 — Ablation: single-teacher vs. multi-teacher (CSA) vs. student.

Re-aggregates the gold-corpus teacher scores under different strategies and
evaluates each against human annotations with identical metrics:

  1. each teacher alone            (single-teacher baselines)
  2. CSA softmax (tau sweep)       (multi-teacher, paper setting tau=0.05)
  3. CSA uniform / linear          (aggregation-rule ablation)
  4. student model                 (optional, via evaluate.py predictions)

This reproduces the paper's key ablation: multi-teacher CSA aggregation
itself (before any training) should already beat the best single teacher
on QWK / rho.

Usage:
    python -m stage4_evaluation.ablation --config config/run_config.json \
        --gold /path/to/gold_corpus.jsonl \
        --gold-scores-dir <data_root>/gold_scores/ \
        --confidence <data_root>/confidence_matrix.json \
        --output <data_root>/ablation_report.json \
        [--student-predictions <data_root>/preds_student.jsonl]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config_loader import apply_overrides, load_config
from stage1_benchmark.build_confidence import load_teacher_scores
from stage2_mtdd.csa_aggregate import CSAAggregator
from stage4_evaluation.evaluate import (compute_report, load_gold,
                                        predictions_from_scores_file)
from lib.scorer import sample_hash

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stage4.ablation")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/run_config.json")
    ap.add_argument("--gold", help="Gold corpus JSONL (default: config.gold_corpus)")
    ap.add_argument("--gold-scores-dir", required=True)
    ap.add_argument("--confidence", required=True)
    ap.add_argument("--student-predictions", default=None,
                    help="Optional predictions JSONL for the student row")
    ap.add_argument("--taus", nargs="*", type=float,
                    default=[0.01, 0.05, 0.2, 1.0],
                    help="Temperature sweep for softmax mode")
    ap.add_argument("--output", help="Report JSON path")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.override)
    gold_path = args.gold or cfg.get("gold_corpus")
    if not gold_path:
        ap.error("no gold corpus: pass --gold or set gold_corpus in config")
    gold = load_gold(gold_path)
    hash_to_group = {sample_hash(s): s.get("sub_group") for s in gold}

    with open(args.confidence, "r", encoding="utf-8") as f:
        confidence = json.load(f)
    names = [t["name"] for t in cfg["teachers"]]
    teacher_scores = load_teacher_scores(Path(args.gold_scores_dir), names)

    rows: List[Dict[str, Any]] = []

    def evaluate_row(label: str, preds: Dict[str, Dict[str, Any]]) -> None:
        try:
            report = compute_report(cfg, gold, preds)
        except SystemExit:
            logger.warning("skipping %s (no paired samples)", label)
            return
        row = {
            "variant": label,
            "n_paired": report["n_paired"],
            "sum_score_rho": report["sum_score_rho"],
            "mean_qwk": _mean_metric(report, "qwk"),
            "mean_lwk": _mean_metric(report, "lwk"),
            "mean_rho_per_dim": _mean_metric(report, "spearman_rho"),
            "f1_at_80pct": report["f1_at_80pct"]["f1"],
        }
        rows.append(row)
        logger.info("%-34s sum_rho=%s mean_qwk=%s mean_rho=%s",
                    label, row["sum_score_rho"], row["mean_qwk"],
                    row["mean_rho_per_dim"])

    def _mean_metric(report: Dict[str, Any], key: str):
        vals = [m[key] for m in report["per_dimension_overall"].values()
                if key in m and m[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    # 1. Single-teacher baselines
    for name in names:
        evaluate_row(f"single:{name}", teacher_scores.get(name, {}))

    # 2. CSA softmax with temperature sweep
    for tau in args.taus:
        preds = _csa_preds(cfg, confidence, teacher_scores, hash_to_group,
                           "softmax", tau)
        evaluate_row(f"csa:softmax(tau={tau})", preds)

    # 3. CSA uniform / linear (tau irrelevant)
    for mode in ("uniform", "linear"):
        preds = _csa_preds(cfg, confidence, teacher_scores, hash_to_group,
                           mode, 0.05)
        evaluate_row(f"csa:{mode}", preds)

    # 4. Student model (optional)
    if args.student_predictions:
        preds = predictions_from_scores_file(args.student_predictions)
        evaluate_row("student_model", preds)

    # Console table
    print("\n=== Ablation summary ===")
    header = (f"{'variant':<36}{'n':>7}{'sum_rho':>9}{'mean_qwk':>10}"
              f"{'mean_rho':>10}")
    print(header)
    for r in rows:
        print(f"{r['variant']:<36}{r['n_paired']:>7}"
              f"{_fmt(r['sum_score_rho']):>9}{_fmt(r['mean_qwk']):>10}"
              f"{_fmt(r['mean_rho_per_dim']):>10}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"rows": rows}, f, ensure_ascii=False, indent=2)
        logger.info("wrote report -> %s", out)


def _csa_preds(cfg: Dict[str, Any],
               confidence: Dict[str, Any],
               teacher_scores: Dict[str, Dict[str, Dict]],
               hash_to_group: Dict[str, Any],
               mode: str, tau: float) -> Dict[str, Dict[str, Any]]:
    agg = CSAAggregator(confidence, cfg["dimensions"], tau=tau, mode=mode)
    all_hashes = set()
    for per_hash in teacher_scores.values():
        all_hashes |= set(per_hash)
    preds: Dict[str, Dict[str, Any]] = {}
    for h in all_hashes:
        per_teacher = {m: ph[h] for m, ph in teacher_scores.items() if h in ph}
        scores, _ = agg.aggregate_sample(per_teacher, hash_to_group.get(h))
        if scores:
            preds[h] = scores
    return preds


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "-"


if __name__ == "__main__":
    main()
