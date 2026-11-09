"""Stage 4 — GAL: Generalization Across (unseen) sub-groups.

Compares student quality estimation on sub-groups SEEN during MTDD
training vs. sub-groups HELD OUT (never in training data). This measures
whether the student learned transferable quality concepts rather than
memorizing sub-group-specific patterns.

Inputs:
  --gold          gold corpus JSONL with human_scores (any sub-groups)
  --model         student model path (runs inference)   OR
  --predictions   pre-computed predictions JSONL (from evaluate.py
                  --dump-predictions)                   OR
  --teacher-scores  a teacher's gold score file (baseline row)
  --seen / --unseen  explicit sub-group lists; defaults: seen =
                  config.languages (21 MTDD targets), unseen =
                  config.qeca_languages - seen (the 23 paper GAL languages)

Output: JSON report + console table with per-set (seen/unseen) macro rho
per dimension and overall, and the seen-vs-unseen gap.

Usage:
    python -m stage4_evaluation.gal_study --config config/run_config.json \
        --gold /path/to/gold_corpus.jsonl \
        --model <data_root>/student_model \
        --output <data_root>/gal_report.json
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

from lib.config_loader import apply_overrides, load_config
from lib.metrics import spearman_rho, sum_score
from lib.scorer import sample_hash
from stage4_evaluation.evaluate import (load_gold, predictions_from_model,
                                        predictions_from_scores_file)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stage4.gal_study")


def split_pairs(gold: List[Dict[str, Any]],
                preds: Dict[str, Dict[str, Any]],
                seen: set, unseen: set
                ) -> Tuple[List[Tuple[Dict, Dict]], List[Tuple[Dict, Dict]],
                           List[Tuple[Dict, Dict]]]:
    """Partition paired (gold, pred) samples into seen / unseen / other."""
    seen_pairs, unseen_pairs, other_pairs = [], [], []
    for s in gold:
        h = sample_hash(s)
        if h not in preds:
            continue
        pair = (s, preds[h])
        g = s.get("sub_group")
        if g in unseen:
            unseen_pairs.append(pair)
        elif g in seen:
            seen_pairs.append(pair)
        else:
            other_pairs.append(pair)
    return seen_pairs, unseen_pairs, other_pairs


def set_metrics(pairs: List[Tuple[Dict, Dict]],
                ordinal_dims: List[str],
                sum_spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Macro rho across sub-groups per dimension + overall sum-score rho."""
    sum_spec = sum_spec or {
        "additive": ["text_fluency", "text_cleanliness", "information_richness"],
        "education_value_key": "education_value",
        "max_of": ["knowledge_value", "reasoning_value",
                   "artistic_cultural_value"]}
    result: Dict[str, Any] = {"n": len(pairs)}
    if not pairs:
        return result

    per_dim: Dict[str, List[float]] = {}
    for d in ordinal_dims:
        by_group: Dict[Any, Tuple[list, list]] = defaultdict(lambda: ([], []))
        for s, p in pairs:
            if d in p and d in s["human_scores"]:
                g = s.get("sub_group")
                by_group[g][0].append(float(p[d]))
                by_group[g][1].append(float(s["human_scores"][d]))
        group_rho = {}
        for g, (pv, gv) in by_group.items():
            r = spearman_rho(pv, gv)
            if r is not None:
                group_rho[str(g)] = round(r, 4)
        if group_rho:
            per_dim[d] = {
                "macro_rho": round(sum(group_rho.values()) / len(group_rho), 4),
                "n_groups": len(group_rho),
                "per_group": group_rho,
            }
    result["per_dimension"] = per_dim

    # Overall sum-score macro rho across groups (paper Sec 5.2 definition:
    # fluency + cleanliness + richness + max(knowledge, reasoning, artistic))
    by_group_sum: Dict[Any, Tuple[list, list]] = defaultdict(lambda: ([], []))
    for s, p in pairs:
        ps = sum_score(p, sum_spec)
        hs = sum_score(s["human_scores"], sum_spec)
        if ps is not None and hs is not None:
            g = s.get("sub_group")
            by_group_sum[g][0].append(ps)
            by_group_sum[g][1].append(hs)
    sum_rhos = [r for r in (spearman_rho(pv, gv)
                            for pv, gv in by_group_sum.values())
                if r is not None]

    # Pooled sum-score rho: all samples in one correlation — this is the
    # paper's headline GAL number (Sec 6.1: rho=0.662 on 5.6K samples).
    pooled_pred = [x for pv, _ in by_group_sum.values() for x in pv]
    pooled_gold = [x for _, gv in by_group_sum.values() for x in gv]
    pooled = spearman_rho(pooled_pred, pooled_gold) \
        if len(pooled_pred) >= 2 else None

    result["sum_score"] = {
        "pooled_rho": round(pooled, 4) if pooled is not None else None,
        "macro_rho": round(sum(sum_rhos) / len(sum_rhos), 4) if sum_rhos else None,
        "n_groups": len(sum_rhos),
    }
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/run_config.json")
    ap.add_argument("--gold", help="Gold corpus JSONL (default: config.gold_corpus)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model")
    src.add_argument("--predictions")
    src.add_argument("--teacher-scores")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seen", nargs="*", default=None,
                    help="Seen sub-groups (default: config.languages)")
    ap.add_argument("--unseen", nargs="*", default=None,
                    help="Unseen sub-groups (default: config.gal_holdout_languages)")
    ap.add_argument("--label", default="model",
                    help="Row label for the report (e.g. student, Gemini)")
    ap.add_argument("--output", help="Report JSON path")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.override)
    gold_path = args.gold or cfg.get("gold_corpus")
    if not gold_path:
        ap.error("no gold corpus: pass --gold or set gold_corpus in config")
    gold = load_gold(gold_path)

    seen = set(args.seen if args.seen is not None else cfg.get("languages", []))
    if args.unseen is not None:
        unseen = set(args.unseen)
    elif cfg.get("qeca_languages"):
        # Paper Sec 6.1: unseen = all QECA languages except MTDD targets (23)
        unseen = set(cfg["qeca_languages"]) - seen
    else:
        unseen = set(cfg.get("gal_holdout_languages", []))
    overlap = seen & unseen
    if overlap:
        logger.warning("sub-groups in BOTH seen and unseen (removed from "
                       "unseen): %s", sorted(overlap))
        unseen -= seen

    if args.model:
        preds, _ = predictions_from_model(cfg, args.model, gold,
                                          args.backend, args.batch_size)
    else:
        preds = predictions_from_scores_file(
            args.predictions or args.teacher_scores)

    seen_pairs, unseen_pairs, other_pairs = split_pairs(gold, preds, seen, unseen)
    logger.info("paired samples: seen=%d unseen=%d other=%d",
                len(seen_pairs), len(unseen_pairs), len(other_pairs))

    ordinal = cfg["ordinal_dimensions"]
    sum_spec = cfg.get("evaluation", {}).get("sum_score")
    report = {
        "label": args.label,
        "n_gold": len(gold),
        "seen": set_metrics(seen_pairs, ordinal, sum_spec),
        "unseen": set_metrics(unseen_pairs, ordinal, sum_spec),
    }
    if other_pairs:
        report["other_groups"] = set_metrics(other_pairs, ordinal, sum_spec)
        report["other_group_names"] = sorted(
            {str(s.get("sub_group")) for s, _ in other_pairs})

    # Headline (paper Sec 6.1): pooled sum-score rho on unseen languages
    seen_pooled = report["seen"].get("sum_score", {}).get("pooled_rho")
    unseen_pooled = report["unseen"].get("sum_score", {}).get("pooled_rho")
    if seen_pooled is not None and unseen_pooled is not None:
        report["sum_score_gap_seen_minus_unseen"] = \
            round(seen_pooled - unseen_pooled, 4)

    print(f"\n=== GAL report [{args.label}] ===")
    for split in ("seen", "unseen"):
        m = report[split]
        ss = m.get("sum_score", {})
        print(f"\n[{split}] n={m.get('n')}")
        for d, dm in (m.get("per_dimension") or {}).items():
            print(f"  {d:<26} macro_rho={dm['macro_rho']} "
                  f"(groups={dm['n_groups']})")
        print(f"  {'SUM-SCORE':<26} pooled_rho={ss.get('pooled_rho')} "
              f"macro_rho={ss.get('macro_rho')}")
    if "sum_score_gap_seen_minus_unseen" in report:
        print(f"\nseen-unseen pooled sum-score rho gap: "
              f"{report['sum_score_gap_seen_minus_unseen']}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("wrote report -> %s", out)


if __name__ == "__main__":
    main()
