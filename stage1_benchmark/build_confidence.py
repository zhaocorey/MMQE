"""Stage 1.4 — Build the confidence matrix from gold-corpus teacher scores.

C[language][model][dimension] = Spearman rho   (ordinal dims, 1-5)
                              = agreement rate (binary dims, e.g. toxicity)

Also records model-level averages (fallback for unseen sub-groups in CSA)
and agreement metrics (LWK/QWK) for reporting.

Input:  gold corpus JSONL + per-teacher score files from score_gold.py
Output: confidence_matrix.json — the single source of truth for Stage 2.

Usage:
    python -m stage1_benchmark.build_confidence \
        --config config/run_config.json \
        --gold /path/to/gold_corpus.jsonl \
        --scores-dir /path/to/data/run_001/gold_scores/ \
        --output /path/to/data/run_001/confidence_matrix.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config_loader import apply_overrides, load_config, teacher_names
from lib.metrics import agreement_rate, cohen_kappa, spearman_rho
from lib.scorer import iter_jsonl, sample_hash

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stage1.build_confidence")


def load_teacher_scores(scores_dir: Path,
                        names: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """{teacher: {hash: scores}} from gold_scores_<teacher>.jsonl files."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name in names:
        path = scores_dir / f"gold_scores_{name}.jsonl"
        if not path.exists():
            logger.warning("missing score file for %s: %s", name, path)
            out[name] = {}
            continue
        per_hash: Dict[str, Dict[str, Any]] = {}
        for rec in iter_jsonl(path):
            if rec.get("scores"):
                per_hash[rec["text_hash"]] = rec["scores"]
        out[name] = per_hash
        logger.info("loaded %d scored samples for %s", len(per_hash), name)
    return out


def build_matrix(cfg: Dict[str, Any],
                 gold_samples: List[Dict[str, Any]],
                 teacher_scores: Dict[str, Dict[str, Dict[str, Any]]]
                 ) -> Dict[str, Any]:
    dims: Dict[str, Dict[str, Any]] = cfg["dimensions"]
    ordinal_dims: List[str] = cfg["ordinal_dimensions"]
    binary_dims: List[str] = cfg["binary_dimensions"]

    # Collect per (group, model, dim) paired observations
    obs: Dict[tuple, List[tuple]] = defaultdict(list)  # (g, m, d) -> [(pred, gold)]
    for s in gold_samples:
        h = sample_hash(s)
        g = s.get("sub_group")
        human = s.get("human_scores") or {}
        for model, per_hash in teacher_scores.items():
            pred = per_hash.get(h)
            if not pred:
                continue
            for d, spec in dims.items():
                if d not in human or d not in pred:
                    continue
                obs[(g, model, d)].append((pred[d], human[d]))

    matrix: Dict[str, Any] = {}
    detail: Dict[str, Any] = {}
    # str() key: sub_group may be None for unlabeled samples
    for (g, model, d), pairs in sorted(
            obs.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
        preds = [p for p, _ in pairs]
        golds = [t for _, t in pairs]
        if d in ordinal_dims:
            rho = spearman_rho([float(p) for p in preds],
                               [float(t) for t in golds])
            lwk = cohen_kappa(golds, preds, weights="linear")
            qwk = cohen_kappa(golds, preds, weights="quadratic")
            conf = rho if rho is not None else 0.0
            detail.setdefault(g, {}).setdefault(model, {})[d] = {
                "rho": _r(rho), "lwk": _r(lwk), "qwk": _r(qwk), "n": len(pairs)}
        else:  # binary dims -> agreement rate
            agree = agreement_rate(golds, preds)
            conf = agree if agree is not None else 0.0
            detail.setdefault(g, {}).setdefault(model, {})[d] = {
                "agreement": _r(agree), "n": len(pairs)}

        matrix.setdefault(g, {}).setdefault(model, {})[d] = round(conf, 4)

    # Model-level average confidence (fallback for unseen sub-groups)
    model_avg: Dict[str, Dict[str, float]] = {}
    for model in teacher_scores:
        per_dim: Dict[str, List[float]] = defaultdict(list)
        for g, models in matrix.items():
            for m, dimvals in models.items():
                if m == model:
                    for d, v in dimvals.items():
                        per_dim[d].append(v)
        model_avg[model] = {d: round(sum(v) / len(v), 4)
                            for d, v in per_dim.items() if v}

    return {
        "meta": {
            "n_gold_samples": len(gold_samples),
            "teachers": sorted(teacher_scores),
            "sub_groups": sorted({g for g in matrix if g is not None}),
            "ordinal_metric": "spearman_rho",
            "binary_metric": "agreement_rate",
        },
        "confidence": matrix,
        "model_avg": model_avg,
        "detail": detail,
    }


def _r(x: Optional[float]) -> Optional[float]:
    return round(x, 4) if x is not None else None


def print_summary(result: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    matrix = result["confidence"]
    dims = list(cfg["dimensions"])
    models = result["meta"]["teachers"]
    groups = result["meta"]["sub_groups"]
    logger.info("confidence matrix: %d groups x %d models x %d dims",
                len(groups), len(models), len(dims))
    for g in groups[:3]:  # show a few sample groups
        print(f"\n[{g}]")
        header = "dim".ljust(26) + "".join(m.ljust(22) for m in models)
        print(header)
        for d in dims:
            row = d.ljust(26)
            for m in models:
                v = matrix.get(g, {}).get(m, {}).get(d)
                row += (f"{v:.3f}" if v is not None else "-").ljust(22)
            print(row)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/run_config.json")
    ap.add_argument("--gold", help="Gold corpus JSONL (default: config.gold_corpus)")
    ap.add_argument("--scores-dir", required=True,
                    help="Directory containing gold_scores_<teacher>.jsonl")
    ap.add_argument("--output",
                    help="Output JSON (default: config.confidence_matrix or "
                         "<data_root>/confidence_matrix.json)")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.override)
    gold_path = args.gold or cfg.get("gold_corpus")
    if not gold_path:
        ap.error("no gold corpus: pass --gold or set gold_corpus in config")

    gold_samples = list(iter_jsonl(gold_path))
    teacher_scores = load_teacher_scores(Path(args.scores_dir),
                                         teacher_names(cfg))
    if not any(teacher_scores.values()):
        ap.error("no teacher scores loaded — run score_gold.py first")

    result = build_matrix(cfg, gold_samples, teacher_scores)
    print_summary(result, cfg)

    out_path = Path(args.output or cfg.get("confidence_matrix") or
                    Path(cfg.get("data_root", ".")) / "confidence_matrix.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
