"""Stage 2.3/2.4 — CSA (Confidence-based Score Aggregation) over teacher scores.

    w_i = softmax(rho_i / tau),   agg[d] = sum_i s_i * w_i

Rules by dimension type:
  - ordinal  -> softmax-weighted average, rounded to integer (clamped to scale)
  - binary   -> confidence-weighted majority vote over string values
  - unseen sub-group fallback -> model_avg confidence from the matrix
  - missing teacher score    -> re-normalize over available teachers

Ablation modes (--mode):
  - softmax  (default, tau from config, paper setting 0.05)
  - uniform  (simple average / majority vote, all teachers equal)
  - linear   (w_i = rho_i / sum rho_j)
  - single:<TeacherName> (use one teacher's raw scores — best-single baseline)

Usage:
    python -m stage2_mtdd.csa_aggregate --config config/run_config.json \
        --scores-dir <data_root>/scores/ \
        --confidence <data_root>/confidence_matrix.json \
        --raw-dir /path/to/raw_data/ \
        --output <data_root>/mtdd_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config_loader import apply_overrides, load_config
from lib.scorer import iter_jsonl, sample_hash

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stage2.csa_aggregate")


# --------------------------------------------------------------------------- #
# Core CSA
# --------------------------------------------------------------------------- #

def softmax_weights(rhos: List[float], tau: float) -> List[float]:
    """Numerically stable softmax over confidence values."""
    if not rhos:
        return []
    if tau <= 0:  # winner-take-all limit
        best = max(rhos)
        winners = [r == best for r in rhos]
        n = sum(winners)
        return [1.0 / n if w else 0.0 for w in winners]
    m = max(rhos)
    exps = [math.exp((r - m) / tau) for r in rhos]
    s = sum(exps)
    return [e / s for e in exps]


def linear_weights(rhos: List[float]) -> List[float]:
    """w_i = rho_i / sum(rho_j); negatives clipped to 0 (paper-style)."""
    clipped = [max(0.0, r) for r in rhos]
    s = sum(clipped)
    if s <= 0:
        return [1.0 / len(rhos)] * len(rhos) if rhos else []
    return [c / s for c in clipped]


class CSAAggregator:
    def __init__(self,
                 confidence: Dict[str, Any],
                 dimensions: Dict[str, Dict[str, Any]],
                 tau: float = 0.05,
                 mode: str = "softmax"):
        self.matrix: Dict[str, Any] = confidence.get("confidence", {})
        self.model_avg: Dict[str, Any] = confidence.get("model_avg", {})
        self.dimensions = dimensions
        self.tau = tau
        self.mode = mode
        self.single_teacher: Optional[str] = None
        if mode.startswith("single:"):
            self.single_teacher = mode.split(":", 1)[1]
        elif mode not in ("softmax", "uniform", "linear"):
            raise ValueError(f"Unknown mode {mode!r}")

    # -- confidence lookup with fallback ---------------------------------- #
    def rho(self, model: str, sub_group: Any, dim: str) -> float:
        g = self.matrix.get(sub_group, {})
        if model in g and dim in g[model]:
            return float(g[model][dim])
        # Fallback: model-level average (unseen sub-group)
        avg = self.model_avg.get(model, {}).get(dim)
        if avg is not None:
            return float(avg)
        return 0.0

    def weights(self, models: List[str], sub_group: Any, dim: str) -> List[float]:
        if self.mode == "uniform":
            return [1.0 / len(models)] * len(models) if models else []
        rhos = [self.rho(m, sub_group, d=dim) for m in models]
        if self.mode == "linear":
            return linear_weights(rhos)
        return softmax_weights(rhos, self.tau)

    # -- per-dimension aggregation ---------------------------------------- #
    def aggregate_ordinal(self, sample_scores: Dict[str, float],
                          sub_group: Any, dim: str) -> Optional[int]:
        models = [m for m, s in sample_scores.items() if s is not None]
        if not models:
            return None
        if self.single_teacher:
            v = sample_scores.get(self.single_teacher)
            if v is None:
                return None
            lo, hi = self.dimensions[dim].get("scale", [1, 5])
            return max(lo, min(hi, int(round(v))))
        w = self.weights(models, sub_group, dim)
        agg = sum(sample_scores[m] * wi for m, wi in zip(models, w))
        lo, hi = self.dimensions[dim].get("scale", [1, 5])
        return max(lo, min(hi, int(round(agg))))

    def aggregate_binary(self, sample_scores: Dict[str, str],
                         sub_group: Any, dim: str) -> Optional[str]:
        models = [m for m, s in sample_scores.items() if s is not None]
        if not models:
            return None
        if self.single_teacher:
            return sample_scores.get(self.single_teacher)
        # Confidence-weighted majority vote: weights must be computed jointly
        # over all available models (a single-model softmax is trivially 1.0).
        weights = self.weights(models, sub_group, dim)
        votes: Dict[str, float] = defaultdict(float)
        for m, w in zip(models, weights):
            votes[sample_scores[m]] += w
        return max(votes.items(), key=lambda kv: kv[1])[0]

    def aggregate_sample(self,
                         per_teacher: Dict[str, Optional[Dict[str, Any]]],
                         sub_group: Any) -> Tuple[Dict[str, Any], List[str]]:
        """Aggregate one sample across teachers. Returns (scores, teachers_used)."""
        scores: Dict[str, Any] = {}
        used = set()
        for dim, spec in self.dimensions.items():
            sample_scores = {}
            for model, s in per_teacher.items():
                if self.single_teacher and model != self.single_teacher:
                    continue
                if s and dim in s:
                    sample_scores[model] = s[dim]
            if spec.get("type") == "binary":
                v = self.aggregate_binary(sample_scores, sub_group, dim)
            else:
                v = self.aggregate_ordinal(
                    {m: float(x) for m, x in sample_scores.items()},
                    sub_group, dim)
            if v is not None:
                scores[dim] = v
                used.update(sample_scores.keys())
        return scores, sorted(used)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def load_score_index(scores_dir: Path,
                     teachers: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """{hash: {teacher: scores}} across all teacher score files."""
    index: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for name in teachers:
        path = scores_dir / f"scores_{name}.jsonl"
        if not path.exists():
            logger.warning("missing %s", path)
            continue
        n = 0
        for rec in iter_jsonl(path):
            if rec.get("scores") is not None:
                index[rec["text_hash"]][name] = rec["scores"]
                n += 1
        logger.info("loaded %d scored samples from %s", n, path.name)
    return dict(index)


def load_raw_index(raw_dir: Path) -> Dict[str, Dict[str, Any]]:
    """{hash: {text, sub_group, domain}} for attaching text to aggregates."""
    index: Dict[str, Dict[str, Any]] = {}
    for fp in sorted(raw_dir.rglob("*.jsonl")):
        for rec in iter_jsonl(fp):
            index[sample_hash(rec)] = {
                "text": rec.get("text", ""),
                "sub_group": rec.get("sub_group"),
                "domain": rec.get("domain"),
            }
    for fp in sorted(raw_dir.rglob("*.json")):
        for rec in iter_jsonl(fp):
            index.setdefault(sample_hash(rec), {
                "text": rec.get("text", ""),
                "sub_group": rec.get("sub_group"),
                "domain": rec.get("domain"),
            })
    return index


def aggregate_dataset(cfg: Dict[str, Any],
                      confidence: Dict[str, Any],
                      score_index: Dict[str, Dict[str, Dict[str, Any]]],
                      raw_index: Dict[str, Dict[str, Any]],
                      mode: str = "softmax",
                      require_teachers: int = 1) -> Tuple[List[Dict[str, Any]], Counter]:
    agg = CSAAggregator(confidence, cfg["dimensions"],
                        tau=float(cfg.get("csa_temperature", 0.05)), mode=mode)
    out: List[Dict[str, Any]] = []
    stats = Counter()
    stats["total_scored_hashes"] = len(score_index)

    for h, per_teacher in score_index.items():
        if len(per_teacher) < require_teachers:
            stats["dropped_few_teachers"] += 1
            continue
        raw = raw_index.get(h, {})
        sub_group = raw.get("sub_group")
        # fall back to sub_group recorded in score files if raw missing
        scores, used = agg.aggregate_sample(per_teacher, sub_group)
        if not scores:
            stats["dropped_no_valid_scores"] += 1
            continue
        stats["aggregated"] += 1
        stats[f"teachers_{len(used)}"] += 1
        out.append({
            "text": raw.get("text", ""),
            "text_hash": h,
            "sub_group": sub_group,
            "domain": raw.get("domain"),
            "scores": scores,
            "teacher_count": len(used),
            "teachers_used": used,
        })
    return out, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/run_config.json")
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--confidence", required=True,
                    help="confidence_matrix.json from Stage 1")
    ap.add_argument("--raw-dir", required=True,
                    help="Raw data dir (to attach text/sub_group/domain)")
    ap.add_argument("--output", help="MTDD JSONL (default: config.mtdd_output)")
    ap.add_argument("--mode", default="softmax",
                    choices=["softmax", "uniform", "linear"],
                    help="Aggregation mode; use --single-teacher for ablation")
    ap.add_argument("--single-teacher",
                    help="Ablation: use only this teacher's scores (mode=single)")
    ap.add_argument("--tau", type=float, help="Override csa_temperature")
    ap.add_argument("--require-teachers", type=int, default=1,
                    help="Minimum teachers with valid scores per sample")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.override)
    if args.tau is not None:
        cfg["csa_temperature"] = args.tau

    with open(args.confidence, "r", encoding="utf-8") as f:
        confidence = json.load(f)

    teachers = [t["name"] for t in cfg["teachers"]]
    if args.single_teacher:
        teachers = [args.single_teacher]
    score_index = load_score_index(Path(args.scores_dir), teachers)
    raw_index = load_raw_index(Path(args.raw_dir))

    mode = f"single:{args.single_teacher}" if args.single_teacher else args.mode
    logger.info("CSA mode=%s tau=%s", mode, cfg["csa_temperature"])

    records, stats = aggregate_dataset(
        cfg, confidence, score_index, raw_index,
        mode=mode, require_teachers=args.require_teachers)

    out_path = Path(args.output or cfg.get("mtdd_output") or
                    Path(cfg.get("data_root", ".")) / "mtdd_dataset.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("wrote %d MTDD samples -> %s", len(records), out_path)
    logger.info("stats: %s", dict(stats))

    # Score distribution report (sanity check for aggregation collapse)
    for dim in cfg["ordinal_dimensions"]:
        dist = Counter(r["scores"][dim] for r in records if dim in r["scores"])
        logger.info("  %-26s %s", dim,
                    {k: dist[k] for k in sorted(dist)})


if __name__ == "__main__":
    main()
