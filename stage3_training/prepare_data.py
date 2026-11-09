"""Stage 3 — Convert the MTDD dataset into SFT training data.

Input : mtdd_dataset.jsonl (from csa_aggregate.py)
Output: train.jsonl / eval.jsonl in chat format:

    {"messages": [
        {"role": "system",    "content": "<English rubric prompt>"},
        {"role": "user",      "content": "<document text>"},
        {"role": "assistant", "content": "{\"text_fluency\": 4, ...}"}]}

Key decisions implemented here (per MMQE skill):
- Rubric prompt is ALWAYS English, even for non-English documents.
- Assistant target is compact JSON (~100 tokens) with dimensions in a
  stable order.
- GAL support: sub-groups listed in config.gal_holdout_languages are
  excluded from training and written to a separate heldout file so the
  student can be evaluated on unseen sub-groups (Stage 4).

Usage:
    python -m stage3_training.prepare_data --config config/run_config.json \
        --mtdd <data_root>/mtdd_dataset.jsonl \
        --output-dir <data_root>/sft/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config_loader import apply_overrides, load_config, load_rubric_prompt
from lib.scorer import iter_jsonl

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stage3.prepare_data")


def make_target(scores: Dict[str, Any], dim_order: List[str]) -> str:
    """Compact JSON target with stable key order (ordinal first, then binary)."""
    ordered = {d: scores[d] for d in dim_order if d in scores}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def to_chat_sample(record: Dict[str, Any], rubric: str,
                   dim_order: List[str]) -> Dict[str, Any] | None:
    text = (record.get("text") or "").strip()
    scores = record.get("scores") or {}
    if not text or not scores:
        return None
    return {
        "messages": [
            {"role": "system", "content": rubric},
            {"role": "user", "content": text},
            {"role": "assistant", "content": make_target(scores, dim_order)},
        ],
        "meta": {
            "text_hash": record.get("text_hash"),
            "sub_group": record.get("sub_group"),
            "domain": record.get("domain"),
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/run_config.json")
    ap.add_argument("--mtdd", help="MTDD dataset (default: config.mtdd_output)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--holdout-languages", nargs="*", default=None,
                    help="Sub-groups excluded from training (GAL study). "
                         "Default: config.gal_holdout_languages")
    ap.add_argument("--eval-frac", type=float, default=None,
                    help="Fraction of seen-group data for eval split "
                         "(default: 1 - student.train_split)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.override)
    mtdd_path = args.mtdd or cfg.get("mtdd_output")
    if not mtdd_path:
        ap.error("no MTDD dataset: pass --mtdd or set mtdd_output in config")

    rubric = load_rubric_prompt(cfg)
    dim_order = cfg["ordinal_dimensions"] + cfg["binary_dimensions"]
    if args.holdout_languages is not None:
        holdout = set(args.holdout_languages)
    elif cfg.get("gal_holdout_languages"):
        holdout = set(cfg["gal_holdout_languages"])
    elif cfg.get("qeca_languages"):
        # Paper Sec 6.1: unseen = QECA languages minus MTDD target languages
        holdout = set(cfg["qeca_languages"]) - set(cfg.get("languages", []))
    else:
        holdout = set()
    seed = args.seed if args.seed is not None else \
        int(cfg.get("student", {}).get("seed", 42))
    eval_frac = args.eval_frac if args.eval_frac is not None else \
        1.0 - float(cfg.get("student", {}).get("train_split", 0.98))
    rng = random.Random(seed)

    seen: List[Dict[str, Any]] = []
    unseen: List[Dict[str, Any]] = []
    n_skipped = 0
    for rec in iter_jsonl(mtdd_path):
        chat = to_chat_sample(rec, rubric, dim_order)
        if chat is None:
            n_skipped += 1
            continue
        if rec.get("sub_group") in holdout:
            unseen.append(chat)
        else:
            seen.append(chat)
    logger.info("converted %d seen-group + %d heldout-group samples "
                "(%d skipped)", len(seen), len(unseen), n_skipped)

    rng.shuffle(seen)
    n_eval = int(len(seen) * eval_frac)
    eval_set, train_set = seen[:n_eval], seen[n_eval:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(path: Path, rows: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("wrote %d rows -> %s", len(rows), path)

    dump(out_dir / "train.jsonl", train_set)
    dump(out_dir / "eval.jsonl", eval_set)
    dump(out_dir / "heldout_groups.jsonl", unseen)  # GAL evaluation set

    # Distribution report
    from collections import Counter
    group_dist = Counter(c["meta"]["sub_group"] for c in train_set)
    logger.info("train sub-group distribution (top 10): %s",
                group_dist.most_common(10))


if __name__ == "__main__":
    main()
