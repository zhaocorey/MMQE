"""Stage 2.2 — Score the raw dataset with each teacher (embarrassingly parallel).

Raw data format (JSONL): {"text": str, "sub_group": str, "domain": str,
                          ["text_hash": str]}
If text_hash is absent it is computed as MD5(text).

Usage (sequential, all teachers):
    python -m stage2_mtdd.score_raw --config config/run_config.json

Usage (shell-parallel, one process per teacher — recommended for 1.69M samples):
    python -m stage2_mtdd.score_raw --config config/run_config.json --teacher Gemini-3.7-Flash &
    python -m stage2_mtdd.score_raw --config config/run_config.json --teacher Qwen3.8-Flash &
    python -m stage2_mtdd.score_raw --config config/run_config.json --teacher DeepSeek-V4-Flash &
    wait

Outputs: <data_root>/scores/scores_<teacher>.jsonl   (resume-safe by hash)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config_loader import apply_overrides, get_teacher, load_config, teacher_names
from lib.scorer import iter_jsonl, run_scoring

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stage2.score_raw")


def load_raw_dataset(raw_dir: str | Path,
                     min_chars: int = 50,
                     dedupe: bool = True) -> List[Dict]:
    """Load and curate raw JSONL files from a directory.

    Curation (Step 2.1): minimum-length filter + text-hash deduplication.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.rglob("*.jsonl"))
    if not files:
        files = sorted(raw_dir.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .jsonl/.json files under {raw_dir}")

    samples: List[Dict] = []
    seen = set()
    n_short = n_dup = 0
    for fp in files:
        for rec in iter_jsonl(fp):
            text = (rec.get("text") or "").strip()
            if len(text) < min_chars:
                n_short += 1
                continue
            h = rec.get("text_hash") or \
                hashlib.md5(text.encode("utf-8")).hexdigest()
            if dedupe and h in seen:
                n_dup += 1
                continue
            seen.add(h)
            samples.append({"text": text, "text_hash": h,
                            "sub_group": rec.get("sub_group"),
                            "domain": rec.get("domain")})
    logger.info("raw dataset: %d kept, %d too short, %d duplicates",
                len(samples), n_short, n_dup)

    # Balance report per sub-group (informational)
    counts: Dict[str, int] = {}
    for s in samples:
        counts[s["sub_group"]] = counts.get(s["sub_group"], 0) + 1
    for g, c in sorted(counts.items(), key=lambda x: -x[1]):
        logger.info("  %-20s %8d", g, c)
    return samples


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/run_config.json")
    ap.add_argument("--raw-dir", help="Raw data dir (default: config.raw_data_dir)")
    ap.add_argument("--output-dir",
                    help="Score output dir (default: <data_root>/scores/)")
    ap.add_argument("--teacher", help="Score only this teacher (by name)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.override)
    raw_dir = args.raw_dir or cfg.get("raw_data_dir")
    if not raw_dir:
        ap.error("no raw data dir: pass --raw-dir or set raw_data_dir in config")

    samples = load_raw_dataset(raw_dir, min_chars=args.min_chars,
                               dedupe=not args.no_dedupe)
    out_dir = Path(args.output_dir or Path(cfg.get("data_root", ".")) / "scores")
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [args.teacher] if args.teacher else teacher_names(cfg)
    for name in names:
        tcfg = get_teacher(cfg, name)
        out_path = out_dir / f"scores_{name}.jsonl"
        logger.info("=== scoring raw data with %s -> %s", name, out_path)
        stats = run_scoring(cfg, tcfg, samples, out_path,
                            max_samples=args.max_samples)
        logger.info("%s stats: %s", name, stats)


if __name__ == "__main__":
    main()
