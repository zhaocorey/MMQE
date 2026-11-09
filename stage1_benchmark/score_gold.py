"""Stage 1.3 — Score the human-annotated gold corpus with each teacher.

Gold corpus format (JSONL), one sample per line:
    {"text": str, "sub_group": "Arabic", "domain": "encyclopedia",
     "human_scores": {"text_fluency": 4, ..., "text_toxicity": "yes"}}

Usage:
    python -m stage1_benchmark.score_gold --config config/run_config.json \
        --gold /path/to/gold_corpus.jsonl \
        --output-dir /path/to/data/run_001/gold_scores/

    # Score a single teacher (for parallel shell invocation):
    python -m stage1_benchmark.score_gold --config config/run_config.json \
        --teacher Gemini-3.7-Flash

Outputs one JSONL per teacher: <output-dir>/gold_scores_<teacher>.jsonl
Resume-safe: re-running skips hashes already present in the output file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config_loader import apply_overrides, get_teacher, load_config, teacher_names
from lib.scorer import iter_jsonl, run_scoring

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stage1.score_gold")


def validate_gold_corpus(samples, dimensions, min_per_group: int = 100,
                         l4_dims=("knowledge_value", "reasoning_value",
                                  "artistic_cultural_value"),
                         collapsed_key: str = "education_value"):
    """Sanity-check the gold corpus; warn (not fail) on small sub-groups.

    QECA annotators rate L4 as a single collapsed "Education Value"
    (paper Sec 4.1), so a sample carrying `education_value` instead of the
    three L4 keys is valid.
    """
    counts = {}
    bad = 0
    for s in samples:
        g = s.get("sub_group")
        counts[g] = counts.get(g, 0) + 1
        hs = s.get("human_scores") or {}
        missing = [d for d in dimensions if d not in hs]
        if missing and collapsed_key in hs:
            missing = [d for d in missing if d not in l4_dims]
        if missing:
            bad += 1
            if bad <= 5:
                logger.warning("sample missing human dims %s: %.60s...",
                               missing, s.get("text", ""))
    logger.info("gold corpus: %d samples, %d sub-groups, %d with missing dims",
                len(samples), len(counts), bad)
    small = {g: c for g, c in counts.items() if c < min_per_group}
    if small:
        logger.warning("sub-groups with < %d samples (unstable rho): %s",
                       min_per_group, small)
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/run_config.json")
    ap.add_argument("--gold", help="Gold corpus JSONL (default: config.gold_corpus)")
    ap.add_argument("--output-dir",
                    help="Directory for per-teacher score files "
                         "(default: <data_root>/gold_scores/)")
    ap.add_argument("--teacher", help="Score only this teacher (by name)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap samples per teacher (smoke tests)")
    ap.add_argument("-o", "--override", action="append", default=[],
                    help="Dotted config override, e.g. csa_temperature=0.1")
    args = ap.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.override)
    gold_path = args.gold or cfg.get("gold_corpus")
    if not gold_path:
        ap.error("no gold corpus: pass --gold or set gold_corpus in config")

    samples = list(iter_jsonl(gold_path))
    validate_gold_corpus(samples, cfg["dimensions"])

    out_dir = Path(args.output_dir or
                   Path(cfg.get("data_root", ".")) / "gold_scores")
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [args.teacher] if args.teacher else teacher_names(cfg)
    for name in names:
        tcfg = get_teacher(cfg, name)
        out_path = out_dir / f"gold_scores_{name}.jsonl"
        logger.info("=== scoring gold corpus with %s -> %s", name, out_path)
        stats = run_scoring(cfg, tcfg, samples, out_path,
                            max_samples=args.max_samples)
        logger.info("%s stats: %s", name, stats)


if __name__ == "__main__":
    main()
