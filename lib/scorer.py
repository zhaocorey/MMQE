"""Batch scoring engine shared by Stage 1 (gold corpus) and Stage 2 (raw data).

Responsibilities:
- read JSONL samples ({"text", "text_hash", "sub_group", ...})
- score each sample with one teacher via AsyncLLMClient
- extract + coerce JSON scores with lib.json_extractor
- resume support: samples whose hash already exists in the output file
  are skipped, so crashes/interrupts never re-spend API budget
- periodic flushing + final stats (throughput, failure rate)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config_loader import load_rubric_prompt
from .json_extractor import coerce_scores, extract_json
from .llm_client import AsyncLLMClient, build_messages

logger = logging.getLogger(__name__)


def sample_hash(sample: Dict[str, Any]) -> str:
    """Use provided hash, else derive one from the text."""
    h = sample.get("text_hash")
    if h:
        return str(h)
    import hashlib
    return hashlib.md5(sample["text"].encode("utf-8")).hexdigest()


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done_hashes(output_path: str | Path) -> set:
    """Hashes already *successfully* scored in a (possibly partial) output file.

    Records with scores=None (API/parse failures) are NOT counted as done,
    so a re-run retries them. Failed records are rewritten on success —
    readers should keep the last occurrence per hash.
    """
    done = set()
    failed = set()
    p = Path(output_path)
    if not p.exists():
        return done
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a truncated last line from a crash
            h = rec.get("text_hash")
            if h is None:
                continue
            if rec.get("scores") is not None:
                done.add(h)
                failed.discard(h)
            else:
                failed.add(h)
    return done - failed


async def score_samples(cfg: Dict[str, Any],
                        teacher_cfg: Dict[str, Any],
                        samples: List[Dict[str, Any]],
                        output_path: str | Path,
                        max_samples: Optional[int] = None) -> Dict[str, Any]:
    """Score `samples` with one teacher, appending results to output_path.

    Output records: {"text_hash", "sub_group", "teacher", "scores", "raw_output"?}
    where scores is None when extraction/validation failed.
    """
    rubric = load_rubric_prompt(cfg)
    dimensions = cfg["dimensions"]
    first_dim = next(iter(dimensions))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_hashes(output_path)
    todo = [s for s in samples if sample_hash(s) not in done]
    if max_samples is not None:
        todo = todo[:max_samples]
    logger.info("teacher=%s total=%d already_done=%d todo=%d",
                teacher_cfg["name"], len(samples), len(samples) - len(todo), len(todo))

    if not todo:
        return {"teacher": teacher_cfg["name"], "scored": 0,
                "skipped_existing": len(samples), "parse_failures": 0}

    parse_failures = 0
    scored = 0
    lock = asyncio.Lock()
    flush_every = 50
    buffer: List[str] = []
    t0 = time.monotonic()

    async def process(client: AsyncLLMClient, sample: Dict[str, Any]) -> None:
        nonlocal parse_failures, scored
        h = sample_hash(sample)
        messages = build_messages(rubric, sample["text"])
        raw = await client.chat(messages)
        scores = None
        if raw is not None:
            extracted = extract_json(raw, first_key_hint=first_dim)
            scores = coerce_scores(extracted, dimensions) if extracted else None
        async with lock:
            if scores is None:
                parse_failures += 1
            else:
                scored += 1
            rec = {"text_hash": h,
                   "sub_group": sample.get("sub_group"),
                   "domain": sample.get("domain"),
                   "teacher": teacher_cfg["name"],
                   "scores": scores}
            buffer.append(json.dumps(rec, ensure_ascii=False))
            if len(buffer) >= flush_every:
                _flush(buffer, output_path)

    # Bounded worker pool: creating one Task per sample would exhaust memory
    # on million-scale datasets. Concurrency is also capped by the client's
    # semaphore; the pool size keeps queueing overhead low.
    cursor = 0

    async def worker_fn(client: AsyncLLMClient) -> None:
        nonlocal cursor
        while True:
            async with lock:
                if cursor >= len(todo):
                    return
                sample = todo[cursor]
                cursor += 1
            await process(client, sample)

    n_workers = min(len(todo), int(teacher_cfg.get("concurrent_workers", 32)) * 2)
    async with AsyncLLMClient(teacher_cfg, cfg.get("inference")) as client:
        workers = [asyncio.create_task(worker_fn(client))
                   for _ in range(n_workers)]
        try:
            await asyncio.gather(*workers)
        finally:
            for t in workers:
                t.cancel()
            async with lock:
                _flush(buffer, output_path)

    elapsed = time.monotonic() - t0
    stats = dict(client.stats)
    stats.update({
        "teacher": teacher_cfg["name"],
        "scored": scored,
        "skipped_existing": len(samples) - len(todo),
        "parse_failures": parse_failures,
        "elapsed_s": round(elapsed, 1),
        "throughput_rps": round(scored / elapsed, 2) if elapsed > 0 else 0.0,
    })
    logger.info("done: %s", stats)
    return stats


def _flush(buffer: List[str], output_path: Path) -> None:
    if not buffer:
        return
    with open(output_path, "a", encoding="utf-8") as f:
        f.write("\n".join(buffer) + "\n")
    buffer.clear()


def run_scoring(cfg: Dict[str, Any],
                teacher_cfg: Dict[str, Any],
                samples: List[Dict[str, Any]],
                output_path: str | Path,
                max_samples: Optional[int] = None) -> Dict[str, Any]:
    """Synchronous wrapper around score_samples."""
    return asyncio.run(
        score_samples(cfg, teacher_cfg, samples, output_path, max_samples))
