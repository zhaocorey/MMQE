"""Configuration loading for MMQE runs.

Loads the JSON run config, resolves paths relative to the repo root, and
provides small helpers for dimension metadata used across all stages.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a run config JSON file and normalize derived fields."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["_config_path"] = str(path)

    # Keep dimension metadata consistent with the dimensions block.
    dims: Dict[str, Dict[str, Any]] = cfg.get("dimensions", {})
    cfg.setdefault("ordinal_dimensions",
                   [d for d, spec in dims.items() if spec.get("type") == "ordinal"])
    cfg.setdefault("binary_dimensions",
                   [d for d, spec in dims.items() if spec.get("type") == "binary"])
    cfg.setdefault("all_dimensions", list(dims.keys()))
    cfg.setdefault("csa_temperature", 0.05)

    # Resolve the rubric prompt file relative to the repo root.
    rubric_file = cfg.get("rubric_prompt_file")
    if rubric_file:
        rubric_path = Path(rubric_file)
        if not rubric_path.is_absolute():
            rubric_path = REPO_ROOT / rubric_path
        cfg["rubric_prompt_path"] = str(rubric_path)

    return cfg


def load_rubric_prompt(cfg: Dict[str, Any]) -> str:
    """Load the rubric prompt text from the path recorded in the config."""
    prompt_path = cfg.get("rubric_prompt_path") or cfg.get("rubric_prompt_file")
    if not prompt_path:
        raise ValueError("Config has no rubric_prompt_file")
    p = Path(prompt_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_teacher(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return the teacher entry with the given name, or raise."""
    for t in cfg.get("teachers", []):
        if t["name"] == name:
            return t
    raise KeyError(f"Teacher {name!r} not found in config. "
                   f"Available: {[t['name'] for t in cfg.get('teachers', [])]}")


def teacher_names(cfg: Dict[str, Any]) -> List[str]:
    return [t["name"] for t in cfg.get("teachers", [])]


def apply_overrides(cfg: Dict[str, Any], overrides: List[str] | None) -> Dict[str, Any]:
    """Apply dotted-path CLI overrides like ``data_root=/tmp/run``."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        # Try JSON parse so numbers/bools/lists work; fall back to string.
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = parsed
    return cfg
