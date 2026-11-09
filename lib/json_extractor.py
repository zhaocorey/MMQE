"""Robust JSON extraction from raw LLM outputs.

Implements the layered strategy from the MTDD skill:
1. Strip <thinking>/<reasoning> blocks.
2. Try a fenced ```json code block.
3. Fall back to locating a JSON object containing a known key.
4. Fall back to brace-matching the first top-level JSON object.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

_THINKING_RE = re.compile(
    r"<(thinking|reasoning|think)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove thinking/reasoning tags and markdown fences noise."""
    return _THINKING_RE.sub("", text or "")


def _loads(candidate: str) -> Optional[Dict[str, Any]]:
    candidate = candidate.strip()
    if not candidate:
        return None
    # Some models emit trailing commas or single quotes; try strict first.
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _find_object_with_key(text: str, key: str) -> Optional[Dict[str, Any]]:
    pattern = re.compile(r"\{[^{}]*\"" + re.escape(key) + r"\"[^{}]*\}", re.DOTALL)
    for m in pattern.finditer(text):
        obj = _loads(m.group(0))
        if obj is not None:
            return obj
    return None


def _find_first_balanced_object(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    obj = _loads(text[start:i + 1])
                    if obj is not None:
                        return obj
                    break
        start = text.find("{", start + 1)
    return None


def extract_json(llm_output: str,
                 first_key_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from raw LLM output. Returns None on failure."""
    if not llm_output:
        return None
    output = strip_thinking(llm_output)

    # 1. JSON code fence
    for m in _CODE_FENCE_RE.finditer(output):
        obj = _loads(m.group(1))
        if obj is not None:
            return obj

    # 2. Object containing a known key
    if first_key_hint:
        obj = _find_object_with_key(output, first_key_hint)
        if obj is not None:
            return obj

    # 3. Whole-output parse, then first balanced object
    obj = _loads(output)
    if obj is not None:
        return obj
    return _find_first_balanced_object(output)


def coerce_scores(raw: Dict[str, Any],
                  dimensions: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate and coerce an extracted JSON dict into canonical scores.

    - Ordinal dims: cast to int, clamp to the scale range.
    - Binary dims: normalize to one of the declared string values
      (lowercased match); anything else -> None.
    Returns None if any required dimension is missing/invalid.
    """
    if not raw:
        return None
    scores: Dict[str, Any] = {}
    for dim, spec in dimensions.items():
        if dim not in raw:
            return None
        value = raw[dim]
        if spec.get("type") == "ordinal":
            lo, hi = spec.get("scale", [1, 5])
            try:
                ivalue = int(round(float(value)))
            except (TypeError, ValueError):
                return None
            scores[dim] = max(lo, min(hi, ivalue))
        elif spec.get("type") == "binary":
            allowed: List[str] = spec.get("values", [])
            sval = str(value).strip().lower()
            if sval not in allowed:
                return None
            scores[dim] = sval
        else:
            scores[dim] = value
    return scores
