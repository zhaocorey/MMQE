"""Evaluation metrics: Spearman rho, Cohen's kappa (unweighted / linear /
quadratic weighted), agreement rate, and F1 at a precision threshold.

All functions take parallel lists/arrays of numbers (or strings for binary)
and are dependency-light: scipy is optional (pure-python Spearman fallback).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from scipy.stats import spearmanr
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


# --------------------------------------------------------------------------- #
# Rank correlation
# --------------------------------------------------------------------------- #

def _rankdata(values: Sequence[float]) -> List[float]:
    """Average ranks for ties (fractional ranking)."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation. Returns None if degenerate."""
    if len(x) != len(y) or len(x) < 2:
        return None
    if _HAS_SCIPY:
        rho = spearmanr(x, y).statistic
        return None if math.isnan(rho) else float(rho)
    rx, ry = _rankdata(list(x)), _rankdata(list(y))
    return pearson_r(rx, ry)


def pearson_r(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


# --------------------------------------------------------------------------- #
# Cohen's kappa family
# --------------------------------------------------------------------------- #

def cohen_kappa(y_true: Sequence[Any], y_pred: Sequence[Any],
                weights: Optional[str] = None,
                labels: Optional[Sequence[Any]] = None) -> Optional[float]:
    """Cohen's kappa.

    weights: None -> unweighted; "linear" -> LWK; "quadratic" -> QWK.
    labels:  explicit label set (e.g. 1..5); inferred from data if absent.
    """
    if len(y_true) != len(y_pred) or not y_true:
        return None
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    labels = list(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    if k < 2:
        return None

    conf = [[0.0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            conf[idx[t]][idx[p]] += 1.0
    n = sum(sum(row) for row in conf)
    if n == 0:
        return None

    row = [sum(conf[i]) / n for i in range(k)]
    col = [sum(conf[i][j] for i in range(k)) / n for j in range(k)]

    if weights is None:
        w = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    elif weights == "linear":
        w = [[1.0 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    elif weights == "quadratic":
        w = [[1.0 - ((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)]
             for i in range(k)]
    else:
        raise ValueError(f"Unknown weights={weights!r}")

    po = sum(w[i][j] * conf[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(w[i][j] * row[i] * col[j] for i in range(k) for j in range(k))
    if pe >= 1.0:
        return 1.0 if abs(po - pe) < 1e-12 else None
    return (po - pe) / (1.0 - pe)


def agreement_rate(y_true: Sequence[Any], y_pred: Sequence[Any]) -> Optional[float]:
    if not y_true or len(y_true) != len(y_pred):
        return None
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


# --------------------------------------------------------------------------- #
# Sum score (paper Sec 5.2)
# --------------------------------------------------------------------------- #
# sum_score = text_fluency + text_cleanliness + information_richness
#             + Education_Value
# Education_Value = max(knowledge_value, reasoning_value, artistic_cultural_value)
# (QECA annotators rate L4 as a single "Education Value" dimension; models
#  may emit either the collapsed key or the three L4 keys.)

def education_value(scores: Dict[str, Any], spec: Dict[str, Any]) -> Optional[float]:
    """Education Value = collapsed key if present, else max over L4 dims."""
    key = spec.get("education_value_key", "education_value")
    if key in scores and scores[key] is not None:
        try:
            return float(scores[key])
        except (TypeError, ValueError):
            return None
    vals = []
    for d in spec.get("max_of", []):
        if d in scores and scores[d] is not None:
            try:
                vals.append(float(scores[d]))
            except (TypeError, ValueError):
                continue
    return max(vals) if vals else None


def sum_score(scores: Dict[str, Any], spec: Dict[str, Any]) -> Optional[float]:
    """Paper sum score: additive dims + Education_Value. None if incomplete."""
    if not scores:
        return None
    parts: List[float] = []
    for d in spec.get("additive", []):
        if d not in scores or scores[d] is None:
            return None
        try:
            parts.append(float(scores[d]))
        except (TypeError, ValueError):
            return None
    ev = education_value(scores, spec)
    if ev is None:
        return None
    parts.append(ev)
    return sum(parts)


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (q in [0,1]); numpy 'linear' method."""
    s = sorted(values)
    if not s:
        return None
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] * (c - k) + s[c] * (k - f))


def f1_at_80pct(model_sums: Sequence[float], model_nontoxic: Sequence[bool],
                human_sums: Sequence[float], human_nontoxic: Sequence[bool],
                pct: float = 0.20) -> Dict[str, Any]:
    """F1@80% per paper Appendix B.

    Each side computes its own sum score; "pass" = (sum >= own P{pct*100}
    percentile) AND non-toxic. Precision/Recall/F1 of model-pass vs
    human-pass. The P20 cutoff yields an ~80% pass rate before the
    toxicity gate.
    """
    default = {"model_p20": None, "human_p20": None, "precision": 0.0,
               "recall": 0.0, "f1": 0.0}
    n = len(model_sums)
    if n == 0 or n != len(human_sums):
        return default
    m_thresh = percentile(model_sums, pct)
    h_thresh = percentile(human_sums, pct)
    model_pass = [s >= m_thresh and bool(nt)
                  for s, nt in zip(model_sums, model_nontoxic)]
    human_pass = [s >= h_thresh and bool(nt)
                  for s, nt in zip(human_sums, human_nontoxic)]
    tp = sum(1 for a, b in zip(model_pass, human_pass) if a and b)
    n_mp = sum(model_pass)
    n_hp = sum(human_pass)
    precision = tp / n_mp if n_mp else 0.0
    recall = tp / n_hp if n_hp else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"model_p20": m_thresh, "human_p20": h_thresh,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4),
            "model_pass_rate": round(n_mp / n, 4),
            "human_pass_rate": round(n_hp / n, 4)}


# --------------------------------------------------------------------------- #
# Classification at a precision threshold (F1@P-threshold)
# --------------------------------------------------------------------------- #

def f1_at_precision_threshold(scores: Sequence[float],
                              labels: Sequence[int],
                              target_precision: float = 0.8) -> Dict[str, float]:
    """Find the highest-recall threshold whose precision >= target.

    Convention: label 1 = "keep / high quality", prediction = score >= thresh.
    Single sorted sweep over distinct thresholds (ties processed together),
    O(n log n) — safe for 10K+ gold samples.
    Returns {"threshold", "precision", "recall", "f1"}.
    """
    n = len(scores)
    default = {"threshold": float("nan"), "precision": 0.0,
               "recall": 0.0, "f1": 0.0}
    if n == 0 or len(labels) != n:
        return default
    total_pos = sum(1 for l in labels if l == 1)
    if total_pos == 0 or total_pos == n:
        return default

    # Descending by score; sweep keeps cumulative tp / pred_pos
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    best: Optional[Dict[str, float]] = None
    tp = 0
    i = 0
    while i < n:
        j = i
        cur = pairs[i][0]
        while j < n and pairs[j][0] == cur:
            tp += 1 if pairs[j][1] == 1 else 0
            j += 1
        pred_pos = j  # everything with score >= cur
        precision = tp / pred_pos
        recall = tp / total_pos
        if precision >= target_precision:
            f1 = (2 * precision * recall / (precision + recall)
                  if precision + recall > 0 else 0.0)
            cand = {"threshold": float(cur), "precision": precision,
                    "recall": recall, "f1": f1}
            if best is None or (cand["recall"], cand["f1"]) > \
                    (best["recall"], best["f1"]):
                best = cand
        i = j
    return best or default


def binary_f1(y_true: Sequence[Any], y_pred: Sequence[Any],
              positive: Any = "no") -> Dict[str, float]:
    """P/R/F1 for a binary dimension; `positive` marks the class of interest
    (default "no" = toxic detected, since toxicity uses yes=non-toxic)."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": agreement_rate(y_true, y_pred) or 0.0}


# --------------------------------------------------------------------------- #
# Grouping helpers
# --------------------------------------------------------------------------- #

def group_pairs(pairs: Sequence[Tuple[Any, Any]],
                keys: Sequence[Any]) -> Dict[Any, Tuple[list, list]]:
    """Split (pred, gold) pairs into groups by `keys`."""
    groups: Dict[Any, Tuple[list, list]] = defaultdict(lambda: ([], []))
    for (p, g), k in zip(pairs, keys):
        groups[k][0].append(p)
        groups[k][1].append(g)
    return dict(groups)


def summarize_correlations(per_group: Dict[str, Optional[float]],
                           group_sizes: Dict[str, int]) -> Dict[str, Any]:
    """Mean (macro) and size-weighted (micro) rho over groups."""
    valid = {k: v for k, v in per_group.items() if v is not None}
    if not valid:
        return {"macro": None, "micro": None, "n_groups": 0}
    macro = sum(valid.values()) / len(valid)
    total_n = sum(group_sizes.get(k, 0) for k in valid)
    micro = (sum(valid[k] * group_sizes.get(k, 0) for k in valid) / total_n
             if total_n else macro)
    return {"macro": round(macro, 4), "micro": round(micro, 4),
            "n_groups": len(valid)}
