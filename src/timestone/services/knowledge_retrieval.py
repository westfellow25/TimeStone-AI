"""Retrieval and empirical-prior services over a list of TransformationCases.

Pure logic - accepts cases in memory, returns priors. The Repository
loads cases from disk; this service does not touch disk.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from ..domain.case import CaseQuery, TransformationCase


SIZE_BUCKETS = {
    "small": (0, 100_000_000),
    "mid": (100_000_000, 5_000_000_000),
    "large": (5_000_000_000, 50_000_000_000),
    "mega": (50_000_000_000, float("inf")),
}

BUCKET_ORDER = ["small", "mid", "large", "mega"]

DEFAULT_WEIGHTS = {
    "industry_exact": 3.0,
    "industry_tag_overlap": 1.0,
    "size_bucket_exact": 2.0,
    "size_bucket_adjacent": 1.0,
    "transformation_type_exact": 3.0,
    "geography_exact": 0.5,
    "prior_failure_mode_overlap": 1.5,
    "segment_keyword_overlap": 0.5,
}


def revenue_to_bucket(revenue_usd: float) -> str:
    for name, (lo, hi) in SIZE_BUCKETS.items():
        if lo <= revenue_usd < hi:
            return name
    return "mega"


def _are_adjacent(a: str, b: str) -> bool:
    try:
        ia, ib = BUCKET_ORDER.index(a), BUCKET_ORDER.index(b)
        return abs(ia - ib) == 1
    except ValueError:
        return False


def _percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


class CaseLibrary:
    """In-memory retrieval over a list of TransformationCases."""

    def __init__(self, cases: List[TransformationCase]):
        self.cases = cases

    def __len__(self) -> int:
        return len(self.cases)

    def score(self, case: TransformationCase, query: CaseQuery,
              weights: Optional[Dict[str, float]] = None) -> float:
        w = weights or DEFAULT_WEIGHTS
        s = 0.0
        if query.industry and case.industry == query.industry:
            s += w["industry_exact"]
        if query.industry_tags:
            overlap = len(set(case.industry_tags) & set(query.industry_tags))
            s += overlap * w["industry_tag_overlap"]
        if query.revenue_usd is not None:
            q_bucket = revenue_to_bucket(query.revenue_usd)
            if case.revenue_bucket == q_bucket:
                s += w["size_bucket_exact"]
            elif _are_adjacent(case.revenue_bucket, q_bucket):
                s += w["size_bucket_adjacent"]
        if query.transformation_type and case.transformation_type == query.transformation_type:
            s += w["transformation_type_exact"]
        if query.geography and case.geography == query.geography:
            s += w["geography_exact"]
        # Prior failure-mode overlap: target has been burned by X before;
        # other cases that ALSO failed via X carry tacit warning signal.
        if query.prior_failure_modes and case.failure_modes:
            overlap = len(set(query.prior_failure_modes) & set(case.failure_modes))
            s += overlap * w["prior_failure_mode_overlap"]
        # Segment keyword overlap on case description / subtype.
        if query.segment_keywords:
            haystack = (case.description + " " + case.transformation_subtype).lower()
            overlap = sum(1 for kw in query.segment_keywords
                          if kw and kw.lower() in haystack)
            s += overlap * w["segment_keyword_overlap"]
        return s

    def find_similar(self, query: CaseQuery, k: int = 5,
                     min_score: float = 1.0) -> List[Tuple[TransformationCase, float]]:
        scored = [(c, self.score(c, query)) for c in self.cases]
        scored = [(c, s) for c, s in scored if s >= min_score]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def empirical_prior(self, retrieved: List[Tuple[TransformationCase, float]],
                        parameter: str,
                        fallback: Tuple[float, float] = (0.0, 0.0)) -> Dict[str, Any]:
        values: List[float] = []
        for case, _ in retrieved:
            v = getattr(case, parameter, None)
            if v is None:
                if case.status == "failed" and parameter.startswith("actual_"):
                    values.append(0.0)
                continue
            values.append(float(v))
        n = len(values)
        if n == 0:
            mean, std = fallback
            return {"mean": mean, "std": std, "p10": mean - std, "p50": mean,
                    "p90": mean + std, "n": 0}
        values_sorted = sorted(values)
        mean = sum(values_sorted) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in values_sorted) / (n - 1)
            std = math.sqrt(var)
        else:
            std = abs(mean) * 0.3 if mean else fallback[1]
        return {
            "mean": mean, "std": std,
            "p10": _percentile(values_sorted, 10),
            "p50": _percentile(values_sorted, 50),
            "p90": _percentile(values_sorted, 90),
            "n": n,
        }

    def failure_rate(self, retrieved: List[Tuple[TransformationCase, float]]) -> float:
        if not retrieved:
            return 0.0
        failed = sum(1 for c, _ in retrieved if c.status == "failed")
        return failed / len(retrieved)
