"""
Case Library: empirical priors for Monte Carlo simulation.

Loads a JSON corpus of real corporate transformation cases (with both promised
and actual outcomes where known) and exposes retrieval + empirical-prior
helpers that the scenario generator and Monte Carlo simulator can use instead
of hard-coded rule-based ranges.

Design choices:
- Pure-Python retrieval scoring (industry / size / type / geography overlap).
  No ML dependency: deterministic, debuggable, fast.
- Optional semantic boost via sentence-transformers if installed (off by default).
- Empirical priors based on `actual_*` fields, not `promised_*`, to avoid
  perpetuating management optimism bias.
- Cases with `status="failed"` are kept in the prior pool: failures are signal,
  not noise.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Size buckets (revenue, USD) - used for similarity scoring
# ---------------------------------------------------------------------------
SIZE_BUCKETS = {
    "small": (0, 100_000_000),
    "mid": (100_000_000, 5_000_000_000),
    "large": (5_000_000_000, 50_000_000_000),
    "mega": (50_000_000_000, float("inf")),
}


def revenue_to_bucket(revenue_usd: float) -> str:
    for name, (lo, hi) in SIZE_BUCKETS.items():
        if lo <= revenue_usd < hi:
            return name
    return "mega"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Case:
    """One transformation case from the corpus."""
    id: str
    company: str
    industry: str
    industry_tags: List[str]
    geography: str
    revenue_usd: float
    employees: int
    revenue_bucket: str
    transformation_type: str
    transformation_subtype: str
    description: str
    start_year: int
    planned_duration_months: int
    actual_duration_months: int
    status: str  # success / partial / failed / cancelled
    vendor: str
    # financials
    promised_investment_usd: Optional[float]
    actual_investment_usd: Optional[float]
    writeoff_usd: Optional[float]
    promised_revenue_uplift_pct: Optional[float]
    actual_revenue_uplift_pct: Optional[float]
    promised_cost_reduction_pct: Optional[float]
    actual_cost_reduction_pct: Optional[float]
    # qualitative
    failure_modes: List[str]
    success_factors: List[str]
    sources: List[Dict[str, Any]]
    tacit_notes: str

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Case":
        size = data["company_size"]
        t = data["transformation"]
        fin = data["financials"]
        return cls(
            id=data["id"],
            company=data["company"],
            industry=data["industry"],
            industry_tags=data.get("industry_tags", []),
            geography=data["geography"],
            revenue_usd=float(size["revenue_usd"]),
            employees=int(size["employees"]),
            revenue_bucket=size.get("revenue_bucket", revenue_to_bucket(size["revenue_usd"])),
            transformation_type=t["type"],
            transformation_subtype=t.get("subtype", ""),
            description=t.get("description", ""),
            start_year=int(t.get("start_year", 0)),
            planned_duration_months=int(t.get("planned_duration_months", 0) or 0),
            actual_duration_months=int(t.get("actual_duration_months", 0) or 0),
            status=t.get("status", "unknown"),
            vendor=t.get("vendor", ""),
            promised_investment_usd=fin.get("promised_investment_usd"),
            actual_investment_usd=fin.get("actual_investment_usd"),
            writeoff_usd=fin.get("writeoff_usd"),
            promised_revenue_uplift_pct=fin.get("promised_revenue_uplift_pct"),
            actual_revenue_uplift_pct=fin.get("actual_revenue_uplift_pct"),
            promised_cost_reduction_pct=fin.get("promised_cost_reduction_pct"),
            actual_cost_reduction_pct=fin.get("actual_cost_reduction_pct"),
            failure_modes=data.get("failure_modes", []),
            success_factors=data.get("success_factors", []),
            sources=data.get("sources", []),
            tacit_notes=data.get("tacit_notes", ""),
        )


@dataclass
class Query:
    """Lookup criteria for retrieving similar cases."""
    industry: Optional[str] = None
    industry_tags: List[str] = field(default_factory=list)
    revenue_usd: Optional[float] = None
    transformation_type: Optional[str] = None
    geography: Optional[str] = None


# ---------------------------------------------------------------------------
# Default scoring weights - tune as the corpus grows
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "industry_exact": 3.0,
    "industry_tag_overlap": 1.0,    # multiplied by overlap count
    "size_bucket_exact": 2.0,
    "size_bucket_adjacent": 1.0,
    "transformation_type_exact": 3.0,
    "geography_exact": 0.5,
}

BUCKET_ORDER = ["small", "mid", "large", "mega"]


# ---------------------------------------------------------------------------
# CaseLibrary
# ---------------------------------------------------------------------------
class CaseLibrary:
    """Loads case corpus and exposes retrieval + empirical prior helpers."""

    def __init__(self, cases: List[Case]):
        self.cases = cases

    @classmethod
    def from_json_file(cls, path: str) -> "CaseLibrary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cases = [Case.from_json(c) for c in data["cases"]]
        return cls(cases)

    def __len__(self) -> int:
        return len(self.cases)

    # ----------------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------------
    def score(self, case: Case, query: Query, weights: Dict[str, float] = None) -> float:
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

        return s

    def find_similar(
        self,
        query: Query,
        k: int = 5,
        min_score: float = 1.0,
    ) -> List[Tuple[Case, float]]:
        """Return up to k cases ranked by similarity to the query."""
        scored = [(c, self.score(c, query)) for c in self.cases]
        scored = [(c, s) for c, s in scored if s >= min_score]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ----------------------------------------------------------------
    # Empirical priors
    # ----------------------------------------------------------------
    def empirical_prior(
        self,
        retrieved: List[Tuple[Case, float]],
        parameter: str,
        fallback: Tuple[float, float] = (0.0, 0.0),
    ) -> Dict[str, float]:
        """
        Build an empirical prior from retrieved cases for a given parameter.

        Returns a dict with mean, std, p10, p50, p90, n (sample size).
        Treats explicit 0.0 as a real datapoint; treats None as missing.
        Failed projects with no measured uplift are coerced to 0.0 (they
        still inform the distribution by pulling its left tail).
        """
        values: List[float] = []
        for case, _score in retrieved:
            v = getattr(case, parameter, None)
            if v is None:
                # If case failed and parameter missing, treat as 0 (no benefit)
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
            "mean": mean,
            "std": std,
            "p10": _percentile(values_sorted, 10),
            "p50": _percentile(values_sorted, 50),
            "p90": _percentile(values_sorted, 90),
            "n": n,
        }

    def failure_rate(self, retrieved: List[Tuple[Case, float]]) -> float:
        """Fraction of retrieved cases with status='failed'. Useful as a
        base rate for execution_failure_prob in Monte Carlo."""
        if not retrieved:
            return 0.0
        failed = sum(1 for c, _ in retrieved if c.status == "failed")
        return failed / len(retrieved)

    def cost_overrun_factor(self, retrieved: List[Tuple[Case, float]]) -> Dict[str, float]:
        """Empirical distribution of (actual_investment / promised_investment - 1)
        across retrieved cases that have both fields."""
        overruns = []
        for case, _ in retrieved:
            p = case.promised_investment_usd
            a = case.actual_investment_usd
            if p and a and p > 0:
                overruns.append(a / p - 1.0)
        if not overruns:
            return {"mean": 0.25, "std": 0.20, "n": 0}
        mean = sum(overruns) / len(overruns)
        if len(overruns) > 1:
            var = sum((v - mean) ** 2 for v in overruns) / (len(overruns) - 1)
            std = math.sqrt(var)
        else:
            std = abs(mean) * 0.3
        return {"mean": mean, "std": std, "n": len(overruns)}

    # ----------------------------------------------------------------
    # Sampling helpers (used directly by Monte Carlo)
    # ----------------------------------------------------------------
    def sample_revenue_uplift(
        self,
        query: Query,
        k: int = 5,
        rng: Optional[random.Random] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Sample a revenue_uplift value from cases similar to the query.
        Returns (sampled_value, metadata) where metadata includes the cases used."""
        rng = rng or random.Random()
        retrieved = self.find_similar(query, k=k)
        prior = self.empirical_prior(retrieved, "actual_revenue_uplift_pct",
                                     fallback=(0.02, 0.02))
        sampled = rng.gauss(prior["mean"], prior["std"]) if prior["n"] > 0 else prior["mean"]
        return sampled, {
            "prior": prior,
            "based_on_cases": [c.id for c, _ in retrieved],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# CLI: demo retrieval
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    lib = CaseLibrary.from_json_file(str(REPO_ROOT / "data" / "case_library.json"))
    print(f"Loaded {len(lib)} cases.\n")

    query = Query(
        industry="transportation_rental",
        industry_tags=["transportation", "logistics"],
        revenue_usd=500_000_000,
        transformation_type="digital_transformation",
        geography="USA",
    )
    retrieved = lib.find_similar(query, k=5)
    print("Top similar cases:")
    for case, score in retrieved:
        print(f"  [{score:5.1f}] {case.id:30s} | {case.company} | {case.industry} | status={case.status}")

    print("\nEmpirical prior on actual_revenue_uplift_pct:")
    prior = lib.empirical_prior(retrieved, "actual_revenue_uplift_pct")
    for k, v in prior.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nEmpirical failure rate among retrieved: {lib.failure_rate(retrieved):.1%}")
    print(f"Empirical cost-overrun: {lib.cost_overrun_factor(retrieved)}")
