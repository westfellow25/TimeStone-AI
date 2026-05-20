"""Bayesian-style calibration loop.

The product gets smarter over time only if we learn from real outcomes.
This module reads every record in `results/outcomes/` and produces
calibration adjustments that the scenario generator applies to its
empirical priors at simulation time.

v0 design (deliberately simple):
  - Group outcomes by (industry, transformation_type).
  - For each group with >= 2 outcomes, compute the mean residual:
        residual = actual_revenue_uplift_pct - predicted_revenue_uplift_pct
    Persist as a "shift" to apply to that group's empirical mean
    next time we build a prior for it.
  - Also persist a global residual for groups with no specific data.

v1 (later): proper Bayesian update with a Beta-distribution prior on
success probability and Normal-Inverse-Gamma on uplifts. For now the
mean-shift approach is enough to demonstrate the learning loop.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..domain.outcome import OutcomeRecord
from ..infrastructure.paths import RESULTS_DIR


CALIBRATION_PATH = RESULTS_DIR / "calibration.json"


@dataclass
class CalibrationEntry:
    """One calibration adjustment for a (industry, transformation_type) bucket."""
    industry: str
    transformation_type: str
    n_outcomes: int
    revenue_uplift_residual: float       # mean(actual - predicted)
    cost_reduction_residual: float
    success_rate_residual: float         # observed success rate - mean predicted P(NPV>0)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict) -> "CalibrationEntry":
        return cls(**d)


@dataclass
class CalibrationTable:
    """Look-up table of calibration adjustments."""
    by_bucket: Dict[Tuple[str, str], CalibrationEntry]
    global_residual: CalibrationEntry

    def shift_for(self, industry: str, transformation_type: str) -> CalibrationEntry:
        entry = self.by_bucket.get((industry, transformation_type))
        if entry is not None:
            return entry
        return self.global_residual

    def to_dict(self) -> Dict:
        return {
            "by_bucket": [e.to_dict() for e in self.by_bucket.values()],
            "global": self.global_residual.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CalibrationTable":
        by_bucket = {}
        for e in d.get("by_bucket", []):
            entry = CalibrationEntry.from_dict(e)
            by_bucket[(entry.industry, entry.transformation_type)] = entry
        global_residual = CalibrationEntry.from_dict(d.get("global", {
            "industry": "*", "transformation_type": "*", "n_outcomes": 0,
            "revenue_uplift_residual": 0.0, "cost_reduction_residual": 0.0,
            "success_rate_residual": 0.0,
        }))
        return cls(by_bucket=by_bucket, global_residual=global_residual)

    @classmethod
    def empty(cls) -> "CalibrationTable":
        return cls(by_bucket={}, global_residual=CalibrationEntry(
            industry="*", transformation_type="*", n_outcomes=0,
            revenue_uplift_residual=0.0, cost_reduction_residual=0.0,
            success_rate_residual=0.0))


def compute_calibration(outcomes: List[OutcomeRecord]) -> CalibrationTable:
    """Compute calibration shifts from a list of recorded outcomes."""
    if not outcomes:
        return CalibrationTable.empty()

    # Group by (industry, type) - we infer industry/type from scenario name
    # as a crude v0. In v1 OutcomeRecord should carry these explicitly.
    grouped: Dict[Tuple[str, str], List[OutcomeRecord]] = defaultdict(list)
    for o in outcomes:
        # Bucketing fields are not yet on OutcomeRecord; we use placeholders
        # so this code is forward-compatible.
        industry = getattr(o, "industry", "*")
        ttype = getattr(o, "transformation_type", "*")
        grouped[(industry, ttype)].append(o)

    by_bucket = {}
    all_rev_residuals: List[float] = []
    all_cost_residuals: List[float] = []
    all_success_residuals: List[float] = []

    for (industry, ttype), recs in grouped.items():
        rev = [(o.actual_revenue_uplift_pct - o.predicted_revenue_uplift_pct)
               for o in recs if o.actual_revenue_uplift_pct is not None]
        cost = [(o.actual_cost_reduction_pct - o.predicted_cost_reduction_pct)
                for o in recs if o.actual_cost_reduction_pct is not None]
        succ = []
        for o in recs:
            if o.actual_status in ("success", "failed", "partial"):
                actual = 1.0 if o.actual_status == "success" else (
                    0.5 if o.actual_status == "partial" else 0.0)
                succ.append(actual - o.predicted_success_probability)
        all_rev_residuals.extend(rev)
        all_cost_residuals.extend(cost)
        all_success_residuals.extend(succ)
        if len(recs) >= 2:
            by_bucket[(industry, ttype)] = CalibrationEntry(
                industry=industry, transformation_type=ttype,
                n_outcomes=len(recs),
                revenue_uplift_residual=(sum(rev) / len(rev)) if rev else 0.0,
                cost_reduction_residual=(sum(cost) / len(cost)) if cost else 0.0,
                success_rate_residual=(sum(succ) / len(succ)) if succ else 0.0,
            )

    global_entry = CalibrationEntry(
        industry="*", transformation_type="*", n_outcomes=len(outcomes),
        revenue_uplift_residual=(sum(all_rev_residuals) / len(all_rev_residuals)) if all_rev_residuals else 0.0,
        cost_reduction_residual=(sum(all_cost_residuals) / len(all_cost_residuals)) if all_cost_residuals else 0.0,
        success_rate_residual=(sum(all_success_residuals) / len(all_success_residuals)) if all_success_residuals else 0.0,
    )
    return CalibrationTable(by_bucket=by_bucket, global_residual=global_entry)


def save_calibration(table: CalibrationTable, path: Optional[Path] = None) -> Path:
    p = path or CALIBRATION_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(table.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load_calibration(path: Optional[Path] = None) -> CalibrationTable:
    p = path or CALIBRATION_PATH
    if not p.exists():
        return CalibrationTable.empty()
    return CalibrationTable.from_dict(json.loads(p.read_text(encoding="utf-8")))


def apply_calibration_to_prior(prior: Dict, calibration: CalibrationEntry,
                                parameter: str) -> Dict:
    """Mutate a copy of the empirical prior by shifting its mean using
    the calibration entry's residual for that parameter."""
    if parameter == "actual_revenue_uplift_pct":
        shift = calibration.revenue_uplift_residual
    elif parameter == "actual_cost_reduction_pct":
        shift = calibration.cost_reduction_residual
    else:
        shift = 0.0

    # Damp the shift by sample-size confidence: shrink toward 0 at low n
    damp = min(calibration.n_outcomes / 5.0, 1.0) if calibration.n_outcomes else 0.0
    effective = shift * damp

    out = dict(prior)
    out["mean"] = prior["mean"] + effective
    out["calibration_shift"] = effective
    out["calibration_n"] = calibration.n_outcomes
    return out
