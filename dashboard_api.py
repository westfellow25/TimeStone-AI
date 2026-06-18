"""TimeStone dashboard API — Monte Carlo distribution endpoint.

A thin, public (no-auth) FastAPI wrapper used by the unified web dashboard.
It reuses the real engine (services/monte_carlo.py + services/sensitivity.py)
and returns the full NPV distribution (percentiles, histogram, tornado) that the
main /assess endpoint does not expose.

Run:
    .venv/Scripts/python.exe -m uvicorn dashboard_api:app --port 8088
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make the src/ package importable without an editable install.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from timestone.domain.simulation import SimulationConfig  # noqa: E402
from timestone.services.monte_carlo import MonteCarloSimulator  # noqa: E402
from timestone.services.sensitivity import sensitivity_analysis  # noqa: E402

app = FastAPI(title="TimeStone Dashboard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_DRIVER_LABELS = {
    "revenue_increase": "Revenue uplift",
    "cost_reduction": "Cost reduction",
    "investment_required": "Capex / investment",
    "implementation_time_months": "Implementation time",
    "discount_rate": "Discount rate",
}


class SimRequest(BaseModel):
    caseName: str = "ERP migration"
    investment: float = 40_000_000
    revenueIncrease: float = 0.06
    costReduction: float = 0.06
    implTimeMonths: int = 12
    riskLevel: str = "high"
    baselineRevenue: float = 500_000_000
    baselineOperatingCosts: float = 300_000_000
    iterations: int = 25_000


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "timestone-dashboard"}


@app.post("/simulate")
def simulate(req: SimRequest) -> dict:
    scenario = {
        "id": 1,
        "name": req.caseName,
        "expected_impact": {
            "revenue_increase": req.revenueIncrease,
            "cost_reduction": req.costReduction,
        },
        "investment_required": req.investment,
        "implementation_time_months": req.implTimeMonths,
        "risk_level": req.riskLevel,
        "empirical_prior": {},
    }

    sim = MonteCarloSimulator(
        SimulationConfig(iterations=req.iterations), random_seed=42
    )
    result = sim.simulate_scenario(
        scenario, req.baselineRevenue, req.baselineOperatingCosts
    )
    samples = sim.last_npv_samples

    p5, p10, p50, p90 = (float(np.percentile(samples, q)) for q in (5, 10, 50, 90))

    counts, edges = np.histogram(samples, bins=16)
    histogram = [
        {
            "npvM": round((edges[i] + edges[i + 1]) / 2 / 1e6, 1),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]

    rows = sensitivity_analysis(
        scenario, req.baselineRevenue, req.baselineOperatingCosts, iterations=400
    )
    tornado = [
        {
            "driver": _DRIVER_LABELS.get(r.parameter, r.parameter),
            "low": round((r.low_npv - r.baseline_npv) / 1e6, 1),
            "high": round((r.high_npv - r.baseline_npv) / 1e6, 1),
        }
        for r in rows
    ]

    return {
        "caseName": req.caseName,
        "investment": req.investment,
        "paths": req.iterations,
        "probPositive": result.success_probability,
        "meanNpv": result.mean_npv,
        "p5": p5,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "histogram": histogram,
        "tornado": tornado,
    }
