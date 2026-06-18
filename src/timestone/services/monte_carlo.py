"""Monte Carlo Simulation Engine.

Pure service: accepts SimulationConfig + Scenario(s) + baseline metrics,
returns SimulationResult(s). No I/O.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..domain.scenario import Scenario
from ..domain.simulation import SimulationConfig, SimulationResult


class MonteCarloSimulator:
    """Monte Carlo over transformation scenarios with realistic risk modelling.

    Financial model:
      - Capex paid up-front in year 0.
      - Benefits realised after implementation period (impl_time months).
      - Adoption ramps up (default 40 / 70 / 95 / 100% by year).
      - cost_reduction applies to operating_costs (not revenue).
      - revenue_increase applies to baseline_revenue.
      - 5-year NPV at configurable discount rate.
      - Per-iteration external shocks: market downturn, competitive response,
        execution failure.
      - Per-scenario execution_failure_prob can be overridden by an
        empirical prior carried on the Scenario (built from real cases).
    """

    def __init__(self, config: Optional[SimulationConfig] = None, random_seed: int = 42):
        self.config = config or SimulationConfig()
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.results: List[SimulationResult] = []

    def simulate_scenario(self, scenario: Scenario | Dict,
                          baseline_revenue: float,
                          baseline_operating_costs: float) -> SimulationResult:
        """Run Monte Carlo for one scenario.

        Accepts either a Scenario domain object or a serialized dict
        (for backward compatibility with results loaded from JSON).
        """
        if isinstance(scenario, Scenario):
            sid = scenario.id
            sname = scenario.name
            ei = scenario.expected_impact
            investment = scenario.investment_required
            impl_time_months = scenario.implementation_time_months
            risk_level = scenario.risk_level
            empirical_prior = scenario.empirical_prior or {}
        else:
            sid = scenario["id"]
            sname = scenario["name"]
            ei = scenario["expected_impact"]
            investment = scenario["investment_required"]
            impl_time_months = scenario["implementation_time_months"]
            risk_level = scenario.get("risk_level", "medium")
            empirical_prior = scenario.get("empirical_prior") or {}

        cfg = self.config
        rng = self.rng
        rev_impact_mean = ei["revenue_increase"]
        cost_impact_mean = ei["cost_reduction"]

        # Per-scenario execution_failure_prob: prefer empirical prior built
        # from >=3 similar cases. Clip to [2%, 60%] to avoid one outlier
        # case-set producing an absurd rate.
        case_failure_rate = empirical_prior.get("failure_rate")
        case_n = (empirical_prior.get("revenue_uplift") or {}).get("n", 0)
        if case_failure_rate is not None and case_n >= 3:
            execution_failure_prob = max(0.02, min(0.60, float(case_failure_rate)))
        else:
            execution_failure_prob = cfg.execution_failure_prob

        npv_samples = np.empty(cfg.iterations)
        roi_samples = np.empty(cfg.iterations)
        payback_samples = np.empty(cfg.iterations)

        for i in range(cfg.iterations):
            rev_impact = rng.normal(
                rev_impact_mean,
                abs(rev_impact_mean) * cfg.revenue_variance[risk_level])
            cost_impact = rng.normal(
                cost_impact_mean,
                abs(cost_impact_mean) * cfg.cost_variance[risk_level])
            delay = 1.0 + rng.uniform(0.0, cfg.delay_factor[risk_level])
            impl_time_actual = impl_time_months * delay
            overrun = 1.0 + rng.uniform(0.0, cfg.cost_overrun[risk_level])
            actual_investment = investment * overrun

            if rng.random() < execution_failure_prob:
                npv = self._compute_npv(-actual_investment, np.zeros(cfg.horizon_years),
                                        cfg.discount_rate)
                npv_samples[i] = npv
                roi_samples[i] = -1.0
                payback_samples[i] = cfg.horizon_years + 1
                continue
            if rng.random() < cfg.market_downturn_prob:
                rev_impact += cfg.market_downturn_impact * abs(rev_impact_mean)
            if rng.random() < cfg.competitive_response_prob:
                rev_impact += cfg.competitive_response_impact * abs(rev_impact_mean)

            year_zero_cost = -actual_investment
            start_year = impl_time_actual / 12.0
            annual_benefits = np.zeros(cfg.horizon_years)
            for y in range(cfg.horizon_years):
                active_fraction = max(0.0, min(1.0, (y + 1) - start_year))
                if active_fraction <= 0:
                    continue
                years_since_impl = max(0, y - int(start_year))
                adoption_idx = min(years_since_impl, len(cfg.adoption_curve) - 1)
                adoption = cfg.adoption_curve[adoption_idx]
                gross_rev = baseline_revenue * rev_impact
                gross_cost = baseline_operating_costs * cost_impact
                annual_benefits[y] = (gross_rev + gross_cost) * adoption * active_fraction

            npv = self._compute_npv(year_zero_cost, annual_benefits, cfg.discount_rate)
            npv_samples[i] = npv
            roi_samples[i] = npv / actual_investment if actual_investment > 0 else 0.0

            cum = -actual_investment
            payback = cfg.horizon_years + 1
            for y in range(cfg.horizon_years):
                cum += annual_benefits[y]
                if cum >= 0:
                    payback = y + 1
                    break
            payback_samples[i] = payback

        # Expose the raw NPV draws so callers can build a distribution
        # (percentiles, histogram). Additive: does not change the result.
        self.last_npv_samples = npv_samples

        return SimulationResult(
            scenario_id=sid, scenario_name=sname,
            mean_npv=float(np.mean(npv_samples)),
            median_npv=float(np.median(npv_samples)),
            mean_roi=float(np.mean(roi_samples)),
            median_roi=float(np.median(roi_samples)),
            std_dev_roi=float(np.std(roi_samples)),
            confidence_90_lower=float(np.percentile(roi_samples, 5)),
            confidence_90_upper=float(np.percentile(roi_samples, 95)),
            success_probability=float(np.mean(npv_samples > 0)),
            high_success_probability=float(np.mean(roi_samples > 0.20)),
            risk_score=float(1.0 - np.mean(npv_samples > 0)),
            payback_years_median=float(np.median(payback_samples)),
            iterations=cfg.iterations,
        )

    @staticmethod
    def _compute_npv(year_zero_cost: float, annual_benefits, discount_rate: float) -> float:
        npv = year_zero_cost
        for y, benefit in enumerate(annual_benefits, start=1):
            npv += benefit / ((1 + discount_rate) ** y)
        return npv

    def simulate_all(self, scenarios: List, baseline_revenue: float,
                     baseline_operating_costs: float) -> List[SimulationResult]:
        self.results = []
        for s in scenarios:
            self.results.append(
                self.simulate_scenario(s, baseline_revenue, baseline_operating_costs))
        return self.results
