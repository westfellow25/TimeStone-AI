"""
Monte Carlo Simulation Engine v2.

Runs probabilistic simulations to predict transformation outcomes
with confidence intervals and risk analysis.

Financial model:
    - Capex paid up-front in year 0
    - Benefits realized after implementation period (impl_time months)
    - Adoption ramps up (default 40% / 70% / 95% / 100% by year)
    - cost_reduction applies to operating_costs (NOT revenue)
    - revenue_increase applies to baseline_revenue
    - 5-year NPV at configurable discount rate
    - External shocks: market downturn, competitive response, execution failure
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json


@dataclass
class SimulationConfig:
    """Configurable financial assumptions for Monte Carlo simulation"""
    iterations: int = 1000
    horizon_years: int = 5
    discount_rate: float = 0.12          # 12% WACC
    adoption_curve: tuple = (0.40, 0.70, 0.95, 1.0, 1.0)  # by year, post-implementation

    # Variance multipliers per risk level
    revenue_variance: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.15, "medium": 0.25, "high": 0.40
    })
    cost_variance: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.10, "medium": 0.20, "high": 0.35
    })
    cost_overrun: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.10, "medium": 0.25, "high": 0.50
    })
    delay_factor: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.20, "medium": 0.40, "high": 0.80
    })

    # External shocks (probability per simulation)
    market_downturn_prob: float = 0.08
    market_downturn_impact: float = -0.30   # -30% to revenue impact
    competitive_response_prob: float = 0.15
    competitive_response_impact: float = -0.20
    execution_failure_prob: float = 0.05    # project effectively dies


@dataclass
class SimulationResult:
    """Result of Monte Carlo simulation for a scenario"""
    scenario_id: int
    scenario_name: str
    mean_npv: float
    median_npv: float
    mean_roi: float
    median_roi: float
    std_dev_roi: float
    confidence_90_lower: float
    confidence_90_upper: float
    success_probability: float           # P(NPV > 0)
    high_success_probability: float      # P(ROI > 20%)
    risk_score: float                    # 1 - success_probability
    payback_years_median: float
    iterations: int

    def __repr__(self):
        return (f"Scenario #{self.scenario_id}: {self.scenario_name}\n"
                f"  Mean ROI: {self.mean_roi:.1%}\n"
                f"  Mean NPV: ${self.mean_npv:,.0f}\n"
                f"  P(NPV>0): {self.success_probability:.1%}\n"
                f"  P(ROI>20%): {self.high_success_probability:.1%}\n"
                f"  90% CI ROI: [{self.confidence_90_lower:.1%}, "
                f"{self.confidence_90_upper:.1%}]\n"
                f"  Median payback: {self.payback_years_median:.1f}y")


class MonteCarloSimulator:
    """
    Monte Carlo simulation for transformation scenarios.

    Models realistic risk: external shocks, execution failure, delays,
    cost overruns, adoption ramp-up. Uses NPV over a multi-year horizon.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        random_seed: int = 42,
    ):
        self.config = config or SimulationConfig()
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.results: List[SimulationResult] = []

    # ------------------------------------------------------------------
    # Core single-scenario simulation
    # ------------------------------------------------------------------
    def simulate_scenario(
        self,
        scenario: Dict,
        baseline_revenue: float,
        baseline_operating_costs: float,
    ) -> SimulationResult:
        """
        Run Monte Carlo for one scenario.

        Args:
            scenario: dict with keys id, name, expected_impact, investment_required,
                      implementation_time_months, risk_level
            baseline_revenue: company revenue
            baseline_operating_costs: company operating costs (cost_reduction applies here)
        """
        cfg = self.config
        rng = self.rng

        rev_impact_mean = scenario["expected_impact"]["revenue_increase"]
        cost_impact_mean = scenario["expected_impact"]["cost_reduction"]
        investment = scenario["investment_required"]
        impl_time_months = scenario["implementation_time_months"]
        risk_level = scenario.get("risk_level", "medium")

        # If the scenario carries an empirical prior from the Case Library,
        # use the observed failure rate of similar cases as a per-scenario
        # execution_failure_prob. We clip to [2%, 60%] to avoid one outlier
        # case-set producing an absurd rate, and we only override when the
        # prior was built from at least 3 cases.
        empirical_prior = scenario.get("empirical_prior") or {}
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
            # ----- 1. Sample uncertain inputs -----
            rev_impact = rng.normal(
                rev_impact_mean,
                abs(rev_impact_mean) * cfg.revenue_variance[risk_level]
            )
            cost_impact = rng.normal(
                cost_impact_mean,
                abs(cost_impact_mean) * cfg.cost_variance[risk_level]
            )

            # Delay drawn from uniform: [1.0, 1.0 + delay_factor]
            delay = 1.0 + rng.uniform(0.0, cfg.delay_factor[risk_level])
            impl_time_actual = impl_time_months * delay

            # Cost overrun
            overrun = 1.0 + rng.uniform(0.0, cfg.cost_overrun[risk_level])
            actual_investment = investment * overrun

            # ----- 2. Apply external shocks -----
            if rng.random() < execution_failure_prob:
                # Project fails: investment lost, no benefits
                annual_benefits = np.zeros(cfg.horizon_years)
                year_zero_cost = -actual_investment
                npv = self._compute_npv(year_zero_cost, annual_benefits, cfg.discount_rate)
                npv_samples[i] = npv
                roi_samples[i] = -1.0
                payback_samples[i] = cfg.horizon_years + 1  # never
                continue

            if rng.random() < cfg.market_downturn_prob:
                rev_impact += cfg.market_downturn_impact * abs(rev_impact_mean)

            if rng.random() < cfg.competitive_response_prob:
                rev_impact += cfg.competitive_response_impact * abs(rev_impact_mean)

            # ----- 3. Build annual cash flows -----
            year_zero_cost = -actual_investment

            # Benefits start after impl_time_actual months
            start_year = impl_time_actual / 12.0  # fractional year when benefits begin
            annual_benefits = np.zeros(cfg.horizon_years)

            for y in range(cfg.horizon_years):
                # How much of year y is post-implementation?
                year_start = y
                year_end = y + 1
                active_fraction = max(0.0, min(1.0, year_end - start_year))
                if active_fraction <= 0:
                    continue

                # Adoption ramp index: how many years since implementation completed
                years_since_impl = max(0, y - int(start_year))
                adoption_idx = min(years_since_impl, len(cfg.adoption_curve) - 1)
                adoption = cfg.adoption_curve[adoption_idx]

                gross_revenue_benefit = baseline_revenue * rev_impact
                gross_cost_benefit = baseline_operating_costs * cost_impact
                annual_benefit = (gross_revenue_benefit + gross_cost_benefit) * adoption * active_fraction
                annual_benefits[y] = annual_benefit

            npv = self._compute_npv(year_zero_cost, annual_benefits, cfg.discount_rate)
            npv_samples[i] = npv

            # ROI = NPV / investment (cleaner than raw multipliers)
            roi_samples[i] = npv / actual_investment if actual_investment > 0 else 0.0

            # Payback: first year where cumulative undiscounted cash flow turns positive
            cum = -actual_investment
            payback = cfg.horizon_years + 1
            for y in range(cfg.horizon_years):
                cum += annual_benefits[y]
                if cum >= 0:
                    payback = y + 1
                    break
            payback_samples[i] = payback

        # ----- 4. Aggregate statistics -----
        return SimulationResult(
            scenario_id=scenario["id"],
            scenario_name=scenario["name"],
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
    def _compute_npv(year_zero_cost: float, annual_benefits: np.ndarray, discount_rate: float) -> float:
        """NPV with year-0 cost outflow and benefits in years 1..N"""
        npv = year_zero_cost
        for y, benefit in enumerate(annual_benefits, start=1):
            npv += benefit / ((1 + discount_rate) ** y)
        return npv

    # ------------------------------------------------------------------
    # Multi-scenario simulation
    # ------------------------------------------------------------------
    def simulate_all_scenarios(
        self,
        scenarios: List[Dict],
        baseline_revenue: float,
        baseline_operating_costs: float,
    ) -> List[SimulationResult]:
        print(f"Running Monte Carlo: {self.config.iterations} iterations x {len(scenarios)} scenarios")
        print(f"Horizon: {self.config.horizon_years}y | Discount: {self.config.discount_rate:.0%}")
        print("=" * 70)

        self.results = []
        for i, scenario in enumerate(scenarios, 1):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(scenarios)} scenarios completed...")
            result = self.simulate_scenario(scenario, baseline_revenue, baseline_operating_costs)
            self.results.append(result)

        print("=" * 70)
        print(f"Simulation complete: {len(self.results)} scenarios analyzed.")
        return self.results

    def get_top_scenarios(
        self,
        n: int = 3,
        sort_by: str = "success_probability",
    ) -> List[SimulationResult]:
        if not self.results:
            return []
        return sorted(self.results, key=lambda x: getattr(x, sort_by), reverse=True)[:n]

    def save_results(self, filepath: str):
        data = {
            "simulation_parameters": {
                "iterations": self.config.iterations,
                "horizon_years": self.config.horizon_years,
                "discount_rate": self.config.discount_rate,
                "random_seed": self.random_seed,
                "total_scenarios": len(self.results),
            },
            "results": [
                {
                    "scenario_id": r.scenario_id,
                    "scenario_name": r.scenario_name,
                    "mean_npv": r.mean_npv,
                    "median_npv": r.median_npv,
                    "mean_roi": r.mean_roi,
                    "median_roi": r.median_roi,
                    "std_dev_roi": r.std_dev_roi,
                    "confidence_90_lower": r.confidence_90_lower,
                    "confidence_90_upper": r.confidence_90_upper,
                    "success_probability": r.success_probability,
                    "high_success_probability": r.high_success_probability,
                    "risk_score": r.risk_score,
                    "payback_years_median": r.payback_years_median,
                }
                for r in self.results
            ],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI / example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading scenarios and digital twin...")
    with open("ktz_scenarios.json", "r") as f:
        scen_data = json.load(f)
    with open("ktz_digital_twin.json", "r") as f:
        twin_data = json.load(f)

    scenarios = scen_data["scenarios"]
    baseline_revenue = twin_data["metrics"]["revenue"]
    baseline_op_costs = twin_data["metrics"]["operating_costs"]

    print(f"Company: {scen_data['company']}")
    print(f"Baseline revenue: ${baseline_revenue:,.0f}")
    print(f"Baseline op costs: ${baseline_op_costs:,.0f}")
    print(f"Scenarios: {len(scenarios)}")
    print()

    simulator = MonteCarloSimulator(config=SimulationConfig(iterations=1000), random_seed=42)
    results = simulator.simulate_all_scenarios(scenarios, baseline_revenue, baseline_op_costs)

    print("\n" + "=" * 70)
    print("TOP-3 SCENARIOS BY P(NPV > 0)")
    print("=" * 70)
    for i, r in enumerate(simulator.get_top_scenarios(n=3), 1):
        print(f"\nRANK #{i}")
        print("-" * 70)
        print(r)

    simulator.save_results("ktz_simulation_results.json")
    print(f"\nResults saved to ktz_simulation_results.json")
