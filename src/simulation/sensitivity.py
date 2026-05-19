"""
Sensitivity analysis (tornado chart) for transformation scenarios.

For a given scenario, vary each input parameter ±X% around its baseline and
measure the resulting change in expected NPV. The wider the swing, the more
sensitive the scenario is to that assumption.

Usage:
    from src.simulation.sensitivity import sensitivity_analysis
    table = sensitivity_analysis(scenario, baseline_revenue, baseline_op_costs)
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List

from src.simulation.monte_carlo import MonteCarloSimulator, SimulationConfig


@dataclass
class SensitivityRow:
    """One row of a tornado chart."""
    parameter: str
    low_npv: float        # NPV when parameter set to low end
    high_npv: float       # NPV when parameter set to high end
    swing: float          # high_npv - low_npv (absolute spread)
    baseline_npv: float

    def __repr__(self):
        return (f"{self.parameter:30s} swing=${self.swing:>12,.0f} "
                f"low=${self.low_npv:>12,.0f} high=${self.high_npv:>12,.0f}")


# Default sensitivity ranges (low_multiplier, high_multiplier) applied to scenario values
DEFAULT_RANGES = {
    "revenue_increase": (0.70, 1.30),       # ±30%
    "cost_reduction": (0.70, 1.30),         # ±30%
    "investment_required": (1.30, 0.70),    # inverted: more investment is worse
    "implementation_time_months": (1.30, 0.70),  # inverted: more delay is worse
    "discount_rate": (1.50, 0.66),          # 18% vs 8% (around 12% default), inverted
}


def _run_with_overrides(
    scenario: Dict,
    baseline_revenue: float,
    baseline_op_costs: float,
    config_overrides: Dict | None = None,
    seed: int = 42,
) -> float:
    """Run a single Monte Carlo simulation and return mean_npv."""
    cfg_kwargs = {"iterations": 500}
    if config_overrides:
        cfg_kwargs.update(config_overrides)
    sim = MonteCarloSimulator(SimulationConfig(**cfg_kwargs), random_seed=seed)
    result = sim.simulate_scenario(scenario, baseline_revenue, baseline_op_costs)
    return result.mean_npv


def sensitivity_analysis(
    scenario: Dict,
    baseline_revenue: float,
    baseline_op_costs: float,
    ranges: Dict | None = None,
    iterations: int = 500,
    seed: int = 42,
) -> List[SensitivityRow]:
    """Return a list of SensitivityRow sorted by swing (largest first)."""
    ranges = ranges or DEFAULT_RANGES

    # Baseline: scenario as-is
    base = deepcopy(scenario)
    baseline_npv = _run_with_overrides(base, baseline_revenue, baseline_op_costs,
                                       {"iterations": iterations}, seed)

    rows: List[SensitivityRow] = []

    for param, (low_mult, high_mult) in ranges.items():
        if param == "discount_rate":
            base_dr = 0.12  # SimulationConfig default
            low_npv = _run_with_overrides(
                base, baseline_revenue, baseline_op_costs,
                {"iterations": iterations, "discount_rate": base_dr * low_mult}, seed
            )
            high_npv = _run_with_overrides(
                base, baseline_revenue, baseline_op_costs,
                {"iterations": iterations, "discount_rate": base_dr * high_mult}, seed
            )
        elif param in {"revenue_increase", "cost_reduction"}:
            low_scen = deepcopy(base)
            high_scen = deepcopy(base)
            low_scen["expected_impact"][param] = base["expected_impact"][param] * low_mult
            high_scen["expected_impact"][param] = base["expected_impact"][param] * high_mult
            low_npv = _run_with_overrides(low_scen, baseline_revenue, baseline_op_costs,
                                          {"iterations": iterations}, seed)
            high_npv = _run_with_overrides(high_scen, baseline_revenue, baseline_op_costs,
                                           {"iterations": iterations}, seed)
        elif param == "investment_required":
            low_scen = deepcopy(base)
            high_scen = deepcopy(base)
            low_scen["investment_required"] = base["investment_required"] * low_mult
            high_scen["investment_required"] = base["investment_required"] * high_mult
            low_npv = _run_with_overrides(low_scen, baseline_revenue, baseline_op_costs,
                                          {"iterations": iterations}, seed)
            high_npv = _run_with_overrides(high_scen, baseline_revenue, baseline_op_costs,
                                           {"iterations": iterations}, seed)
        elif param == "implementation_time_months":
            low_scen = deepcopy(base)
            high_scen = deepcopy(base)
            low_scen["implementation_time_months"] = max(1, int(base["implementation_time_months"] * low_mult))
            high_scen["implementation_time_months"] = max(1, int(base["implementation_time_months"] * high_mult))
            low_npv = _run_with_overrides(low_scen, baseline_revenue, baseline_op_costs,
                                          {"iterations": iterations}, seed)
            high_npv = _run_with_overrides(high_scen, baseline_revenue, baseline_op_costs,
                                           {"iterations": iterations}, seed)
        else:
            continue

        rows.append(SensitivityRow(
            parameter=param,
            low_npv=low_npv,
            high_npv=high_npv,
            swing=abs(high_npv - low_npv),
            baseline_npv=baseline_npv,
        ))

    rows.sort(key=lambda r: r.swing, reverse=True)
    return rows


if __name__ == "__main__":
    # Demo
    scenario = {
        "id": 1,
        "name": "Dynamic Pricing Implementation",
        "expected_impact": {"revenue_increase": 0.03, "cost_reduction": 0.005},
        "investment_required": 3_500_000.0,
        "implementation_time_months": 9,
        "risk_level": "medium",
    }
    rows = sensitivity_analysis(scenario, 500_000_000.0, 450_000_000.0, iterations=300)
    print("Sensitivity analysis (sorted by swing):")
    print("=" * 80)
    for r in rows:
        print(r)
