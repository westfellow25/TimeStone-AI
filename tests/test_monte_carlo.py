"""Tests for the Monte Carlo simulator.

These tests are the safety net that prevents the regressions
we just fixed:
    - Don't let every scenario report success_probability == 1.0
    - Don't let cost_reduction silently multiply revenue
    - Don't let ROI numbers explode when financial model breaks
"""
import math

import pytest

from src.simulation.monte_carlo import (
    MonteCarloSimulator,
    SimulationConfig,
    SimulationResult,
)


def test_simulate_scenario_returns_result_type(ktz_baseline, low_risk_scenario):
    sim = MonteCarloSimulator(SimulationConfig(iterations=200), random_seed=42)
    result = sim.simulate_scenario(
        low_risk_scenario, ktz_baseline["revenue"], ktz_baseline["operating_costs"]
    )
    assert isinstance(result, SimulationResult)
    assert result.scenario_id == 1
    assert result.iterations == 200


def test_success_probability_in_unit_interval(ktz_baseline, realistic_scenarios):
    sim = MonteCarloSimulator(SimulationConfig(iterations=500), random_seed=42)
    for s in realistic_scenarios:
        r = sim.simulate_scenario(s, ktz_baseline["revenue"], ktz_baseline["operating_costs"])
        assert 0.0 <= r.success_probability <= 1.0
        assert 0.0 <= r.high_success_probability <= 1.0


def test_high_risk_has_more_variance_than_low_risk(ktz_baseline):
    """Same scenario at different risk levels must produce wider relative CI at higher risk.
    We use coefficient of variation (std / |mean|) to normalize across scale differences."""
    scenario_low = {
        "id": 1, "name": "Test",
        "expected_impact": {"revenue_increase": 0.02, "cost_reduction": 0.02},
        "investment_required": 2_000_000.0, "implementation_time_months": 6,
        "risk_level": "low",
    }
    scenario_high = {**scenario_low, "id": 2, "risk_level": "high"}

    sim_low = MonteCarloSimulator(SimulationConfig(iterations=3000), random_seed=42)
    sim_high = MonteCarloSimulator(SimulationConfig(iterations=3000), random_seed=42)
    low = sim_low.simulate_scenario(scenario_low, ktz_baseline["revenue"], ktz_baseline["operating_costs"])
    high = sim_high.simulate_scenario(scenario_high, ktz_baseline["revenue"], ktz_baseline["operating_costs"])

    cv_low = low.std_dev_roi / abs(low.mean_roi) if low.mean_roi != 0 else 0
    cv_high = high.std_dev_roi / abs(high.mean_roi) if high.mean_roi != 0 else 0
    assert cv_high > cv_low, (
        f"High-risk coefficient of variation ({cv_high:.3f}) should exceed low-risk ({cv_low:.3f})"
    )


def test_not_every_scenario_is_perfect(ktz_baseline):
    """Regression guard: the original simulator returned success_probability==1.0
    for every scenario. Make sure realistic high-risk inputs still produce uncertainty."""
    bad_scenario = {
        "id": 99,
        "name": "Risky moonshot",
        "expected_impact": {"revenue_increase": 0.005, "cost_reduction": 0.005},
        "investment_required": 50_000_000.0,
        "implementation_time_months": 24,
        "risk_level": "high",
    }
    sim = MonteCarloSimulator(SimulationConfig(iterations=2000), random_seed=42)
    r = sim.simulate_scenario(bad_scenario, ktz_baseline["revenue"], ktz_baseline["operating_costs"])
    assert r.success_probability < 0.99, (
        f"Risky scenario should not be a sure thing, got {r.success_probability}"
    )


def test_cost_reduction_applies_to_operating_costs_not_revenue():
    """Regression guard for the core math bug: cost_reduction used to multiply
    baseline_revenue. Verify it now actually scales linearly with operating_costs.

    Approach: NPV without investment should equal sum of discounted annual benefits,
    which scales linearly with op_costs when revenue impact is zero.
    """
    scenario = {
        "id": 100,
        "name": "Pure cost play",
        "expected_impact": {"revenue_increase": 0.0, "cost_reduction": 0.05},
        "investment_required": 1.0,  # token investment — won't dominate
        "implementation_time_months": 1,
        "risk_level": "low",
    }
    cfg = SimulationConfig(
        iterations=3000,
        execution_failure_prob=0.0,
        market_downturn_prob=0.0,
        competitive_response_prob=0.0,
        # Disable variance for clean comparison
        cost_variance={"low": 0.0, "medium": 0.0, "high": 0.0},
        revenue_variance={"low": 0.0, "medium": 0.0, "high": 0.0},
        cost_overrun={"low": 0.0, "medium": 0.0, "high": 0.0},
        delay_factor={"low": 0.0, "medium": 0.0, "high": 0.0},
    )

    big = MonteCarloSimulator(cfg, random_seed=42).simulate_scenario(
        scenario, 500_000_000.0, 100_000_000.0
    )
    small = MonteCarloSimulator(cfg, random_seed=42).simulate_scenario(
        scenario, 500_000_000.0, 10_000_000.0
    )

    # With no variance and no investment, NPV should be ~10x larger
    ratio = big.mean_npv / small.mean_npv
    assert 9.5 < ratio < 10.5, (
        f"Cost reduction should scale linearly with operating_costs; got NPV ratio {ratio:.2f}"
    )


def test_npv_falls_with_higher_discount_rate(ktz_baseline, low_risk_scenario):
    """Higher discount rate → lower NPV. Sanity check on financial model."""
    cfg_low = SimulationConfig(iterations=1000, discount_rate=0.05,
                               execution_failure_prob=0.0, market_downturn_prob=0.0,
                               competitive_response_prob=0.0)
    cfg_high = SimulationConfig(iterations=1000, discount_rate=0.25,
                                execution_failure_prob=0.0, market_downturn_prob=0.0,
                                competitive_response_prob=0.0)
    low = MonteCarloSimulator(cfg_low, random_seed=42).simulate_scenario(
        low_risk_scenario, ktz_baseline["revenue"], ktz_baseline["operating_costs"]
    )
    high = MonteCarloSimulator(cfg_high, random_seed=42).simulate_scenario(
        low_risk_scenario, ktz_baseline["revenue"], ktz_baseline["operating_costs"]
    )
    assert high.mean_npv < low.mean_npv


def test_deterministic_with_same_seed(ktz_baseline, medium_risk_scenario):
    sim_a = MonteCarloSimulator(SimulationConfig(iterations=500), random_seed=42)
    sim_b = MonteCarloSimulator(SimulationConfig(iterations=500), random_seed=42)
    a = sim_a.simulate_scenario(medium_risk_scenario,
                                ktz_baseline["revenue"], ktz_baseline["operating_costs"])
    b = sim_b.simulate_scenario(medium_risk_scenario,
                                ktz_baseline["revenue"], ktz_baseline["operating_costs"])
    assert a.mean_npv == pytest.approx(b.mean_npv)
    assert a.success_probability == pytest.approx(b.success_probability)


def test_npv_computation_pure():
    """Direct unit test on the NPV helper — no randomness."""
    import numpy as np
    npv = MonteCarloSimulator._compute_npv(
        year_zero_cost=-100.0,
        annual_benefits=np.array([50.0, 50.0, 50.0]),
        discount_rate=0.10,
    )
    expected = -100.0 + 50.0 / 1.10 + 50.0 / 1.21 + 50.0 / 1.331
    assert npv == pytest.approx(expected, rel=1e-6)


def test_save_results_round_trip(ktz_baseline, realistic_scenarios, tmp_path):
    sim = MonteCarloSimulator(SimulationConfig(iterations=100), random_seed=42)
    sim.simulate_all_scenarios(
        realistic_scenarios, ktz_baseline["revenue"], ktz_baseline["operating_costs"]
    )
    out = tmp_path / "results.json"
    sim.save_results(str(out))
    import json
    data = json.loads(out.read_text())
    assert "simulation_parameters" in data
    assert "results" in data
    assert len(data["results"]) == len(realistic_scenarios)
    first = data["results"][0]
    for key in ("scenario_id", "scenario_name", "mean_npv", "mean_roi",
                "success_probability", "payback_years_median"):
        assert key in first


def test_empirical_failure_rate_overrides_default(ktz_baseline):
    """Regression guard: a scenario carrying an empirical prior with high
    failure rate must produce noticeably lower success probability than
    the same scenario without that prior."""
    base = {
        "id": 1,
        "name": "Test",
        "expected_impact": {"revenue_increase": 0.04, "cost_reduction": 0.04},
        "investment_required": 5_000_000.0,
        "implementation_time_months": 12,
        "risk_level": "medium",
    }
    risky = {
        **base,
        "empirical_prior": {
            "revenue_uplift": {"n": 10, "mean": 0.04, "std": 0.05,
                               "p10": -0.02, "p50": 0.04, "p90": 0.10},
            "failure_rate": 0.55,
        },
    }

    cfg = SimulationConfig(iterations=2000)
    no_prior = MonteCarloSimulator(cfg, random_seed=42).simulate_scenario(
        base, ktz_baseline["revenue"], ktz_baseline["operating_costs"])
    with_prior = MonteCarloSimulator(cfg, random_seed=42).simulate_scenario(
        risky, ktz_baseline["revenue"], ktz_baseline["operating_costs"])

    assert with_prior.success_probability < no_prior.success_probability, (
        f"Empirical failure rate did not propagate: "
        f"no_prior P(NPV>0)={no_prior.success_probability:.2f}, "
        f"with_prior P(NPV>0)={with_prior.success_probability:.2f}"
    )


def test_empirical_failure_rate_clipped():
    """Empirical failure rate should be clipped to [2%, 60%]
    to prevent a single bad case-set from killing all scenarios."""
    base = {
        "id": 1,
        "name": "Test",
        "expected_impact": {"revenue_increase": 0.04, "cost_reduction": 0.04},
        "investment_required": 5_000_000.0,
        "implementation_time_months": 12,
        "risk_level": "medium",
        "empirical_prior": {
            "revenue_uplift": {"n": 10, "mean": 0.04, "std": 0.05,
                               "p10": -0.02, "p50": 0.04, "p90": 0.10},
            "failure_rate": 0.99,
        },
    }
    cfg = SimulationConfig(iterations=2000)
    res = MonteCarloSimulator(cfg, random_seed=42).simulate_scenario(
        base, 500_000_000, 450_000_000)
    # Even with 99% empirical failure, we clip to 60% so the prob of NPV>0
    # should be at least ~30%+ (some upside still survives clipping).
    assert res.success_probability > 0.20, (
        f"Clipping failed: P(NPV>0)={res.success_probability:.2f}"
    )
