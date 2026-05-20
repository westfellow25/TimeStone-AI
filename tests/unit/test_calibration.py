"""Unit tests for the calibration service."""
import pytest

from timestone.domain.outcome import OutcomeRecord
from timestone.services.calibration import (
    compute_calibration, apply_calibration_to_prior, CalibrationTable,
)


def _outcome(predicted_rev=0.05, actual_rev=0.02,
             predicted_cost=0.03, actual_cost=0.01,
             predicted_p=0.7, status="partial", oid="o1") -> OutcomeRecord:
    return OutcomeRecord(
        id=oid, run_id="r1", company_name="X", scenario_id=1, scenario_name="s",
        prediction_date="2024-01-01", measurement_date="2025-01-01",
        months_elapsed=12,
        predicted_mean_npv=10_000_000,
        predicted_success_probability=predicted_p,
        predicted_revenue_uplift_pct=predicted_rev,
        predicted_cost_reduction_pct=predicted_cost,
        predicted_investment_usd=2_000_000, predicted_payback_years=2.5,
        actual_revenue_uplift_pct=actual_rev,
        actual_cost_reduction_pct=actual_cost,
        actual_investment_usd=2_500_000,
        actual_status=status,
    )


def test_empty_outcomes_produces_empty_table():
    table = compute_calibration([])
    assert table.global_residual.n_outcomes == 0
    assert table.global_residual.revenue_uplift_residual == 0.0


def test_residuals_are_actual_minus_predicted():
    outcomes = [
        _outcome(predicted_rev=0.05, actual_rev=0.02, oid="a"),
        _outcome(predicted_rev=0.05, actual_rev=0.03, oid="b"),
    ]
    table = compute_calibration(outcomes)
    # mean residual = average(-0.03, -0.02) = -0.025
    assert table.global_residual.revenue_uplift_residual == pytest.approx(-0.025, abs=1e-6)


def test_apply_calibration_shifts_mean_by_damped_residual():
    table = CalibrationTable.empty()
    entry = table.global_residual
    entry.revenue_uplift_residual = -0.04
    entry.n_outcomes = 5  # full damping weight = 1.0
    prior = {"mean": 0.05, "std": 0.02, "n": 5}
    shifted = apply_calibration_to_prior(prior, entry, "actual_revenue_uplift_pct")
    assert shifted["mean"] == pytest.approx(0.05 + (-0.04 * 1.0))
    assert shifted["calibration_shift"] == pytest.approx(-0.04)


def test_calibration_damped_at_low_n():
    """At n=1, damping factor = 1/5 = 0.2, so shift should be 20% of residual."""
    table = CalibrationTable.empty()
    entry = table.global_residual
    entry.revenue_uplift_residual = -0.04
    entry.n_outcomes = 1
    prior = {"mean": 0.05, "std": 0.02, "n": 5}
    shifted = apply_calibration_to_prior(prior, entry, "actual_revenue_uplift_pct")
    assert shifted["calibration_shift"] == pytest.approx(-0.04 * 0.2)


def test_roundtrip_calibration_table(tmp_path):
    from timestone.services.calibration import save_calibration, load_calibration
    outcomes = [_outcome(oid="x"), _outcome(oid="y")]
    table = compute_calibration(outcomes)
    p = tmp_path / "calibration.json"
    save_calibration(table, p)
    loaded = load_calibration(p)
    assert loaded.global_residual.n_outcomes == 2
