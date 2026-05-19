"""Tests for the CaseLibrary knowledge module."""
from pathlib import Path

import pytest

from src.knowledge.case_library import (
    Case,
    CaseLibrary,
    Query,
    revenue_to_bucket,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_LIB_PATH = REPO_ROOT / "data" / "case_library.json"


@pytest.fixture(scope="module")
def lib() -> CaseLibrary:
    return CaseLibrary.from_json_file(str(CASE_LIB_PATH))


def test_library_loads_min_cases(lib):
    assert len(lib) >= 10, "Seed library should have at least 10 cases"


def test_library_has_both_successes_and_failures(lib):
    statuses = {c.status for c in lib.cases}
    assert "success" in statuses
    assert "failed" in statuses, "Need failure cases to avoid optimism bias"


def test_revenue_to_bucket():
    assert revenue_to_bucket(50_000_000) == "small"
    assert revenue_to_bucket(500_000_000) == "mid"
    assert revenue_to_bucket(20_000_000_000) == "large"
    assert revenue_to_bucket(200_000_000_000) == "mega"


def test_find_similar_returns_top_k(lib):
    q = Query(
        industry="transportation_rental",
        industry_tags=["transportation"],
        revenue_usd=500_000_000,
        transformation_type="digital_transformation",
        geography="USA",
    )
    results = lib.find_similar(q, k=3)
    assert 1 <= len(results) <= 3
    # Sorted descending by score
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_find_similar_prefers_industry_match(lib):
    q = Query(industry="retail_grocery")
    results = lib.find_similar(q, k=5)
    # Lidl should be the closest exact match for retail_grocery
    top = results[0][0]
    assert top.industry == "retail_grocery"


def test_empirical_prior_uses_actual_not_promised(lib):
    # Filter to only failed cases - actual_revenue_uplift should not match promised
    failed_only = [(c, 1.0) for c in lib.cases if c.status == "failed"]
    prior = lib.empirical_prior(failed_only, "actual_revenue_uplift_pct")
    # Failed projects should have low/negative mean uplift, not the promised 5-10%
    assert prior["mean"] < 0.05, (
        f"Failed cases empirical mean should be low, got {prior['mean']:.3f}"
    )


def test_empirical_prior_handles_empty():
    lib = CaseLibrary([])
    prior = lib.empirical_prior([], "actual_revenue_uplift_pct", fallback=(0.02, 0.01))
    assert prior["n"] == 0
    assert prior["mean"] == 0.02


def test_failure_rate_computation(lib):
    # Across all cases, failure rate should be > 0 (we curated some failures)
    all_with_score = [(c, 1.0) for c in lib.cases]
    rate = lib.failure_rate(all_with_score)
    assert 0.3 < rate < 0.9, f"Sanity check on failure rate: got {rate:.2f}"


def test_query_with_no_matches_returns_empty(lib):
    q = Query(industry="space_tourism", transformation_type="rocket_science")
    results = lib.find_similar(q, k=5, min_score=10.0)
    assert results == []


def test_scenario_generator_uses_case_library():
    """Integration: scenario generator with case library produces based_on_cases field."""
    from src.simulation.scenario_generator import ScenarioGenerator
    lib = CaseLibrary.from_json_file(str(CASE_LIB_PATH))
    gen = ScenarioGenerator(
        company_name="Test Co",
        industry="Transportation & Logistics",
        case_library=lib,
        company_profile={
            "industry": "transportation_rental",
            "industry_tags": ["transportation", "logistics"],
            "revenue_usd": 500_000_000,
            "geography": "USA",
        },
    )
    scenarios = gen.generate_scenarios(count=20)
    with_cases = [s for s in scenarios if s.based_on_cases]
    assert len(with_cases) > 0, "At least some scenarios should be backed by retrieved cases"
    # Each scenario should have empirical_prior metadata
    for s in with_cases:
        assert s.empirical_prior is not None
        assert "revenue_uplift" in s.empirical_prior
        assert "failure_rate" in s.empirical_prior


def test_blended_prior_caps_extreme_samples():
    """Regression guard: revenue_impact should not exceed 15% (our cap)."""
    from src.simulation.scenario_generator import ScenarioGenerator
    lib = CaseLibrary.from_json_file(str(CASE_LIB_PATH))
    gen = ScenarioGenerator(
        company_name="Test Co",
        industry="Transportation & Logistics",
        case_library=lib,
        company_profile={
            "industry": "transportation_rental",
            "industry_tags": ["transportation"],
            "revenue_usd": 500_000_000,
            "geography": "USA",
        },
    )
    scenarios = gen.generate_scenarios(count=500)
    for s in scenarios:
        assert -0.05 <= s.expected_impact["revenue_increase"] <= 0.15, (
            f"Revenue impact out of bounds: {s.expected_impact['revenue_increase']}"
        )
        assert -0.02 <= s.expected_impact["cost_reduction"] <= 0.15
