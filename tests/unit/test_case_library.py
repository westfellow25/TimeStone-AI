"""Unit tests for CaseLibrary retrieval and empirical priors."""
from pathlib import Path
import pytest

from timestone.domain.case import CaseQuery
from timestone.repositories.case_library import CaseLibraryRepository
from timestone.services.knowledge_retrieval import CaseLibrary, revenue_to_bucket


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_LIB_PATH = REPO_ROOT / "data" / "case_library.json"


@pytest.fixture(scope="module")
def lib() -> CaseLibrary:
    cases = CaseLibraryRepository(CASE_LIB_PATH).load_all()
    return CaseLibrary(cases)


def test_library_loads_min_cases(lib):
    assert len(lib) >= 25


def test_library_has_both_successes_and_failures(lib):
    statuses = {c.status for c in lib.cases}
    assert "success" in statuses and "failed" in statuses


def test_revenue_to_bucket():
    assert revenue_to_bucket(50_000_000) == "small"
    assert revenue_to_bucket(500_000_000) == "mid"
    assert revenue_to_bucket(20_000_000_000) == "large"
    assert revenue_to_bucket(200_000_000_000) == "mega"


def test_find_similar_returns_top_k(lib):
    q = CaseQuery(
        industry="transportation_rental",
        industry_tags=["transportation"],
        revenue_usd=500_000_000,
        transformation_type="digital_transformation",
        geography="USA")
    results = lib.find_similar(q, k=3)
    assert 1 <= len(results) <= 3
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_find_similar_prefers_industry_match(lib):
    q = CaseQuery(industry="retail_grocery")
    results = lib.find_similar(q, k=5)
    assert results[0][0].industry == "retail_grocery"


def test_empirical_prior_uses_actual_not_promised(lib):
    failed_only = [(c, 1.0) for c in lib.cases if c.status == "failed"]
    prior = lib.empirical_prior(failed_only, "actual_revenue_uplift_pct")
    assert prior["mean"] < 0.05


def test_empirical_prior_handles_empty():
    empty = CaseLibrary([])
    prior = empty.empirical_prior([], "actual_revenue_uplift_pct", fallback=(0.02, 0.01))
    assert prior["n"] == 0 and prior["mean"] == 0.02


def test_failure_rate_computation(lib):
    rate = lib.failure_rate([(c, 1.0) for c in lib.cases])
    assert 0.3 < rate < 0.9


def test_query_with_no_matches_returns_empty(lib):
    q = CaseQuery(industry="space_tourism", transformation_type="rocket_science")
    assert lib.find_similar(q, k=5, min_score=10.0) == []


def test_prior_failure_modes_boost_score(lib):
    """If query carries a failure_mode the case shares, score must increase."""
    from timestone.domain.case import CaseQuery
    # Hertz case has failure_modes including 'vendor_underperformance'
    base = CaseQuery(industry="transportation_rental")
    boosted = CaseQuery(industry="transportation_rental",
                        prior_failure_modes=["vendor_underperformance"])
    hertz = next(c for c in lib.cases if c.id == "hertz-accenture-2016")
    s_base = lib.score(hertz, base)
    s_boosted = lib.score(hertz, boosted)
    assert s_boosted > s_base


def test_segment_keyword_boost(lib):
    """Segment keyword matching case description must increase score."""
    from timestone.domain.case import CaseQuery
    q_base = CaseQuery(industry="financial_services")
    q_boost = CaseQuery(industry="financial_services",
                        segment_keywords=["mobile"])
    # Find a banking case whose description likely contains 'mobile'
    for case in lib.cases:
        if "mobile" in (case.description.lower() + " " + case.transformation_subtype.lower()):
            assert lib.score(case, q_boost) > lib.score(case, q_base)
            break
    else:
        pytest.skip("No case with 'mobile' in description - test inapplicable")
