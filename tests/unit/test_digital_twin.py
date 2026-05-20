"""Unit tests for Company / CompanyMetrics domain objects."""
import pytest

from timestone.domain.company import Company, CompanyMetrics


def test_profit_margin_basic():
    m = CompanyMetrics(revenue=1000.0, operating_costs=700.0)
    assert m.computed_profit_margin == pytest.approx(0.30)


def test_profit_margin_zero_revenue():
    m = CompanyMetrics(revenue=0.0, operating_costs=0.0)
    assert m.computed_profit_margin == 0.0


def test_revenue_per_employee():
    m = CompanyMetrics(revenue=1_000_000.0, operating_costs=800_000.0, employees=10)
    assert m.revenue_per_employee() == 100_000.0


def test_revenue_per_employee_zero_employees():
    m = CompanyMetrics(revenue=1000.0, operating_costs=800.0, employees=0)
    assert m.revenue_per_employee() == 0.0


def test_company_roundtrip():
    c = Company(
        company_name="Test",
        metrics=CompanyMetrics(
            revenue=1e9, operating_costs=9e8, employees=1000,
            industry="manufacturing", geography="USA",
            industry_tags=["manufacturing", "B2B"]))
    payload = c.to_dict()
    c2 = Company.from_dict(payload)
    assert c2.company_name == c.company_name
    assert c2.metrics.revenue == c.metrics.revenue
    assert c2.metrics.industry_tags == c.metrics.industry_tags


def test_company_from_legacy_format():
    """Old twin JSONs use 'metrics' dict — must still load."""
    legacy = {
        "company_name": "Legacy Co",
        "metrics": {"revenue": 100.0, "industry": "saas"}}
    c = Company.from_dict(legacy)
    assert c.company_name == "Legacy Co"
    assert c.metrics.revenue == 100.0
    assert c.metrics.industry == "saas"


def test_company_with_segments_competitors_priors():
    """Rich Company model: segments, competitors, prior_transformations round-trip."""
    from timestone.domain.company import (
        Company, CompanyMetrics, BusinessSegment, Competitor, PriorTransformation,
    )
    c = Company(
        company_name="Rich Co",
        metrics=CompanyMetrics(
            revenue=1e9, operating_costs=9e8, employees=10000,
            industry="banking", geography="KZ",
            segments=[BusinessSegment(name="Retail", revenue_share=0.6, margin_pct=0.15)],
            competitors=[Competitor(name="Halyk", relationship="direct", market_share_ratio=0.5)],
            prior_transformations=[
                PriorTransformation(name="ERP rollout", year=2020, outcome="partial",
                                     investment_usd=50_000_000)],
            strategic_priorities=["Mobile-first", "ESG"],
            pain_points=["Tech talent shortage"],
        ))
    payload = c.to_dict()
    c2 = Company.from_dict(payload)
    assert len(c2.metrics.segments) == 1
    assert c2.metrics.segments[0].name == "Retail"
    assert len(c2.metrics.competitors) == 1
    assert c2.metrics.competitors[0].name == "Halyk"
    assert len(c2.metrics.prior_transformations) == 1
    assert c2.metrics.prior_transformations[0].outcome == "partial"
    assert c2.metrics.strategic_priorities == ["Mobile-first", "ESG"]
    assert c2.metrics.pain_points == ["Tech talent shortage"]


def test_legacy_twin_still_loads_without_new_fields():
    """Old JSON without segments/competitors/etc. must continue to work."""
    from timestone.domain.company import Company
    legacy = {
        "company_name": "Legacy", "metrics": {"revenue": 100.0, "industry": "saas"}}
    c = Company.from_dict(legacy)
    assert c.metrics.segments == []
    assert c.metrics.competitors == []
    assert c.metrics.prior_transformations == []
    assert c.metrics.strategic_priorities == []
