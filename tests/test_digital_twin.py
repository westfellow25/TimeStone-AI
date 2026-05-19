"""Tests for the DigitalTwin model."""
import pytest

from src.models.digital_twin import CompanyMetrics, DigitalTwin


def test_profit_margin_basic():
    m = CompanyMetrics(
        revenue=100.0, operating_costs=80.0, employee_count=10,
        market_share=0.1, customer_count=100, avg_transaction_value=1.0,
        growth_rate=0.05, industry="x",
    )
    assert m.profit_margin == pytest.approx(0.20)


def test_profit_margin_zero_revenue():
    m = CompanyMetrics(
        revenue=0.0, operating_costs=10.0, employee_count=1,
        market_share=0.0, customer_count=0, avg_transaction_value=0.0,
        growth_rate=0.0, industry="x",
    )
    assert m.profit_margin == 0.0


def test_revenue_per_employee():
    m = CompanyMetrics(
        revenue=1_000_000.0, operating_costs=500_000.0, employee_count=10,
        market_share=0.0, customer_count=0, avg_transaction_value=0.0,
        growth_rate=0.0, industry="x",
    )
    assert m.revenue_per_employee == 100_000.0


def test_revenue_per_employee_zero_employees():
    m = CompanyMetrics(
        revenue=1_000_000.0, operating_costs=500_000.0, employee_count=0,
        market_share=0.0, customer_count=0, avg_transaction_value=0.0,
        growth_rate=0.0, industry="x",
    )
    assert m.revenue_per_employee == 0.0


def test_twin_roundtrip(tmp_path):
    m = CompanyMetrics(
        revenue=500.0, operating_costs=400.0, employee_count=100,
        market_share=0.5, customer_count=1000, avg_transaction_value=10.0,
        growth_rate=0.1, industry="rail",
    )
    twin = DigitalTwin("Test Co", m, metadata={"country": "KZ"})
    fp = tmp_path / "twin.json"
    twin.save(str(fp))

    loaded = DigitalTwin.load(str(fp))
    assert loaded.company_name == "Test Co"
    assert loaded.metrics.revenue == 500.0
    assert loaded.metrics.profit_margin == pytest.approx(0.20)
    assert loaded.metadata == {"country": "KZ"}


def test_baseline_metrics_shape():
    m = CompanyMetrics(
        revenue=100.0, operating_costs=80.0, employee_count=10,
        market_share=0.1, customer_count=100, avg_transaction_value=1.0,
        growth_rate=0.05, industry="x",
    )
    twin = DigitalTwin("Co", m)
    baseline = twin.get_baseline()
    assert set(baseline) == {"revenue", "profit_margin", "market_share", "employee_productivity"}
