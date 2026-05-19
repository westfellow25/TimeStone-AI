"""Shared pytest fixtures."""
import pytest


@pytest.fixture
def ktz_baseline():
    """KTZ baseline financials used across simulator tests."""
    return {
        "revenue": 500_000_000.0,
        "operating_costs": 450_000_000.0,
    }


@pytest.fixture
def low_risk_scenario():
    return {
        "id": 1,
        "name": "Digital Booking Platform",
        "expected_impact": {
            "revenue_increase": 0.02,
            "cost_reduction": 0.02,
            "profit_margin_improvement": 0.04,
        },
        "investment_required": 2_000_000.0,
        "implementation_time_months": 6,
        "risk_level": "low",
    }


@pytest.fixture
def medium_risk_scenario():
    return {
        "id": 2,
        "name": "Dynamic Pricing Implementation",
        "expected_impact": {
            "revenue_increase": 0.03,
            "cost_reduction": 0.005,
            "profit_margin_improvement": 0.035,
        },
        "investment_required": 3_500_000.0,
        "implementation_time_months": 9,
        "risk_level": "medium",
    }


@pytest.fixture
def high_risk_scenario():
    return {
        "id": 3,
        "name": "Predictive Maintenance System",
        "expected_impact": {
            "revenue_increase": 0.01,
            "cost_reduction": 0.05,
            "profit_margin_improvement": 0.06,
        },
        "investment_required": 10_000_000.0,
        "implementation_time_months": 18,
        "risk_level": "high",
    }


@pytest.fixture
def realistic_scenarios(low_risk_scenario, medium_risk_scenario, high_risk_scenario):
    return [low_risk_scenario, medium_risk_scenario, high_risk_scenario]
