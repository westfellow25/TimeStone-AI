"""Tests for the rule-based scenario generator."""
from src.simulation.scenario_generator import ScenarioGenerator, TransformationType


def test_generates_requested_count():
    gen = ScenarioGenerator("Test Co", "Transportation & Logistics")
    scenarios = gen.generate_scenarios(count=100)
    assert len(scenarios) == 100


def test_scenarios_have_required_fields():
    gen = ScenarioGenerator("Test Co", "Transportation & Logistics")
    scenarios = gen.generate_scenarios(count=20)
    for s in scenarios:
        assert s.expected_impact["revenue_increase"] >= 0.0
        assert s.expected_impact["cost_reduction"] >= 0.0
        assert s.investment_required > 0
        assert s.implementation_time_months > 0
        assert s.risk_level in {"low", "medium", "high"}
        assert isinstance(s.transformation_type, TransformationType)


def test_realistic_revenue_impact_caps():
    """After the realism fix, no single scenario should claim >10% annual revenue uplift."""
    gen = ScenarioGenerator("Test Co", "Transportation & Logistics")
    scenarios = gen.generate_scenarios(count=500)
    for s in scenarios:
        assert s.expected_impact["revenue_increase"] < 0.10, (
            f"Scenario #{s.id} claims unrealistic revenue uplift: "
            f"{s.expected_impact['revenue_increase']:.1%}"
        )


def test_save_and_load_scenarios(tmp_path):
    gen = ScenarioGenerator("Test Co", "Transportation & Logistics")
    gen.generate_scenarios(count=10)
    fp = tmp_path / "scenarios.json"
    gen.save_scenarios(str(fp))

    import json
    data = json.loads(fp.read_text())
    assert data["company"] == "Test Co"
    assert data["total_scenarios"] == 10
    assert len(data["scenarios"]) == 10
