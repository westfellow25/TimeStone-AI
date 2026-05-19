"""
Transformation Scenario Generator

Generates business transformation hypotheses based on company profile,
industry benchmarks, and (optionally) a Case Library of real transformations.

Two modes:
- Rule-based (default): uses hard-coded industry templates with calibrated
  parameter ranges based on published benchmarks.
- Case-library-backed: if a CaseLibrary is provided, parameter ranges are
  pulled from empirical distributions of real similar transformations.
  Each scenario records which cases informed its priors.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import random


class TransformationType(Enum):
    """Types of business transformations"""
    DIGITAL = "digital_transformation"
    PRICING = "pricing_optimization"
    OPERATIONS = "operational_efficiency"
    AUTOMATION = "process_automation"
    MARKET_EXPANSION = "market_expansion"
    PRODUCT_INNOVATION = "product_innovation"
    SUPPLY_CHAIN = "supply_chain_optimization"
    CUSTOMER_EXPERIENCE = "customer_experience"


@dataclass
class TransformationScenario:
    """Individual transformation scenario"""
    id: int
    name: str
    transformation_type: TransformationType
    description: str
    expected_impact: Dict[str, float]
    investment_required: float
    implementation_time_months: int
    risk_level: str  # "low", "medium", "high"
    based_on_cases: List[str] = field(default_factory=list)  # case IDs from CaseLibrary
    empirical_prior: Optional[Dict] = None                    # raw prior stats, for transparency

    def __repr__(self):
        return (f"Scenario #{self.id}: {self.name} "
                f"(Type: {self.transformation_type.value}, "
                f"Impact: {self.expected_impact.get('revenue_increase', 0):.1%})")


class ScenarioGenerator:
    """
    Generates transformation scenarios for simulation.

    If a CaseLibrary is supplied via `case_library` plus a company profile,
    parameter ranges are pulled from empirical distributions of real similar
    transformations rather than from hard-coded templates.
    """

    def __init__(
        self,
        company_name: str,
        industry: str,
        case_library=None,
        company_profile: Optional[Dict] = None,
    ):
        self.company_name = company_name
        self.industry = industry
        self.scenarios: List[TransformationScenario] = []
        self.case_library = case_library
        self.company_profile = company_profile or {}

    def generate_scenarios(self, count: int = 1000) -> List[TransformationScenario]:
        """Generate `count` transformation scenarios.

        Behaviour:
        - If case_library is available -> retrieved priors per scenario.
        - Otherwise -> rule-based templates.
        """
        self.scenarios = []
        scenario_templates = self._get_scenario_templates()
        for i in range(count):
            template = random.choice(scenario_templates)
            if self.case_library is not None:
                scenario = self._create_scenario_from_case_library(i + 1, template)
            else:
                scenario = self._create_scenario_from_template(i + 1, template)
            self.scenarios.append(scenario)
        return self.scenarios

    # ----------------------------------------------------------------
    # Rule-based templates (calibrated against public benchmarks)
    # ----------------------------------------------------------------
    def _get_scenario_templates(self) -> List[Dict]:
        # revenue_impact = expected annual recurring revenue uplift as % of baseline
        # cost_impact    = expected annual reduction as % of operating_costs
        if "transportation" in self.industry.lower() or "rail" in self.industry.lower():
            return [
                {
                    "type": TransformationType.PRICING,
                    "name": "Dynamic Pricing Implementation",
                    "desc": "Real-time pricing based on demand, route, and capacity",
                    "revenue_impact": (0.02, 0.05),
                    "cost_impact": (0.00, 0.01),
                    "investment": (2_000_000, 5_000_000),
                    "time": (6, 12),
                    "risk": "medium",
                },
                {
                    "type": TransformationType.AUTOMATION,
                    "name": "AI-Powered Route Optimization",
                    "desc": "Machine learning for optimal routing and scheduling",
                    "revenue_impact": (0.01, 0.03),
                    "cost_impact": (0.03, 0.06),
                    "investment": (3_000_000, 8_000_000),
                    "time": (8, 15),
                    "risk": "medium",
                },
                {
                    "type": TransformationType.OPERATIONS,
                    "name": "Predictive Maintenance System",
                    "desc": "IoT sensors + AI for predictive equipment maintenance",
                    "revenue_impact": (0.00, 0.02),
                    "cost_impact": (0.03, 0.08),
                    "investment": (5_000_000, 15_000_000),
                    "time": (12, 24),
                    "risk": "high",
                },
                {
                    "type": TransformationType.DIGITAL,
                    "name": "Digital Booking Platform",
                    "desc": "Modern online booking system with mobile app",
                    "revenue_impact": (0.01, 0.03),
                    "cost_impact": (0.01, 0.03),
                    "investment": (1_000_000, 3_000_000),
                    "time": (4, 8),
                    "risk": "low",
                },
                {
                    "type": TransformationType.CUSTOMER_EXPERIENCE,
                    "name": "Real-Time Tracking & Notifications",
                    "desc": "GPS tracking with automated customer notifications",
                    "revenue_impact": (0.005, 0.02),
                    "cost_impact": (0.005, 0.02),
                    "investment": (500_000, 2_000_000),
                    "time": (3, 6),
                    "risk": "low",
                },
                {
                    "type": TransformationType.AUTOMATION,
                    "name": "Automated Freight Classification",
                    "desc": "Computer vision for cargo classification and routing",
                    "revenue_impact": (0.01, 0.03),
                    "cost_impact": (0.02, 0.05),
                    "investment": (2_000_000, 6_000_000),
                    "time": (9, 15),
                    "risk": "medium",
                },
            ]

        # Generic
        return [
            {
                "type": TransformationType.DIGITAL,
                "name": "Digital Transformation Initiative",
                "desc": "Comprehensive digital modernization",
                "revenue_impact": (0.02, 0.06),
                "cost_impact": (0.02, 0.05),
                "investment": (1_000_000, 10_000_000),
                "time": (6, 18),
                "risk": "medium",
            },
            {
                "type": TransformationType.AUTOMATION,
                "name": "Process Automation",
                "desc": "RPA and workflow automation",
                "revenue_impact": (0.01, 0.04),
                "cost_impact": (0.03, 0.08),
                "investment": (500_000, 5_000_000),
                "time": (4, 12),
                "risk": "low",
            },
        ]

    # ----------------------------------------------------------------
    # Path A: classic rule-based
    # ----------------------------------------------------------------
    def _create_scenario_from_template(
        self,
        scenario_id: int,
        template: Dict,
    ) -> TransformationScenario:
        revenue_impact = random.uniform(*template["revenue_impact"])
        cost_impact = random.uniform(*template["cost_impact"])
        investment = random.uniform(*template["investment"])
        impl_time = random.randint(*template["time"])

        variation_suffix = ""
        if scenario_id % 10 == 0:
            variation_suffix = " (Aggressive)"
            revenue_impact *= 1.2
            investment *= 1.3
        elif scenario_id % 7 == 0:
            variation_suffix = " (Conservative)"
            revenue_impact *= 0.8
            investment *= 0.7

        return TransformationScenario(
            id=scenario_id,
            name=template["name"] + variation_suffix,
            transformation_type=template["type"],
            description=template["desc"],
            expected_impact={
                "revenue_increase": revenue_impact,
                "cost_reduction": cost_impact,
                "profit_margin_improvement": revenue_impact + cost_impact,
            },
            investment_required=investment,
            implementation_time_months=impl_time,
            risk_level=template["risk"],
            based_on_cases=[],
            empirical_prior=None,
        )

    # ----------------------------------------------------------------
    # Path B: case-library-backed (empirical priors)
    # ----------------------------------------------------------------
    def _create_scenario_from_case_library(
        self,
        scenario_id: int,
        template: Dict,
    ) -> TransformationScenario:
        from src.knowledge.case_library import Query

        query = Query(
            industry=self.company_profile.get("industry"),
            industry_tags=self.company_profile.get("industry_tags", []),
            revenue_usd=self.company_profile.get("revenue_usd"),
            transformation_type=template["type"].value,
            geography=self.company_profile.get("geography"),
        )
        retrieved = self.case_library.find_similar(query, k=5)

        # Empirical prior on revenue uplift
        rev_prior = self.case_library.empirical_prior(
            retrieved, "actual_revenue_uplift_pct",
            fallback=(sum(template["revenue_impact"]) / 2, 0.02),
        )
        cost_prior = self.case_library.empirical_prior(
            retrieved, "actual_cost_reduction_pct",
            fallback=(sum(template["cost_impact"]) / 2, 0.02),
        )

        # Blended prior: shrink empirical mean toward the template midpoint when
        # sample size is small (n<5). At n>=5 trust empirical fully. Cap extreme
        # samples to a realistic window so a single outlier (Boeing, Domino's)
        # cannot dominate the simulation.
        rev_template_mid = sum(template["revenue_impact"]) / 2
        cost_template_mid = sum(template["cost_impact"]) / 2

        def _blended_sample(prior, template_mid, lo_cap, hi_cap):
            n = prior["n"]
            if n == 0:
                return random.uniform(lo_cap, hi_cap)
            w = min(n / 5.0, 1.0)
            mean = w * prior["mean"] + (1 - w) * template_mid
            std = max(prior["std"], 0.01)
            sample = random.gauss(mean, std)
            return max(lo_cap, min(hi_cap, sample))

        revenue_impact = _blended_sample(rev_prior, rev_template_mid, -0.05, 0.15)
        cost_impact = _blended_sample(cost_prior, cost_template_mid, -0.02, 0.15)

        # Investment & time still from template (no good way to scale by retrieved company size yet)
        investment = random.uniform(*template["investment"])
        impl_time = random.randint(*template["time"])

        # Risk level: derive from empirical failure rate among similar cases
        failure_rate = self.case_library.failure_rate(retrieved)
        if failure_rate >= 0.5:
            risk_level = "high"
        elif failure_rate >= 0.25:
            risk_level = "medium"
        else:
            risk_level = "low"

        return TransformationScenario(
            id=scenario_id,
            name=template["name"],
            transformation_type=template["type"],
            description=template["desc"],
            expected_impact={
                "revenue_increase": revenue_impact,
                "cost_reduction": cost_impact,
                "profit_margin_improvement": revenue_impact + cost_impact,
            },
            investment_required=investment,
            implementation_time_months=impl_time,
            risk_level=risk_level,
            based_on_cases=[c.id for c, _ in retrieved],
            empirical_prior={
                "revenue_uplift": rev_prior,
                "cost_reduction": cost_prior,
                "failure_rate": failure_rate,
            },
        )

    # ----------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------
    def get_top_scenarios(self, n: int = 10) -> List[TransformationScenario]:
        def calculate_roi(scenario: TransformationScenario) -> float:
            total_impact = (scenario.expected_impact["revenue_increase"]
                            + scenario.expected_impact["cost_reduction"])
            return total_impact / (scenario.investment_required / 1_000_000)
        return sorted(self.scenarios, key=calculate_roi, reverse=True)[:n]

    def save_scenarios(self, filepath: str):
        import json
        data = {
            "company": self.company_name,
            "industry": self.industry,
            "total_scenarios": len(self.scenarios),
            "uses_case_library": self.case_library is not None,
            "scenarios": [
                {
                    "id": s.id,
                    "name": s.name,
                    "type": s.transformation_type.value,
                    "description": s.description,
                    "expected_impact": s.expected_impact,
                    "investment_required": s.investment_required,
                    "implementation_time_months": s.implementation_time_months,
                    "risk_level": s.risk_level,
                    "based_on_cases": s.based_on_cases,
                    "empirical_prior": s.empirical_prior,
                }
                for s in self.scenarios
            ],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    print("Generating transformation scenarios for KTZ...")
    print("=" * 60)

    # Try with case library first; fall back to rule-based if missing
    case_library = None
    case_lib_path = Path(__file__).resolve().parents[2] / "data" / "case_library.json"
    if case_lib_path.exists():
        try:
            from src.knowledge.case_library import CaseLibrary
            case_library = CaseLibrary.from_json_file(str(case_lib_path))
            print(f"Loaded {len(case_library)} cases for empirical priors.\n")
        except Exception as exc:
            print(f"Could not load case library ({exc}); using rule-based mode.\n")

    company_profile = {
        "industry": "transportation_rail",
        "industry_tags": ["transportation", "logistics"],
        "revenue_usd": 500_000_000,
        "geography": "Kazakhstan",
    }

    generator = ScenarioGenerator(
        company_name="Kazakhstan Temir Zholy (KTZ)",
        industry="Transportation & Logistics",
        case_library=case_library,
        company_profile=company_profile,
    )
    scenarios = generator.generate_scenarios(count=1000)

    print(f"Generated {len(scenarios)} scenarios "
          f"({'with empirical priors' if case_library else 'rule-based'}).\n")
    print("Top 10 scenarios by expected ROI:")
    print("-" * 60)
    for i, scenario in enumerate(generator.get_top_scenarios(n=10), 1):
        print(f"\n{i}. {scenario.name}  [risk={scenario.risk_level}]")
        print(f"   Type: {scenario.transformation_type.value}")
        print(f"   Revenue Impact: {scenario.expected_impact['revenue_increase']:+.1%}")
        print(f"   Cost Reduction: {scenario.expected_impact['cost_reduction']:+.1%}")
        print(f"   Investment: ${scenario.investment_required:,.0f}")
        print(f"   Timeline: {scenario.implementation_time_months} months")
        if scenario.based_on_cases:
            print(f"   Based on cases: {', '.join(scenario.based_on_cases[:3])}")

    generator.save_scenarios("ktz_scenarios.json")
    print(f"\n{'=' * 60}")
    print("Scenarios saved to ktz_scenarios.json")
