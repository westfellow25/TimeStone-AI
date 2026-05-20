"""Scenario generation - rule-based and case-library-backed."""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..domain.scenario import Scenario, TransformationType
from ..domain.case import CaseQuery
from .knowledge_retrieval import CaseLibrary


class ScenarioGenerator:
    """Generates transformation scenarios.

    If a CaseLibrary is supplied with a company profile, parameter ranges
    are pulled from empirical distributions of real similar transformations.
    Otherwise falls back to rule-based industry templates.
    """

    def __init__(
        self,
        company_name: str,
        industry: str,
        case_library: Optional[CaseLibrary] = None,
        company_profile: Optional[Dict] = None,
    ):
        self.company_name = company_name
        self.industry = industry
        self.scenarios: List[Scenario] = []
        self.case_library = case_library
        self.company_profile = company_profile or {}

    def generate(self, count: int = 1000) -> List[Scenario]:
        self.scenarios = []
        templates = self._templates()
        for i in range(count):
            template = random.choice(templates)
            if self.case_library is not None:
                scen = self._from_case_library(i + 1, template)
            else:
                scen = self._from_template(i + 1, template)
            self.scenarios.append(scen)
        return self.scenarios

    def _templates(self) -> List[Dict]:
        if "transportation" in self.industry.lower() or "rail" in self.industry.lower():
            return [
                {"type": TransformationType.PRICING, "name": "Dynamic Pricing Implementation",
                 "desc": "Real-time pricing based on demand, route, and capacity",
                 "revenue_impact": (0.02, 0.05), "cost_impact": (0.00, 0.01),
                 "investment": (2_000_000, 5_000_000), "time": (6, 12), "risk": "medium"},
                {"type": TransformationType.AUTOMATION, "name": "AI-Powered Route Optimization",
                 "desc": "Machine learning for optimal routing and scheduling",
                 "revenue_impact": (0.01, 0.03), "cost_impact": (0.03, 0.06),
                 "investment": (3_000_000, 8_000_000), "time": (8, 15), "risk": "medium"},
                {"type": TransformationType.OPERATIONS, "name": "Predictive Maintenance System",
                 "desc": "IoT sensors + AI for predictive equipment maintenance",
                 "revenue_impact": (0.00, 0.02), "cost_impact": (0.03, 0.08),
                 "investment": (5_000_000, 15_000_000), "time": (12, 24), "risk": "high"},
                {"type": TransformationType.DIGITAL, "name": "Digital Booking Platform",
                 "desc": "Modern online booking system with mobile app",
                 "revenue_impact": (0.01, 0.03), "cost_impact": (0.01, 0.03),
                 "investment": (1_000_000, 3_000_000), "time": (4, 8), "risk": "low"},
                {"type": TransformationType.CUSTOMER_EXPERIENCE, "name": "Real-Time Tracking & Notifications",
                 "desc": "GPS tracking with automated customer notifications",
                 "revenue_impact": (0.005, 0.02), "cost_impact": (0.005, 0.02),
                 "investment": (500_000, 2_000_000), "time": (3, 6), "risk": "low"},
                {"type": TransformationType.AUTOMATION, "name": "Automated Freight Classification",
                 "desc": "Computer vision for cargo classification and routing",
                 "revenue_impact": (0.01, 0.03), "cost_impact": (0.02, 0.05),
                 "investment": (2_000_000, 6_000_000), "time": (9, 15), "risk": "medium"},
            ]
        return [
            {"type": TransformationType.DIGITAL, "name": "Digital Transformation Initiative",
             "desc": "Comprehensive digital modernization",
             "revenue_impact": (0.02, 0.06), "cost_impact": (0.02, 0.05),
             "investment": (1_000_000, 10_000_000), "time": (6, 18), "risk": "medium"},
            {"type": TransformationType.AUTOMATION, "name": "Process Automation",
             "desc": "RPA and workflow automation",
             "revenue_impact": (0.01, 0.04), "cost_impact": (0.03, 0.08),
             "investment": (500_000, 5_000_000), "time": (4, 12), "risk": "low"},
        ]

    def _from_template(self, scenario_id: int, template: Dict) -> Scenario:
        revenue_impact = random.uniform(*template["revenue_impact"])
        cost_impact = random.uniform(*template["cost_impact"])
        investment = random.uniform(*template["investment"])
        impl_time = random.randint(*template["time"])
        suffix = ""
        if scenario_id % 10 == 0:
            suffix = " (Aggressive)"
            revenue_impact *= 1.2; investment *= 1.3
        elif scenario_id % 7 == 0:
            suffix = " (Conservative)"
            revenue_impact *= 0.8; investment *= 0.7
        return Scenario(
            id=scenario_id, name=template["name"] + suffix,
            transformation_type=template["type"], description=template["desc"],
            expected_impact={
                "revenue_increase": revenue_impact, "cost_reduction": cost_impact,
                "profit_margin_improvement": revenue_impact + cost_impact},
            investment_required=investment,
            implementation_time_months=impl_time,
            risk_level=template["risk"],
            based_on_cases=[], empirical_prior=None,
        )

    def _from_case_library(self, scenario_id: int, template: Dict) -> Scenario:
        query = CaseQuery(
            industry=self.company_profile.get("industry"),
            industry_tags=self.company_profile.get("industry_tags", []),
            revenue_usd=self.company_profile.get("revenue_usd"),
            transformation_type=template["type"].value,
            geography=self.company_profile.get("geography"),
            prior_failure_modes=self.company_profile.get("prior_failure_modes", []),
            segment_keywords=self.company_profile.get("segment_keywords", []),
        )
        retrieved = self.case_library.find_similar(query, k=5)
        rev_prior = self.case_library.empirical_prior(
            retrieved, "actual_revenue_uplift_pct",
            fallback=(sum(template["revenue_impact"]) / 2, 0.02))
        cost_prior = self.case_library.empirical_prior(
            retrieved, "actual_cost_reduction_pct",
            fallback=(sum(template["cost_impact"]) / 2, 0.02))

        rev_template_mid = sum(template["revenue_impact"]) / 2
        cost_template_mid = sum(template["cost_impact"]) / 2

        def blended(prior, template_mid, lo_cap, hi_cap):
            n = prior["n"]
            if n == 0:
                return random.uniform(lo_cap, hi_cap)
            w = min(n / 5.0, 1.0)
            mean = w * prior["mean"] + (1 - w) * template_mid
            std = max(prior["std"], 0.01)
            sample = random.gauss(mean, std)
            return max(lo_cap, min(hi_cap, sample))

        revenue_impact = blended(rev_prior, rev_template_mid, -0.05, 0.15)
        cost_impact = blended(cost_prior, cost_template_mid, -0.02, 0.15)
        investment = random.uniform(*template["investment"])
        impl_time = random.randint(*template["time"])

        failure_rate = self.case_library.failure_rate(retrieved)
        if failure_rate >= 0.5:
            risk_level = "high"
        elif failure_rate >= 0.25:
            risk_level = "medium"
        else:
            risk_level = "low"

        return Scenario(
            id=scenario_id, name=template["name"],
            transformation_type=template["type"], description=template["desc"],
            expected_impact={
                "revenue_increase": revenue_impact, "cost_reduction": cost_impact,
                "profit_margin_improvement": revenue_impact + cost_impact},
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
