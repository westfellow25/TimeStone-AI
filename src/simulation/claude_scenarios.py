"""
Claude-powered scenario generator.

Uses the Anthropic SDK to ask Claude to brainstorm unique, industry-specific
transformation scenarios with realistic financial assumptions. Falls back
gracefully to the rule-based ScenarioGenerator when ANTHROPIC_API_KEY is
not set.

Usage:
    from src.simulation.claude_scenarios import ClaudeScenarioGenerator
    gen = ClaudeScenarioGenerator(company_name="KTZ", industry="Transportation")
    scenarios = gen.generate_scenarios(count=1000)
"""

from __future__ import annotations

import json
import os
import random
from typing import List, Optional

from src.simulation.scenario_generator import (
    ScenarioGenerator,
    TransformationScenario,
    TransformationType,
)


CLAUDE_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

PROMPT_TEMPLATE = """You are a senior management consultant specializing in business transformation.

Generate {batch_size} unique, realistic transformation scenarios for:
- Company: {company_name}
- Industry: {industry}
- Annual revenue: ${revenue:,.0f}
- Operating costs: ${operating_costs:,.0f}
- Employees: {employee_count:,}

Each scenario should be DIFFERENT from the others - vary technology, scope, risk,
and time horizon. Stay grounded in published transformation case studies
(McKinsey, BCG, Gartner). Use realistic ANNUAL impact percentages:
- revenue_increase: 0.005 to 0.08 (0.5% to 8% annual recurring revenue lift)
- cost_reduction: 0.005 to 0.10 (% of operating costs)
- investment_required: $500K to $50M
- implementation_time_months: 3 to 36
- risk_level: "low", "medium", or "high"

Available transformation_type values:
    digital_transformation, pricing_optimization, operational_efficiency,
    process_automation, market_expansion, product_innovation,
    supply_chain_optimization, customer_experience

Return ONLY a JSON array, no commentary. Each item:
{{
  "name": "...",
  "transformation_type": "...",
  "description": "...",
  "revenue_increase": 0.0,
  "cost_reduction": 0.0,
  "investment_required": 0,
  "implementation_time_months": 0,
  "risk_level": "..."
}}
"""


class ClaudeScenarioGenerator(ScenarioGenerator):
    """ScenarioGenerator that uses Claude for creative variation.

    Inherits from rule-based generator so it can always fall back.
    """

    def __init__(
        self,
        company_name: str,
        industry: str,
        revenue: float = 500_000_000,
        operating_costs: float = 450_000_000,
        employee_count: int = 10_000,
        api_key: Optional[str] = None,
        model: str = CLAUDE_MODEL,
        batch_size: int = 25,
    ):
        super().__init__(company_name, industry)
        self.revenue = revenue
        self.operating_costs = operating_costs
        self.employee_count = employee_count
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.batch_size = batch_size
        self._client = None

    def _get_client(self):
        """Lazy-load anthropic client; returns None if SDK or key missing."""
        if self._client is not None:
            return self._client
        if not self.api_key:
            return None
        try:
            import anthropic  # noqa: WPS433 - local import for graceful fallback
        except ImportError:
            return None
        self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate_scenarios(self, count: int = 1000) -> List[TransformationScenario]:
        """Generate count scenarios via Claude, with rule-based fallback per batch."""
        client = self._get_client()
        if client is None:
            print("ANTHROPIC_API_KEY not set or SDK unavailable - falling back to rule-based generator.")
            return super().generate_scenarios(count=count)

        self.scenarios = []
        scenario_id = 1
        remaining = count
        batch_failures = 0
        max_batch_failures = 3

        while remaining > 0:
            batch = min(self.batch_size, remaining)
            try:
                scenarios = self._claude_batch(scenario_id, batch)
                self.scenarios.extend(scenarios)
                scenario_id += len(scenarios)
                remaining -= len(scenarios)
                print(f"  Generated {len(self.scenarios)}/{count} via Claude...")
            except Exception as exc:  # noqa: BLE001 - wide catch for any API issue
                batch_failures += 1
                print(f"  Batch failed ({exc!r}); failures: {batch_failures}/{max_batch_failures}")
                if batch_failures >= max_batch_failures:
                    print(f"  Too many Claude failures, filling remainder with rule-based generator.")
                    fallback_gen = ScenarioGenerator(self.company_name, self.industry)
                    fallback = fallback_gen.generate_scenarios(count=remaining)
                    # Re-id fallback scenarios to continue numbering
                    for s in fallback:
                        s.id = scenario_id
                        scenario_id += 1
                        self.scenarios.append(s)
                    remaining = 0
        return self.scenarios

    def _claude_batch(self, start_id: int, batch_size: int) -> List[TransformationScenario]:
        """Single API call -> batch_size scenarios."""
        client = self._client
        prompt = PROMPT_TEMPLATE.format(
            batch_size=batch_size,
            company_name=self.company_name,
            industry=self.industry,
            revenue=self.revenue,
            operating_costs=self.operating_costs,
            employee_count=self.employee_count,
        )
        message = client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        content = message.content[0].text.strip()
        # Strip code fences if present
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        items = json.loads(content)

        scenarios = []
        for offset, item in enumerate(items):
            try:
                t_type = TransformationType(item["transformation_type"])
            except (ValueError, KeyError):
                t_type = TransformationType.DIGITAL
            scenarios.append(TransformationScenario(
                id=start_id + offset,
                name=item.get("name", f"Scenario {start_id + offset}"),
                transformation_type=t_type,
                description=item.get("description", ""),
                expected_impact={
                    "revenue_increase": float(item.get("revenue_increase", 0.0)),
                    "cost_reduction": float(item.get("cost_reduction", 0.0)),
                    "profit_margin_improvement": float(item.get("revenue_increase", 0.0))
                                                + float(item.get("cost_reduction", 0.0)),
                },
                investment_required=float(item.get("investment_required", 1_000_000)),
                implementation_time_months=int(item.get("implementation_time_months", 12)),
                risk_level=item.get("risk_level", "medium"),
            ))
        return scenarios


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    gen = ClaudeScenarioGenerator(
        company_name="Kazakhstan Temir Zholy (KTZ)",
        industry="Transportation & Logistics",
        revenue=500_000_000,
        operating_costs=450_000_000,
        employee_count=10_000,
        batch_size=25,
    )
    scenarios = gen.generate_scenarios(count=100)
    gen.save_scenarios("ktz_scenarios.json")
    print(f"\nGenerated {len(scenarios)} scenarios.")
