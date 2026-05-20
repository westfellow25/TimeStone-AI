"""Company / digital twin domain model.

The Company is a structured snapshot of a target organisation. Earlier
versions captured only 7 financial fields. To produce credible
transformation recommendations, the twin now also carries:

  - Business segments (revenue mix)
  - Competitive landscape
  - Prior transformations (what was tried, what worked)
  - Strategic priorities
  - Known pain points

All new fields are optional. Legacy twins continue to load unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BusinessSegment:
    """One revenue-producing segment of the company."""
    name: str
    revenue_share: float = 0.0          # 0.0 - 1.0
    description: str = ""
    growth_rate_pct: Optional[float] = None   # YoY change, e.g. 0.05 = +5%
    margin_pct: Optional[float] = None         # segment operating margin

    @classmethod
    def from_dict(cls, d: Dict) -> "BusinessSegment":
        return cls(
            name=d.get("name", ""),
            revenue_share=float(d.get("revenue_share", 0)),
            description=d.get("description", ""),
            growth_rate_pct=d.get("growth_rate_pct"),
            margin_pct=d.get("margin_pct"),
        )

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Competitor:
    """One competitor with our relationship type and relative scale."""
    name: str
    relationship: str = "direct"         # direct / adjacent / potential / substitute
    market_share_ratio: Optional[float] = None   # our share / their share
    notes: str = ""

    @classmethod
    def from_dict(cls, d: Dict) -> "Competitor":
        return cls(
            name=d.get("name", ""),
            relationship=d.get("relationship", "direct"),
            market_share_ratio=d.get("market_share_ratio"),
            notes=d.get("notes", ""),
        )

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class PriorTransformation:
    """A transformation initiative the company has already attempted.

    Captured to (a) avoid recommending what failed before and (b) inform
    risk priors for similar future scenarios at this specific company.
    """
    name: str
    year: int
    outcome: str = "ongoing"             # success / partial / failed / ongoing
    description: str = ""
    investment_usd: Optional[float] = None
    learnings: str = ""

    @classmethod
    def from_dict(cls, d: Dict) -> "PriorTransformation":
        return cls(
            name=d.get("name", ""),
            year=int(d.get("year", 0)),
            outcome=d.get("outcome", "ongoing"),
            description=d.get("description", ""),
            investment_usd=d.get("investment_usd"),
            learnings=d.get("learnings", ""),
        )

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class CompanyMetrics:
    """Baseline financial, operational and contextual metrics."""
    # Financial
    revenue: float = 0.0
    operating_costs: float = 0.0
    employees: int = 0
    profit_margin: float = 0.0
    market_share: float = 0.0

    # Identity
    industry: str = ""
    geography: str = ""
    industry_tags: List[str] = field(default_factory=list)

    # Strategic context (all optional - filled progressively per engagement)
    segments: List[BusinessSegment] = field(default_factory=list)
    competitors: List[Competitor] = field(default_factory=list)
    prior_transformations: List[PriorTransformation] = field(default_factory=list)
    strategic_priorities: List[str] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)

    def revenue_per_employee(self) -> float:
        if self.employees == 0:
            return 0.0
        return self.revenue / self.employees

    @property
    def computed_profit_margin(self) -> float:
        if self.revenue == 0:
            return 0.0
        return (self.revenue - self.operating_costs) / self.revenue

    def to_dict(self) -> Dict:
        return {
            "revenue": self.revenue,
            "operating_costs": self.operating_costs,
            "employees": self.employees,
            "profit_margin": self.profit_margin,
            "market_share": self.market_share,
            "industry": self.industry,
            "geography": self.geography,
            "industry_tags": self.industry_tags,
            "segments": [s.to_dict() for s in self.segments],
            "competitors": [c.to_dict() for c in self.competitors],
            "prior_transformations": [t.to_dict() for t in self.prior_transformations],
            "strategic_priorities": self.strategic_priorities,
            "pain_points": self.pain_points,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CompanyMetrics":
        return cls(
            revenue=float(d.get("revenue", 0)),
            operating_costs=float(d.get("operating_costs", d.get("revenue", 0) * 0.9)),
            employees=int(d.get("employees", 0)),
            profit_margin=float(d.get("profit_margin", 0)),
            market_share=float(d.get("market_share", 0)),
            industry=d.get("industry", ""),
            geography=d.get("geography", ""),
            industry_tags=d.get("industry_tags", []),
            segments=[BusinessSegment.from_dict(s) for s in d.get("segments", [])],
            competitors=[Competitor.from_dict(c) for c in d.get("competitors", [])],
            prior_transformations=[PriorTransformation.from_dict(t)
                                    for t in d.get("prior_transformations", [])],
            strategic_priorities=d.get("strategic_priorities", []),
            pain_points=d.get("pain_points", []),
        )


@dataclass
class Company:
    """Digital twin of a target company."""
    company_name: str
    metrics: CompanyMetrics
    notes: str = ""
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: Dict) -> "Company":
        m = data.get("metrics", {})
        # Legacy: top-level geography
        if "geography" not in m and "geography" in data:
            m = {**m, "geography": data["geography"]}
        return cls(
            company_name=data.get("company_name", ""),
            metrics=CompanyMetrics.from_dict(m),
            notes=data.get("notes", ""),
            created_at=data.get("created_at", ""),
        )

    def to_dict(self) -> Dict:
        return {
            "company_name": self.company_name,
            "metrics": self.metrics.to_dict(),
            "notes": self.notes,
            "created_at": self.created_at,
        }
