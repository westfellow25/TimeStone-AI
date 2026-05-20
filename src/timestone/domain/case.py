"""TransformationCase domain model - a real historical transformation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CaseQuery:
    """Lookup criteria for retrieving similar transformation cases.

    Beyond basic similarity (industry, size, type, geo) the query may carry
    context from the target company that biases retrieval:
      - prior_failure_modes: failure tags the target has experienced before.
        Cases sharing these are upweighted ("learn from same mistake").
      - segment_keywords: tokens from the target's business segments.
        Loosely match case description and subtype.
      - require_geography_region: if set, restrict to nearby geos
        (used when caller is confident their case has region-specific dynamics).
    """
    industry: Optional[str] = None
    industry_tags: List[str] = field(default_factory=list)
    revenue_usd: Optional[float] = None
    transformation_type: Optional[str] = None
    geography: Optional[str] = None
    prior_failure_modes: List[str] = field(default_factory=list)
    segment_keywords: List[str] = field(default_factory=list)


@dataclass
class TransformationCase:
    """A real historical corporate transformation with promised and actual outcomes."""
    id: str
    company: str
    industry: str
    industry_tags: List[str]
    geography: str
    revenue_usd: float
    employees: int
    revenue_bucket: str
    transformation_type: str
    transformation_subtype: str
    description: str
    start_year: int
    planned_duration_months: int
    actual_duration_months: int
    status: str  # success / partial / failed / cancelled
    vendor: str
    promised_investment_usd: Optional[float]
    actual_investment_usd: Optional[float]
    writeoff_usd: Optional[float]
    promised_revenue_uplift_pct: Optional[float]
    actual_revenue_uplift_pct: Optional[float]
    promised_cost_reduction_pct: Optional[float]
    actual_cost_reduction_pct: Optional[float]
    failure_modes: List[str]
    success_factors: List[str]
    sources: List[Dict[str, Any]]
    tacit_notes: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformationCase":
        size = data["company_size"]
        t = data["transformation"]
        fin = data["financials"]
        return cls(
            id=data["id"],
            company=data["company"],
            industry=data["industry"],
            industry_tags=data.get("industry_tags", []),
            geography=data["geography"],
            revenue_usd=float(size["revenue_usd"]),
            employees=int(size["employees"]),
            revenue_bucket=size.get("revenue_bucket", "mid"),
            transformation_type=t["type"],
            transformation_subtype=t.get("subtype", ""),
            description=t.get("description", ""),
            start_year=int(t.get("start_year", 0)),
            planned_duration_months=int(t.get("planned_duration_months", 0) or 0),
            actual_duration_months=int(t.get("actual_duration_months", 0) or 0),
            status=t.get("status", "unknown"),
            vendor=t.get("vendor", ""),
            promised_investment_usd=fin.get("promised_investment_usd"),
            actual_investment_usd=fin.get("actual_investment_usd"),
            writeoff_usd=fin.get("writeoff_usd"),
            promised_revenue_uplift_pct=fin.get("promised_revenue_uplift_pct"),
            actual_revenue_uplift_pct=fin.get("actual_revenue_uplift_pct"),
            promised_cost_reduction_pct=fin.get("promised_cost_reduction_pct"),
            actual_cost_reduction_pct=fin.get("actual_cost_reduction_pct"),
            failure_modes=data.get("failure_modes", []),
            success_factors=data.get("success_factors", []),
            sources=data.get("sources", []),
            tacit_notes=data.get("tacit_notes", ""),
        )
