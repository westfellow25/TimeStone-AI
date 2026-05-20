"""Domain layer - pure data classes, no I/O, no external dependencies."""
from .company import Company, CompanyMetrics, BusinessSegment, Competitor, PriorTransformation
from .case import TransformationCase, CaseQuery
from .scenario import Scenario, TransformationType
from .simulation import SimulationConfig, SimulationResult
from .outcome import OutcomeRecord
from .report import Recommendation, AssessmentReport

__all__ = [
    "Company", "CompanyMetrics", "BusinessSegment", "Competitor", "PriorTransformation",
    "TransformationCase", "CaseQuery",
    "Scenario", "TransformationType",
    "SimulationConfig", "SimulationResult",
    "OutcomeRecord",
    "Recommendation", "AssessmentReport",
]
