"""Services - pure business logic operating on domain objects."""
from .knowledge_retrieval import CaseLibrary, revenue_to_bucket
from .scenario_generation import ScenarioGenerator
from .monte_carlo import MonteCarloSimulator
from .sensitivity import sensitivity_analysis, SensitivityRow
from .recommendation import build_report
from .calibration import (
    compute_calibration, save_calibration, load_calibration,
    apply_calibration_to_prior, CalibrationTable, CalibrationEntry,
)

__all__ = [
    "CaseLibrary", "revenue_to_bucket",
    "ScenarioGenerator",
    "MonteCarloSimulator",
    "sensitivity_analysis", "SensitivityRow",
    "build_report",
    "compute_calibration", "save_calibration", "load_calibration",
    "apply_calibration_to_prior", "CalibrationTable", "CalibrationEntry",
]
