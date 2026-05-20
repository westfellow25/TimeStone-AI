"""assess_company: the single entry point that drives the full pipeline.

Loads the case library, runs scenario generation, simulates, builds the
report, and persists everything under results/runs/{run_id}/.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..domain.company import Company
from ..domain.report import AssessmentReport
from ..domain.simulation import SimulationConfig
from ..repositories.case_library import CaseLibraryRepository
from ..repositories.results import ResultsRepository
from ..services.knowledge_retrieval import CaseLibrary
from ..services.scenario_generation import ScenarioGenerator
from ..services.monte_carlo import MonteCarloSimulator
from ..services.recommendation import build_report
from ..interfaces.reports import generate_pdf, ReportContext


@dataclass
class AssessOptions:
    scenario_count: int = 1000
    iterations: int = 1000
    discount_rate: float = 0.12
    horizon_years: int = 5
    random_seed: int = 42
    use_case_library: bool = True


def assess_company(twin: Company,
                   options: Optional[AssessOptions] = None,
                   case_library_repo: Optional[CaseLibraryRepository] = None,
                   results_repo: Optional[ResultsRepository] = None,
                   ) -> AssessmentReport:
    """Run the full TimeStone pipeline for a single company.

    Steps:
      1. Load case library
      2. Generate `options.scenario_count` scenarios using retrieved priors
      3. Simulate each with Monte Carlo
      4. Build the assessment report (top recommendations)
      5. Persist all artefacts under results/runs/{run_id}/
      6. Return the AssessmentReport
    """
    opts = options or AssessOptions()
    case_repo = case_library_repo or CaseLibraryRepository()
    results_repo = results_repo or ResultsRepository()

    # 1. Load corpus
    cases = case_repo.load_all()
    library = CaseLibrary(cases) if (cases and opts.use_case_library) else None

    # 2. Generate scenarios
    # Aggregate failure modes from prior transformations the company already
    # botched. This is one of the most valuable retrieval signals: "this
    # company has already failed at X, weight cases that also failed at X".
    prior_failure_modes = []
    for pt in twin.metrics.prior_transformations:
        if pt.outcome in ("failed", "partial"):
            # Best-effort tokenisation of the prior name/description into
            # failure-mode keywords - users may also list explicit ones later.
            for token in (pt.name + " " + pt.description).lower().split():
                if len(token) > 3:
                    prior_failure_modes.append(token)
    segment_keywords = [s.name.split()[0].lower() for s in twin.metrics.segments]

    profile = {
        "industry": twin.metrics.industry,
        "industry_tags": twin.metrics.industry_tags,
        "revenue_usd": twin.metrics.revenue,
        "geography": twin.metrics.geography,
        "prior_failure_modes": prior_failure_modes,
        "segment_keywords": segment_keywords,
    }
    generator = ScenarioGenerator(
        company_name=twin.company_name,
        industry=twin.metrics.industry,
        case_library=library,
        company_profile=profile,
    )
    scenarios = generator.generate(count=opts.scenario_count)

    # 3. Simulate
    config = SimulationConfig(
        iterations=opts.iterations,
        horizon_years=opts.horizon_years,
        discount_rate=opts.discount_rate,
    )
    simulator = MonteCarloSimulator(config=config, random_seed=opts.random_seed)
    results = simulator.simulate_all(
        scenarios, twin.metrics.revenue, twin.metrics.operating_costs)

    # 4. Build report
    run_dir = results_repo.new_run(twin.company_name)
    run_id = run_dir.name
    report = build_report(
        run_id=run_id, company_name=twin.company_name,
        scenarios=scenarios, results=results,
        case_library_size=len(cases),
        config_summary={
            "iterations": opts.iterations, "horizon_years": opts.horizon_years,
            "discount_rate": opts.discount_rate, "random_seed": opts.random_seed,
            "uses_case_library": library is not None,
        },
    )

    # 5. Persist artefacts
    results_repo.save_scenarios(run_dir, {
        "company": twin.company_name,
        "industry": twin.metrics.industry,
        "total_scenarios": len(scenarios),
        "uses_case_library": library is not None,
        "scenarios": [s.to_dict() for s in scenarios],
    })
    results_repo.save_simulation(run_dir, {
        "simulation_parameters": {
            "iterations": opts.iterations, "horizon_years": opts.horizon_years,
            "discount_rate": opts.discount_rate, "random_seed": opts.random_seed,
            "total_scenarios": len(scenarios),
        },
        "results": [r.to_dict() for r in results],
    })
    results_repo.save_report(run_dir, report.to_dict())

    # 6. Generate executive PDF (best-effort; do not fail the pipeline if it crashes)
    try:
        scenarios_payload = {
            "company": twin.company_name,
            "industry": twin.metrics.industry,
            "total_scenarios": len(scenarios),
            "uses_case_library": library is not None,
            "scenarios": [s.to_dict() for s in scenarios],
        }
        simulation_payload = {
            "simulation_parameters": {
                "iterations": opts.iterations, "horizon_years": opts.horizon_years,
                "discount_rate": opts.discount_rate, "random_seed": opts.random_seed,
                "total_scenarios": len(scenarios),
            },
            "results": [r.to_dict() for r in results],
        }
        cases_by_id = {}
        for case in cases:
            cases_by_id[case.id] = {
                "company": case.company, "industry": case.industry,
                "geography": case.geography,
                "transformation": {
                    "status": case.status, "description": case.description,
                    "start_year": case.start_year,
                },
                "financials": {
                    "actual_revenue_uplift_pct": case.actual_revenue_uplift_pct,
                    "actual_cost_reduction_pct": case.actual_cost_reduction_pct,
                    "writeoff_usd": case.writeoff_usd,
                },
                "sources": case.sources,
                "tacit_notes": case.tacit_notes,
            }
        ctx = ReportContext(
            report=report,
            scenarios_payload=scenarios_payload,
            simulation_payload=simulation_payload,
            cases_by_id=cases_by_id,
        )
        generate_pdf(ctx, run_dir / "report.pdf")
    except Exception as exc:  # noqa: BLE001 - PDF failure must not break the run
        import logging
        logging.getLogger(__name__).warning(
            "PDF generation failed (run artefacts still saved): %r", exc)

    return report
