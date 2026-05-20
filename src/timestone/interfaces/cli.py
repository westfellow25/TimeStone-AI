"""TimeStone CLI.

Usage:
    python -m timestone assess <company_name>
    python -m timestone list-companies
    python -m timestone list-runs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..application import assess_company, AssessOptions
from ..repositories.company import CompanyRepository
from ..repositories.results import ResultsRepository


def cmd_assess(args: argparse.Namespace) -> int:
    company_repo = CompanyRepository()
    twin = company_repo.load_by_name(args.company)
    if twin is None:
        print(f"Company not found: {args.company}", file=sys.stderr)
        print("Available companies:", file=sys.stderr)
        for c in company_repo.list_all():
            print(f"  - {c.company_name}", file=sys.stderr)
        return 1
    opts = AssessOptions(
        scenario_count=args.scenarios,
        iterations=args.iterations,
        discount_rate=args.discount_rate,
        random_seed=args.seed,
    )
    report = assess_company(twin, options=opts)
    print(f"Run ID: {report.run_id}")
    print(f"Company: {report.company_name}")
    print(f"Scenarios: {report.total_scenarios}")
    print(f"Failure rate (% scenarios with P<50%): {report.failure_rate_among_scenarios:.1%}")
    print(f"Case library size: {report.case_library_size}")
    print()
    print("Top recommendations:")
    for rec in report.top_recommendations:
        print(f"  #{rec.rank} {rec.scenario_name}")
        print(f"     P(NPV>0): {rec.success_probability:.1%}")
        print(f"     Mean NPV: ${rec.mean_npv:,.0f}")
        print(f"     Mean ROI: {rec.mean_roi:.1f}x")
        print(f"     Payback: {rec.payback_years:.1f}y")
        print(f"     Headline: {rec.headline}")
        if rec.based_on_cases:
            print(f"     Based on cases: {', '.join(rec.based_on_cases[:3])}")
        print()
    return 0


def cmd_list_companies(args: argparse.Namespace) -> int:
    repo = CompanyRepository()
    for c in repo.list_all():
        print(f"  {c.company_name}  ({c.metrics.industry}, ${c.metrics.revenue/1e9:.1f}B)")
    return 0


def cmd_list_runs(args: argparse.Namespace) -> int:
    repo = ResultsRepository()
    runs = repo.list_runs()
    if not runs:
        print("No runs yet. Use `python -m timestone assess <company>`.")
        return 0
    for p in runs[:20]:
        print(f"  {p.name}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Regenerate the executive PDF for an existing run."""
    import json
    from ..application.assess_company import AssessOptions  # noqa: F401
    from ..domain.report import AssessmentReport, Recommendation
    from ..repositories.results import ResultsRepository
    from ..repositories.case_library import CaseLibraryRepository
    from .reports import generate_pdf, ReportContext

    runs_repo = ResultsRepository()
    matching = [p for p in runs_repo.list_runs()
                if args.run_id in p.name]
    if not matching:
        print(f"No run matching: {args.run_id}", file=sys.stderr)
        return 1
    run_dir = matching[0]
    data = runs_repo.load_run(run_dir)
    rep_data = data.get("report")
    scen_data = data.get("scenarios")
    sim_data = data.get("simulation")
    if not (rep_data and scen_data and sim_data):
        print(f"Run is missing artifacts: {run_dir.name}", file=sys.stderr)
        return 1

    # Rebuild AssessmentReport
    recs = [Recommendation(**r) for r in rep_data["top_recommendations"]]
    report = AssessmentReport(
        run_id=rep_data["run_id"], company_name=rep_data["company_name"],
        generated_at=rep_data["generated_at"],
        config_summary=rep_data["config_summary"],
        top_recommendations=recs,
        total_scenarios=rep_data["total_scenarios"],
        failure_rate_among_scenarios=rep_data["failure_rate_among_scenarios"],
        case_library_size=rep_data["case_library_size"],
        notes=rep_data.get("notes", ""),
    )

    # Load cases for sources appendix
    cases = CaseLibraryRepository().load_all()
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
            "sources": case.sources, "tacit_notes": case.tacit_notes,
        }

    ctx = ReportContext(
        report=report, scenarios_payload=scen_data,
        simulation_payload=sim_data, cases_by_id=cases_by_id,
    )
    out = generate_pdf(ctx, run_dir / "report.pdf")
    print(f"PDF written to: {out}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Compute calibration shifts from accumulated outcomes; save table."""
    from ..repositories.outcomes import OutcomesRepository
    from ..services.calibration import compute_calibration, save_calibration

    outcomes = OutcomesRepository().list_all()
    table = compute_calibration(outcomes)
    path = save_calibration(table)

    if not outcomes:
        print("No outcomes recorded yet. Nothing to calibrate.")
        print(f"Empty calibration table written to: {path}")
        return 0

    g = table.global_residual
    print(f"Outcomes processed: {len(outcomes)}")
    print(f"Specific buckets calibrated: {len(table.by_bucket)}")
    print()
    print("Global residual (mean across all outcomes):")
    print(f"  Revenue uplift: {g.revenue_uplift_residual:+.3f}")
    print(f"  Cost reduction: {g.cost_reduction_residual:+.3f}")
    print(f"  Success rate:   {g.success_rate_residual:+.3f}")
    if table.by_bucket:
        print()
        print("Per-bucket residuals:")
        for (ind, tt), entry in sorted(table.by_bucket.items()):
            print(f"  {ind} / {tt}: n={entry.n_outcomes}, "
                  f"rev shift {entry.revenue_uplift_residual:+.3f}")
    print()
    print(f"Calibration table written to: {path}")
    return 0


def cmd_show_outcomes(args: argparse.Namespace) -> int:
    """List recorded outcomes."""
    from ..repositories.outcomes import OutcomesRepository
    outcomes = OutcomesRepository().list_all()
    if not outcomes:
        print("No outcomes recorded yet.")
        return 0
    for o in outcomes:
        print(f"  {o.id}  ({o.company_name} / scenario #{o.scenario_id})")
        print(f"     predicted P(NPV>0)={o.predicted_success_probability:.0%}, "
              f"actual_status={o.actual_status}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="timestone", description="TimeStone AI CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("assess", help="Run the full assessment pipeline for a company.")
    a.add_argument("company", help="Company name (case-insensitive match)")
    a.add_argument("--scenarios", type=int, default=1000)
    a.add_argument("--iterations", type=int, default=1000)
    a.add_argument("--discount-rate", type=float, default=0.12)
    a.add_argument("--seed", type=int, default=42)
    a.set_defaults(func=cmd_assess)

    lc = sub.add_parser("list-companies", help="List available digital twins.")
    lc.set_defaults(func=cmd_list_companies)

    lr = sub.add_parser("list-runs", help="List recent assessment runs.")
    lr.set_defaults(func=cmd_list_runs)

    r = sub.add_parser("report", help="Regenerate the PDF report for an existing run.")
    r.add_argument("run_id", help="Run id or partial match (e.g. 'ktz').")
    r.set_defaults(func=cmd_report)

    cal = sub.add_parser("calibrate",
        help="Process recorded outcomes and produce a calibration table.")
    cal.set_defaults(func=cmd_calibrate)

    out = sub.add_parser("outcomes",
        help="List recorded outcomes (the proprietary moat).")
    out.set_defaults(func=cmd_show_outcomes)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
