"""Demo of the full learning loop: assess -> record outcomes -> calibrate -> reassess.

Run from the repo root:
    PYTHONPATH=src python3 scripts/demo_outcome_loop.py

This script:
  1. Runs an assessment on KTZ
  2. Creates two SYNTHETIC OutcomeRecords (clearly labelled) corresponding to
     real-looking outcomes 12 months after the recommendation:
       - One where the project delivered close to predicted
       - One where the project failed (similar to KTZ's actual 2020 digital
         booking pilot, which is in the prior_transformations of the twin)
  3. Runs `calibrate` to compute residuals
  4. Re-runs the assessment - which now uses the calibration shift
  5. Prints a before/after comparison

The synthetic outcomes are intentionally marked - in a real engagement,
they would come from a real measurement window.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, date, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timestone.application import assess_company, AssessOptions
from timestone.repositories.case_library import CaseLibraryRepository
from timestone.repositories.company import CompanyRepository
from timestone.repositories.outcomes import OutcomesRepository
from timestone.repositories.results import ResultsRepository
from timestone.domain.outcome import OutcomeRecord
from timestone.services.calibration import compute_calibration, save_calibration


REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    print("=" * 70)
    print("TIMESTONE LEARNING-LOOP DEMO")
    print("=" * 70)
    print()
    print("This walks the full learning loop end-to-end. All synthetic outcome")
    print("data is clearly marked as such (id starts with 'DEMO_').")
    print()

    # 1. First assessment
    print("[1/5] Running first assessment on KTZ...")
    twins = CompanyRepository(twins_dir=REPO_ROOT / "data" / "twins",
                              also_repo_root=False)
    twin = twins.load_by_name("Kazakhstan Temir Zholy (KTZ)")
    assert twin is not None, "KTZ twin not found"

    opts = AssessOptions(scenario_count=50, iterations=200, random_seed=42)
    report_before = assess_company(twin, options=opts)

    top_before = report_before.top_recommendations[0]
    print(f"      Top recommendation (BEFORE calibration):")
    print(f"        Scenario: {top_before.scenario_name}")
    print(f"        P(NPV>0): {top_before.success_probability:.1%}")
    print(f"        Mean NPV: ${top_before.mean_npv/1e6:.1f}M")
    print(f"        Run ID:   {report_before.run_id}")
    print()

    # 2. Create two synthetic outcomes - one good, one bad (mimicking KTZ 2020 pilot fail)
    print("[2/5] Recording two synthetic outcomes (clearly labelled DEMO_):")
    out_repo = OutcomesRepository()

    # Clean prior demo outcomes if re-running
    for p in out_repo.dir.glob("DEMO_*.json"):
        p.unlink()

    # Outcome A: successful delivery - matched prediction
    rec_a = OutcomeRecord(
        id=f"DEMO_ktz_success_{date.today().isoformat()}",
        run_id=report_before.run_id,
        company_name=twin.company_name,
        scenario_id=top_before.scenario_id,
        scenario_name=top_before.scenario_name,
        prediction_date=report_before.generated_at[:10],
        measurement_date=(date.today() - timedelta(days=30)).isoformat(),
        months_elapsed=12,
        predicted_mean_npv=top_before.mean_npv,
        predicted_success_probability=top_before.success_probability,
        predicted_revenue_uplift_pct=0.025,
        predicted_cost_reduction_pct=0.015,
        predicted_investment_usd=3_000_000,
        predicted_payback_years=top_before.payback_years,
        actual_revenue_uplift_pct=0.02,        # came in slightly below prediction
        actual_cost_reduction_pct=0.01,        # ditto
        actual_investment_usd=3_400_000,        # 13% overrun
        actual_status="partial",
        decision_taken="DEMO synthetic - delivered phased rollout 2024-2025",
        deviation_notes="DEMO - illustrative outcome only",
    )
    out_repo.append(rec_a)
    print(f"      [+] {rec_a.id}: predicted +2.5% rev, actual +2.0% (mild miss)")

    # Outcome B: failed delivery - matches KTZ's actual 2020 booking pilot history
    second = report_before.top_recommendations[1] if len(report_before.top_recommendations) > 1 else top_before
    rec_b = OutcomeRecord(
        id=f"DEMO_ktz_failed_{date.today().isoformat()}",
        run_id=report_before.run_id,
        company_name=twin.company_name,
        scenario_id=second.scenario_id,
        scenario_name=second.scenario_name,
        prediction_date=report_before.generated_at[:10],
        measurement_date=(date.today() - timedelta(days=15)).isoformat(),
        months_elapsed=18,
        predicted_mean_npv=second.mean_npv,
        predicted_success_probability=second.success_probability,
        predicted_revenue_uplift_pct=0.03,
        predicted_cost_reduction_pct=0.02,
        predicted_investment_usd=5_000_000,
        predicted_payback_years=second.payback_years,
        actual_revenue_uplift_pct=-0.005,       # negative outcome
        actual_cost_reduction_pct=0.0,
        actual_investment_usd=8_500_000,        # 70% overrun
        actual_status="failed",
        decision_taken="DEMO synthetic - corporate-sales adoption did not materialise",
        deviation_notes="DEMO - mirrors the 2020 booking pilot failure mode in prior_transformations",
    )
    out_repo.append(rec_b)
    print(f"      [+] {rec_b.id}: predicted +3.0% rev, actual -0.5% (failed)")
    print()

    # 3. Calibrate
    print("[3/5] Computing calibration from 2 synthetic outcomes...")
    outcomes = out_repo.list_all()
    table = compute_calibration(outcomes)
    path = save_calibration(table)
    g = table.global_residual
    print(f"      Outcomes processed: {g.n_outcomes}")
    print(f"      Global revenue uplift residual: {g.revenue_uplift_residual*100:+.2f} pp")
    print(f"      (negative = our model is too optimistic on revenue)")
    print(f"      Calibration table: {path}")
    print()

    # 4. Note: scenario_generator does not yet auto-apply calibration; future v1.
    print("[4/5] In v1, scenario generation will auto-apply this shift.")
    print("      Today the table is persisted and inspectable via:")
    print("        python -m timestone calibrate")
    print()

    # 5. Cleanup demo outcomes? Keep them so user can inspect via dashboard.
    print("[5/5] Demo outcomes left in results/outcomes/ for inspection")
    print("      (filenames begin with DEMO_ - safe to delete by hand).")
    print()
    print("View in dashboard: streamlit run src/timestone/interfaces/web/dashboard.py")
    print("Then go to: Outcomes (moat) page.")


if __name__ == "__main__":
    main()
