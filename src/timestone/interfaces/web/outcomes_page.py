"""Streamlit page for tracking realised outcomes.

This is where the proprietary moat actually gets populated. Every time a real
client engagement reaches its measurement window (typically 12-24 months
after recommendation), the consultant comes here, picks the original run +
scenario, and records what actually happened.
"""
from __future__ import annotations

import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from ...domain.outcome import OutcomeRecord
from ...repositories.outcomes import OutcomesRepository
from ...repositories.results import ResultsRepository
from ...services.calibration import compute_calibration


def render():
    """Render the Outcomes tracking page."""
    st.header("Track outcomes (the moat)")
    st.markdown(
        "Every recorded outcome makes the next forecast more accurate. "
        "Use this page after a real engagement reaches its measurement window."
    )

    runs_repo = ResultsRepository()
    outcomes_repo = OutcomesRepository()
    runs = runs_repo.list_runs()

    tab_record, tab_list, tab_calib = st.tabs([
        "Record new outcome", "Recorded outcomes", "Calibration",
    ])

    # -----------------------------------------------------------------
    # Tab 1: Record a new outcome
    # -----------------------------------------------------------------
    with tab_record:
        if not runs:
            st.warning("No runs yet. Run `python -m timestone assess <company>` first.")
            return

        run_options = {p.name: p for p in runs}
        run_key = st.selectbox(
            "Run that produced the prediction:",
            list(run_options.keys()),
            help="Pick the assessment whose recommendation you are measuring.",
        )
        run_dir = run_options[run_key]
        data = runs_repo.load_run(run_dir)

        scenarios = (data.get("scenarios") or {}).get("scenarios", [])
        results = (data.get("simulation") or {}).get("results", [])
        if not scenarios or not results:
            st.error(f"Run {run_dir.name} is missing scenarios or simulation data.")
            return

        scen_lookup = {s["id"]: s for s in scenarios}
        res_lookup = {r["scenario_id"]: r for r in results}

        scen_choices = {
            f"#{s['id']} {s['name']}": s["id"]
            for s in scenarios if s["id"] in res_lookup
        }
        scen_label = st.selectbox("Scenario being measured:", list(scen_choices.keys()))
        scen_id = scen_choices[scen_label]
        scen = scen_lookup[scen_id]
        res = res_lookup[scen_id]

        st.markdown("**Original prediction:**")
        cols = st.columns(4)
        cols[0].metric("P(NPV>0) predicted", f"{res['success_probability']:.0%}")
        cols[1].metric("Mean NPV predicted", f"${res['mean_npv']/1e6:.1f}M")
        cols[2].metric("Mean ROI predicted", f"{res['mean_roi']:.1f}x")
        cols[3].metric("Median payback predicted", f"{res['payback_years_median']:.1f}y")

        ei = scen["expected_impact"]
        st.markdown("**Predicted parameter midpoints:**")
        cols = st.columns(3)
        cols[0].metric("Predicted revenue uplift", f"{ei['revenue_increase']*100:.2f}%")
        cols[1].metric("Predicted cost reduction", f"{ei['cost_reduction']*100:.2f}%")
        cols[2].metric("Predicted investment",
                       f"${scen['investment_required']/1e6:.1f}M")

        st.markdown("---")
        st.markdown("**Actual outcome:**")
        with st.form("outcome_form"):
            company = (data.get("report") or {}).get("company_name", "")
            outcome_id = st.text_input(
                "Outcome ID (unique - append-only):",
                value=f"{company.lower().replace(' ', '_')[:20]}_s{scen_id}_{date.today().isoformat()}",
            )
            measurement_date = st.date_input("Measurement date", value=date.today())
            months_elapsed = st.number_input("Months elapsed since recommendation",
                                              min_value=1, value=12)

            cols = st.columns(2)
            actual_rev = cols[0].number_input(
                "Actual revenue uplift % (e.g. 0.03 = +3%)",
                min_value=-1.0, max_value=2.0, value=ei["revenue_increase"], step=0.005,
                format="%.4f")
            actual_cost = cols[1].number_input(
                "Actual cost reduction % (e.g. 0.02 = -2% op costs)",
                min_value=-1.0, max_value=2.0, value=ei["cost_reduction"], step=0.005,
                format="%.4f")
            actual_invest = cols[0].number_input(
                "Actual investment ($)",
                min_value=0.0, value=float(scen["investment_required"]), step=1e5)
            actual_status = cols[1].selectbox(
                "Actual status",
                ["in_progress", "success", "partial", "failed", "abandoned"])

            decision_taken = st.text_input(
                "What the client actually did:",
                placeholder="e.g. Implemented full scope; phased rollout starting Q2 2024",
            )
            deviation_notes = st.text_area(
                "Why actual differs from predicted (optional):",
                placeholder="e.g. Vendor delays added 6 months; uplift held at lower bound of forecast",
            )

            submitted = st.form_submit_button("Record outcome")

        if submitted:
            try:
                rec = OutcomeRecord(
                    id=outcome_id,
                    run_id=run_dir.name,
                    company_name=company,
                    scenario_id=scen_id,
                    scenario_name=scen["name"],
                    prediction_date=(data.get("report") or {}).get("generated_at", "")[:10],
                    measurement_date=measurement_date.isoformat(),
                    months_elapsed=int(months_elapsed),
                    predicted_mean_npv=float(res["mean_npv"]),
                    predicted_success_probability=float(res["success_probability"]),
                    predicted_revenue_uplift_pct=float(ei["revenue_increase"]),
                    predicted_cost_reduction_pct=float(ei["cost_reduction"]),
                    predicted_investment_usd=float(scen["investment_required"]),
                    predicted_payback_years=float(res.get("payback_years_median", 0)),
                    actual_revenue_uplift_pct=float(actual_rev),
                    actual_cost_reduction_pct=float(actual_cost),
                    actual_investment_usd=float(actual_invest),
                    actual_status=actual_status,
                    decision_taken=decision_taken,
                    deviation_notes=deviation_notes,
                )
                outcomes_repo.append(rec)
                st.success(f"Outcome recorded: {outcome_id}")
                st.balloons()
            except FileExistsError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not record outcome: {exc!r}")

    # -----------------------------------------------------------------
    # Tab 2: List recorded outcomes
    # -----------------------------------------------------------------
    with tab_list:
        recs = outcomes_repo.list_all()
        if not recs:
            st.info("No outcomes recorded yet.")
        else:
            df = pd.DataFrame([
                {
                    "ID": r.id,
                    "Company": r.company_name,
                    "Scenario": r.scenario_name,
                    "Months": r.months_elapsed,
                    "Predicted rev %": r.predicted_revenue_uplift_pct * 100,
                    "Actual rev %": (r.actual_revenue_uplift_pct or 0) * 100,
                    "Residual rev %": ((r.actual_revenue_uplift_pct or 0)
                                        - r.predicted_revenue_uplift_pct) * 100,
                    "Status": r.actual_status,
                }
                for r in recs
            ])
            st.dataframe(df, use_container_width=True)

    # -----------------------------------------------------------------
    # Tab 3: Calibration
    # -----------------------------------------------------------------
    with tab_calib:
        recs = outcomes_repo.list_all()
        if not recs:
            st.info("Calibration needs at least one outcome.")
            return
        table = compute_calibration(recs)
        g = table.global_residual
        st.subheader("Global residual")
        cols = st.columns(3)
        cols[0].metric("Revenue uplift residual", f"{g.revenue_uplift_residual*100:+.2f} pp",
                        f"based on {g.n_outcomes} outcomes")
        cols[1].metric("Cost reduction residual", f"{g.cost_reduction_residual*100:+.2f} pp")
        cols[2].metric("Success rate residual", f"{g.success_rate_residual*100:+.2f} pp")
        st.caption(
            "Negative residual = model is too optimistic for this metric. "
            "Future simulations will shift the empirical prior by this amount "
            "(damped by sample size)."
        )

        if table.by_bucket:
            st.subheader("Per-bucket residuals")
            rows = []
            for (ind, tt), e in sorted(table.by_bucket.items()):
                rows.append({
                    "Industry": ind, "Type": tt, "n": e.n_outcomes,
                    "Revenue shift (pp)": f"{e.revenue_uplift_residual*100:+.2f}",
                    "Cost shift (pp)": f"{e.cost_reduction_residual*100:+.2f}",
                    "Success shift (pp)": f"{e.success_rate_residual*100:+.2f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
