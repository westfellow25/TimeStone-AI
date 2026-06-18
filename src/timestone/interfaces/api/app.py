"""FastAPI app — TimeStone REST API.

This module imports FastAPI lazily so the rest of the codebase keeps
working when FastAPI is not installed. Install with:

    pip install fastapi uvicorn

Run:

    uvicorn timestone.interfaces.api.app:app --reload --port 8088
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    HAS_FASTAPI = True
except ImportError:  # pragma: no cover — graceful degrade
    HAS_FASTAPI = False
    FastAPI = object  # type: ignore[misc,assignment]

from ...application.assess_company import AssessOptions, assess_company
from ...infrastructure.paths import DATA_DIR, REPO_ROOT
from ...repositories.case_library import CaseLibraryRepository
from ...repositories.company import CompanyRepository
from . import schemas
from .auth import QuotaTracker, Tenant, TenantRegistry
from .tenant_runs import RunSummary, TenantRunLog

# Optional synthetic-data router; import-safe if FastAPI is missing.
try:
    from .synthesis import router as _synthesis_router
except Exception:  # noqa: BLE001
    _synthesis_router = None

VERSION = "0.6.0"

_PREDICTIONS_PATH = REPO_ROOT / "predictions" / "2026-05-20.json"
_TENANTS_PATH = DATA_DIR / "tenants.json"


def _load_predictions() -> list[dict]:
    if not _PREDICTIONS_PATH.exists():
        return []
    try:
        data = json.loads(_PREDICTIONS_PATH.read_text())
        return data.get("predictions", [])
    except json.JSONDecodeError:
        return []


def create_app() -> "FastAPI":  # noqa: F821
    """Factory so tests can spin up isolated app instances."""
    if not HAS_FASTAPI:
        raise RuntimeError(
            "FastAPI is not installed. Run: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="TimeStone AI",
        version=VERSION,
        description="REST API for the TimeStone transformation simulator.",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount privacy-preserving synthetic-data endpoints:
    #   POST /synthesize/records, POST /synthesize/csv, GET /synthesize/health.
    if _synthesis_router is not None:
        app.include_router(_synthesis_router)

    registry = TenantRegistry(store_path=_TENANTS_PATH)
    quotas = QuotaTracker()
    run_log = TenantRunLog()
    case_repo = CaseLibraryRepository()
    company_repo = CompanyRepository()

    # ----- Auth dependency -----

    def require_tenant(x_api_key: Optional[str] = Header(default=None)) -> Tenant:
        tenant = registry.resolve(x_api_key)
        if tenant is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid X-API-Key header. Use 'demo-key' for local dev.",
            )
        return tenant

    def enforce_quota(tenant: Tenant) -> None:
        used = quotas.used(tenant.tenant_id)
        if used >= tenant.quota_assess_per_day:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Daily assess quota of {tenant.quota_assess_per_day} reached for tenant {tenant.tenant_id}.",
            )

    # ----- Health -----

    @app.get("/health", response_model=schemas.HealthOut, tags=["meta"])
    def health() -> schemas.HealthOut:
        cases = case_repo.load_all()
        twins = company_repo.list_all()
        return schemas.HealthOut(
            status="ok",
            version=VERSION,
            case_library_size=len(cases),
            twins_loaded=len(twins),
        )

    # ----- Auth -----

    @app.get("/auth/whoami", response_model=schemas.TenantInfo, tags=["auth"])
    def whoami(tenant: Tenant = Depends(require_tenant)) -> schemas.TenantInfo:
        return schemas.TenantInfo(
            tenant_id=tenant.tenant_id,
            name=tenant.name,
            plan=tenant.plan,
            quota_assess_per_day=tenant.quota_assess_per_day,
            used_today=quotas.used(tenant.tenant_id),
        )

    # ----- Library -----

    @app.get("/library", response_model=schemas.LibraryResponse, tags=["library"])
    def library(industry: Optional[str] = None,
                status_filter: Optional[str] = None,
                limit: int = 100,
                tenant: Tenant = Depends(require_tenant)) -> schemas.LibraryResponse:
        cases = case_repo.load_all()
        if industry:
            cases = [c for c in cases if c.industry.lower() == industry.lower()]
        if status_filter:
            cases = [c for c in cases if c.status.lower() == status_filter.lower()]
        out = []
        for c in cases[:limit]:
            out.append(schemas.CaseSummary(
                case_id=c.id,
                company=c.company,
                industry=c.industry,
                year=c.start_year,
                status=c.status,
                transformation_type=getattr(c, "transformation_type", "") or "",
                geography=getattr(c, "geography", "") or "",
                promised_investment_usd=getattr(c, "promised_investment_usd", None),
                actual_investment_usd=getattr(c, "actual_investment_usd", None),
                writeoff_usd=getattr(c, "writeoff_usd", None),
                failure_modes=list(getattr(c, "failure_modes", []) or []),
                sources=list(getattr(c, "sources", []) or []),
            ))
        return schemas.LibraryResponse(total=len(cases), cases=out)

    @app.get("/library/{case_id}", tags=["library"])
    def library_one(case_id: str,
                    tenant: Tenant = Depends(require_tenant)):
        cases = case_repo.load_all()
        for c in cases:
            if c.id == case_id:
                return c.__dict__
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    # ----- Companies -----

    @app.get("/companies", response_model=schemas.CompanyListResponse, tags=["companies"])
    def companies(tenant: Tenant = Depends(require_tenant)) -> schemas.CompanyListResponse:
        twins = company_repo.list_all()
        out = []
        for t in twins:
            m = getattr(t, "metrics", None)
            out.append(schemas.CompanyOut(
                name=t.company_name,
                industry=getattr(m, "industry", "") if m else "",
                region=getattr(m, "geography", None) if m else None,
                annual_revenue_usd=getattr(m, "annual_revenue_usd", None) if m else None,
                annual_op_costs_usd=getattr(m, "annual_operating_costs_usd", None) if m else None,
                employees=getattr(m, "employees", None) if m else None,
                prior_transformation_count=len(getattr(m, "prior_transformations", []) or []) if m else 0,
            ))
        return schemas.CompanyListResponse(total=len(out), companies=out)

    # ----- Assess -----

    @app.post("/assess", response_model=schemas.AssessResponse, tags=["assess"])
    def assess(req: schemas.AssessRequest,
               tenant: Tenant = Depends(require_tenant)) -> schemas.AssessResponse:
        enforce_quota(tenant)
        twin = company_repo.load_by_name(req.company_name)
        if twin is None:
            raise HTTPException(status_code=404, detail=f"Company '{req.company_name}' not found. Use /companies for the list.")
        opts = AssessOptions(
            scenario_count=req.options.scenario_count,
            iterations=req.options.iterations,
            discount_rate=req.options.discount_rate,
            horizon_years=req.options.horizon_years,
            random_seed=req.options.random_seed,
            use_case_library=req.options.use_case_library,
        )
        report = assess_company(twin, options=opts)
        quotas.increment(tenant.tenant_id)
        # Append to per-tenant run log (multi-tenant scoping)
        if report.top_recommendations:
            top = report.top_recommendations[0]
            run_log.append(RunSummary(
                run_id=report.run_id,
                tenant_id=tenant.tenant_id,
                company_name=report.company_name,
                generated_at=report.generated_at,
                top_scenario=top.scenario_name,
                top_success_p=top.success_probability,
                top_npv_musd=top.mean_npv / 1e6,
                headline=top.headline,
            ))
        recs = [
            schemas.RecommendationOut(
                rank=r.rank,
                scenario_id=r.scenario_id,
                scenario_name=r.scenario_name,
                headline=r.headline,
                success_probability=r.success_probability,
                mean_npv=r.mean_npv,
                mean_roi=r.mean_roi,
                payback_years=r.payback_years,
                based_on_cases=r.based_on_cases,
                empirical_failure_rate=r.empirical_failure_rate,
                risk_level=r.risk_level,
                description=r.description,
            )
            for r in report.top_recommendations
        ]
        return schemas.AssessResponse(
            run_id=report.run_id,
            company_name=report.company_name,
            generated_at=report.generated_at,
            total_scenarios=report.total_scenarios,
            failure_rate_among_scenarios=report.failure_rate_among_scenarios,
            case_library_size=report.case_library_size,
            top_recommendations=recs,
        )

    # ----- Pre-registered predictions -----

    @app.get("/predict", tags=["predict"])
    def predictions(tenant: Tenant = Depends(require_tenant)) -> dict:
        preds = _load_predictions()
        return {"count": len(preds), "committed_at": "2026-05-20", "predictions": preds}

    @app.get("/predict/{prediction_id}", tags=["predict"])
    def prediction_one(prediction_id: str,
                       tenant: Tenant = Depends(require_tenant)) -> dict:
        preds = _load_predictions()
        for p in preds:
            if p.get("id") == prediction_id:
                return p
        raise HTTPException(status_code=404, detail=f"Prediction '{prediction_id}' not found")

    # ----- Multi-tenant customer dashboard -----

    @app.get("/me/runs", tags=["tenant"])
    def my_runs(limit: int = 20,
                tenant: Tenant = Depends(require_tenant)) -> dict:
        runs = run_log.list(tenant.tenant_id, limit=limit)
        return {
            "tenant_id": tenant.tenant_id,
            "tenant_name": tenant.name,
            "plan": tenant.plan,
            "count": len(runs),
            "runs": runs,
        }

    @app.get("/me/dashboard", response_class=HTMLResponse, tags=["tenant"])
    def my_dashboard(tenant: Tenant = Depends(require_tenant)) -> str:
        runs = run_log.list(tenant.tenant_id, limit=10)
        rows_html = "".join(
            (
                f"<tr>"
                f"<td><code>{r['run_id'][:10]}…</code></td>"
                f"<td>{r['company_name']}</td>"
                f"<td>{r['top_scenario'][:60]}</td>"
                f"<td>{r['top_success_p']*100:.0f}%</td>"
                f"<td>${r['top_npv_musd']:+.1f}M</td>"
                f"<td>{r['headline']}</td>"
                f"<td><small>{r['generated_at']}</small></td>"
                f"</tr>"
            )
            for r in runs
        )
        if not rows_html:
            rows_html = (
                "<tr><td colspan='7' style='text-align:center; color:#888; padding:24px'>"
                "No assessments yet. POST to /assess to populate this dashboard."
                "</td></tr>"
            )
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TimeStone · {tenant.name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
          background: #060912; color: #E5E7EB; padding: 32px; }}
  h1 {{ margin: 0 0 6px; font-weight: 700; }}
  .sub {{ color: #94A3B8; font-size: 13px; margin-bottom: 24px; }}
  .meta {{ display: flex; gap: 18px; margin-bottom: 24px; font-size: 13px; }}
  .meta span {{ background: rgba(99,102,241,0.10); padding: 6px 12px; border-radius: 6px; color: #C7D2FE; }}
  table {{ width: 100%; border-collapse: collapse; background: rgba(255,255,255,0.03); border-radius: 10px; overflow: hidden; }}
  th, td {{ padding: 11px 14px; text-align: left; font-size: 13px; border-bottom: 1px solid rgba(255,255,255,0.06); }}
  th {{ background: rgba(99,102,241,0.10); color: #C7D2FE; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }}
  code {{ font-family: "JetBrains Mono", monospace; color: #22D3EE; }}
</style></head>
<body>
  <h1>{tenant.name}</h1>
  <div class="sub">TimeStone AI · customer dashboard (v0.6 preview)</div>
  <div class="meta">
    <span>Plan: <strong>{tenant.plan}</strong></span>
    <span>Today: <strong>{quotas.used(tenant.tenant_id)} / {tenant.quota_assess_per_day}</strong> assessments</span>
    <span>Tenant ID: <code>{tenant.tenant_id}</code></span>
  </div>
  <table>
    <thead><tr>
      <th>Run</th><th>Company</th><th>Top scenario</th><th>P(NPV&gt;0)</th>
      <th>Mean NPV</th><th>Verdict</th><th>When</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</body></html>"""

    return app


# Module-level app for the conventional `uvicorn module:app` invocation.
# If FastAPI is missing we still expose `app = None` so imports don't blow up.
app = create_app() if HAS_FASTAPI else None
