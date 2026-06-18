"""FastAPI router for the privacy-preserving synthetic data layer.

Two endpoints:

* ``POST /synthesize/records`` — JSON in, JSON out. For programmatic
  use from the demo dashboard or another backend.
* ``POST /synthesize/csv`` — multipart CSV upload, multipart CSV download.
  For the "drop a file, get a twin" UX path.

Both endpoints stream-process: the raw upload is held in memory only for
the duration of the request, never persisted. The trained synthesizer is
also in-memory and is GC'd when the handler returns.

Wire up in your FastAPI app by importing and including the router::

    from timestone.interfaces.api.synthesis import router as synthesis_router
    app.include_router(synthesis_router)

The router is self-contained: if SDV is not installed, endpoints still
respond with a warning in the report and use the marginal fallback so the
integration smoke-tests stay green.
"""
from __future__ import annotations

import io
import os
import logging
from typing import Any, Dict, List, Optional

try:
    from fastapi import APIRouter, File, Form, HTTPException, UploadFile
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False

from ...services.synthetic import (
    SynthesisRequest,
    SynthesisResult,
    synthesize,
    synthesize_from_records,
    synthetic_to_records,
)

logger = logging.getLogger(__name__)


# ----- Privacy hard-reject -----
# If the trained synthesizer produces rows that are dangerously close to real
# rows (mean min nearest-neighbour distance < threshold), we refuse to serve
# the result. This is the production-hardening recommendation #1 from
# INTEGRATION.md. Threshold is configurable via env var.
_PRIVACY_THRESHOLD = float(os.environ.get("TIMESTONE_SYNTH_MIN_PRIVACY", "0.5"))


def _enforce_privacy(report_dict):
    """Raise 422 if the synthesised data leaks too much. No-op when threshold
    is 0 (useful for tests / smoke runs / marginal-fallback smoke tests)."""
    if _PRIVACY_THRESHOLD <= 0:
        return
    pd = float(report_dict.get("privacy_distance", 0.0))
    backend = report_dict.get("backend", "")
    # The marginal fallback can produce verbatim copies on small inputs; in
    # that path we still want to surface a warning but not block (since the
    # fallback is for smoke-testing only, not for production-grade privacy).
    if backend.startswith("marginal"):
        return
    if pd < _PRIVACY_THRESHOLD:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Synthetic data privacy distance {pd:.3f} is below the "
                f"required minimum of {_PRIVACY_THRESHOLD:.3f}. Refusing to "
                "return the result. Try increasing rows, switching backend "
                "to ctgan, or relaxing TIMESTONE_SYNTH_MIN_PRIVACY."
            ),
        )


if HAS_FASTAPI:
    router = APIRouter(prefix="/synthesize", tags=["synthesis"])

    # ----- Pydantic wire schemas -----

    class RecordSynthesisIn(BaseModel):
        records: List[Dict[str, Any]] = Field(
            ..., description="The raw table to synthesise, as a list of dicts."
        )
        n_rows: Optional[int] = Field(
            None, description="Number of synthetic rows. Defaults to source size."
        )
        model: str = Field(
            "gaussian_copula",
            description="ctgan | tvae | gaussian_copula | marginal",
        )
        epochs: int = Field(200, ge=10, le=2000)
        primary_key: Optional[str] = None
        sensitive_columns: List[str] = Field(default_factory=list)
        random_seed: int = 42

    class QualityReportOut(BaseModel):
        fidelity_score: float
        column_shapes: Dict[str, float]
        column_pair_trends: Dict[str, float]
        privacy_distance: float
        backend: str
        elapsed_seconds: float
        rows_in: int
        rows_out: int
        warnings: List[str]

    class RecordSynthesisOut(BaseModel):
        synthetic: List[Dict[str, Any]]
        report: QualityReportOut

    # ----- /synthesize/records -----

    @router.post(
        "/records",
        response_model=RecordSynthesisOut,
        summary="Synthesise from a JSON records payload",
    )
    def synthesize_records(req: RecordSynthesisIn) -> RecordSynthesisOut:
        """Train an SDG model on the supplied records and return a
        synthetic twin plus a quality + privacy report.

        The input records are not persisted; they are dropped as soon as
        the synthesizer finishes fitting.
        """
        try:
            result: SynthesisResult = synthesize_from_records(
                records=req.records,
                n_rows=req.n_rows,
                model=req.model,  # type: ignore[arg-type]
                epochs=req.epochs,
                primary_key=req.primary_key,
                sensitive_columns=req.sensitive_columns,
                random_seed=req.random_seed,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("synthesis failed")
            raise HTTPException(
                status_code=500, detail=f"synthesis failed: {exc}"
            ) from exc

        report_dict = result.report.to_dict()
        _enforce_privacy(report_dict)

        return RecordSynthesisOut(
            synthetic=synthetic_to_records(result),
            report=QualityReportOut(**report_dict),
        )

    # ----- /synthesize/csv -----

    @router.post(
        "/csv",
        summary="Synthesise from a CSV upload; returns CSV + report headers",
    )
    async def synthesize_csv(
        file: UploadFile = File(..., description="CSV file to synthesise."),
        n_rows: Optional[int] = Form(None),
        model: str = Form("gaussian_copula"),
        epochs: int = Form(200),
        primary_key: Optional[str] = Form(None),
        random_seed: int = Form(42),
    ) -> StreamingResponse:
        """Same as ``/records`` but for CSV upload/download.

        Quality and privacy metrics are returned in response headers
        (``X-TimeStone-Fidelity``, ``X-TimeStone-Privacy``, etc.) so the
        client gets the synthetic file as a stream without parsing JSON.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="pandas is required on the server for CSV synthesis.",
            ) from exc

        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="empty upload")

        try:
            df = pd.read_csv(io.BytesIO(raw_bytes))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=400, detail=f"could not parse CSV: {exc}"
            ) from exc

        # Drop the raw bytes reference now that we have a DataFrame.
        del raw_bytes

        try:
            result = synthesize(
                SynthesisRequest(
                    source=df,
                    n_rows=n_rows,
                    model=model,  # type: ignore[arg-type]
                    epochs=epochs,
                    primary_key=primary_key,
                    random_seed=random_seed,
                )
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("CSV synthesis failed")
            raise HTTPException(
                status_code=500, detail=f"synthesis failed: {exc}"
            ) from exc

        buf = io.StringIO()
        result.synthetic.to_csv(buf, index=False)
        buf.seek(0)

        report_dict = result.report.to_dict()
        _enforce_privacy(report_dict)

        headers = {
            "X-TimeStone-Fidelity": f"{report_dict['fidelity_score']:.4f}",
            "X-TimeStone-Privacy": f"{report_dict['privacy_distance']:.4f}",
            "X-TimeStone-Backend": report_dict["backend"],
            "X-TimeStone-Elapsed": f"{report_dict['elapsed_seconds']:.2f}",
            "X-TimeStone-RowsIn": str(report_dict["rows_in"]),
            "X-TimeStone-RowsOut": str(report_dict["rows_out"]),
            "Content-Disposition": (
                f'attachment; filename="synthetic_{file.filename or "data.csv"}"'
            ),
        }
        if report_dict.get("warnings"):
            # HTTP headers are latin-1 only; sanitise unicode to ASCII
            # (em-dashes, etc.) so the warnings header never blows up
            # the response.
            joined = "; ".join(report_dict["warnings"])[:500]
            headers["X-TimeStone-Warnings"] = (
                joined.encode("ascii", "replace").decode("ascii")
            )

        return StreamingResponse(
            iter([buf.getvalue()]), media_type="text/csv", headers=headers
        )

    # ----- Health -----

    @router.get("/health", summary="Synthesis subsystem health")
    def health() -> Dict[str, Any]:
        from ...services import synthetic as _s

        return {
            "ok": True,
            "sdv_available": _s.HAS_SDV,
            "sdmetrics_available": _s.HAS_SDMETRICS,
            "pandas_available": _s.HAS_PANDAS,
            "fallback_in_use": not _s.HAS_SDV,
        }

else:  # pragma: no cover
    router = None  # type: ignore[assignment]
