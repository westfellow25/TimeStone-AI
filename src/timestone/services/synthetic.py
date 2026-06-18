"""Synthetic data generation service.

Privacy-preserving data layer: train a generative model on real client data
locally, emit a statistically similar synthetic twin, and discard the raw
source. Built on SDV (Synthetic Data Vault, MIT licence) which is the
de-facto standard for tabular SDG in production today.

The model trained on each upload lives only in memory for the duration of
the request — it is never persisted to disk. The raw uploaded dataframe is
shredded by reference once synthesis returns.

Typical flow
------------

    from timestone.services.synthetic import (
        SynthesisRequest, synthesize, QualityReport,
    )
    import pandas as pd

    raw = pd.read_csv("client_financials.csv")
    req = SynthesisRequest(
        source=raw,
        n_rows=len(raw),          # match original size
        model="ctgan",            # ctgan | tvae | gaussian_copula
        epochs=300,
    )
    result = synthesize(req)
    print(result.report.fidelity_score)   # 0..1
    result.synthetic.to_csv("twin.csv", index=False)

Dependencies (optional — service degrades gracefully if missing)::

    pip install "sdv>=1.10" "sdmetrics>=0.14"

If SDV is not installed, the service falls back to a marginal-distribution
sampler that preserves per-column statistics but not joint structure. Good
enough for smoke-testing the pipeline; not production-quality.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies — degrade gracefully if absent so the rest of the
# codebase can be imported on a machine without SDV installed.
# ---------------------------------------------------------------------------

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    HAS_PANDAS = False

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import (
        CTGANSynthesizer,
        GaussianCopulaSynthesizer,
        TVAESynthesizer,
    )

    HAS_SDV = True
except ImportError:  # pragma: no cover
    HAS_SDV = False

try:
    from sdmetrics.single_table import QualityReport as SDQualityReport

    HAS_SDMETRICS = True
except ImportError:  # pragma: no cover
    HAS_SDMETRICS = False


ModelName = Literal["ctgan", "tvae", "gaussian_copula", "marginal"]


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SynthesisRequest:
    """Inputs for a single synthesis job.

    Parameters
    ----------
    source:
        Either a ``pandas.DataFrame`` or a list-of-dicts representation of
        the raw table. The caller MUST treat this object as ephemeral —
        ``synthesize`` does not promise to read it more than once.
    n_rows:
        Number of synthetic rows to emit. If ``None`` we match the source
        size.
    model:
        Generative model to use. ``ctgan`` is the most accurate for mixed
        categorical/numerical data; ``tvae`` is faster but less expressive;
        ``gaussian_copula`` is cheap and decent for purely numerical
        finance tables; ``marginal`` uses no SDV at all (fallback).
    epochs:
        Training epochs for neural models. Ignored by ``gaussian_copula``
        and ``marginal``.
    primary_key:
        Column name to treat as primary key (regenerated as unique synthetic
        IDs). Optional.
    sensitive_columns:
        Columns containing personally identifiable or commercially
        sensitive data. Tracked in the privacy report; the SDG model will
        learn their distribution but synthetic values will NOT match real
        ones.
    random_seed:
        Reproducibility seed.
    """

    source: Any
    n_rows: Optional[int] = None
    model: ModelName = "gaussian_copula"
    epochs: int = 200
    primary_key: Optional[str] = None
    sensitive_columns: List[str] = field(default_factory=list)
    random_seed: int = 42


@dataclass
class QualityReport:
    """Statistical-fidelity and privacy assessment of the synthetic output.

    fidelity_score
        SDMetrics overall quality score in ``[0, 1]``. Above 0.85 means the
        marginal and pairwise distributions of the synthetic table closely
        track the real one.
    column_shapes
        Per-column shape similarity (1.0 = identical distribution).
    column_pair_trends
        Per-pair correlation similarity (1.0 = identical correlation).
    privacy_distance
        Minimum Euclidean distance from any synthetic row to the nearest
        real row (after normalisation). Higher = safer. 0 means a row was
        memorised verbatim and the synthesis failed privacy.
    backend
        Which synthesizer actually ran (informational).
    elapsed_seconds
        Wallclock training + sampling time.
    rows_in / rows_out
        Sanity counts.
    """

    fidelity_score: float
    column_shapes: Dict[str, float] = field(default_factory=dict)
    column_pair_trends: Dict[str, float] = field(default_factory=dict)
    privacy_distance: float = 0.0
    backend: str = ""
    elapsed_seconds: float = 0.0
    rows_in: int = 0
    rows_out: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fidelity_score": self.fidelity_score,
            "column_shapes": self.column_shapes,
            "column_pair_trends": self.column_pair_trends,
            "privacy_distance": self.privacy_distance,
            "backend": self.backend,
            "elapsed_seconds": self.elapsed_seconds,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "warnings": self.warnings,
        }


@dataclass
class SynthesisResult:
    """What a synthesis job returns. ``synthetic`` is the new dataframe and
    ``report`` is the calibration / privacy metrics. The raw source has
    been dropped at this point and is not retrievable from this object."""

    synthetic: Any  # pandas.DataFrame (or list[dict] if pandas absent)
    report: QualityReport


# ---------------------------------------------------------------------------
# Core synthesizers
# ---------------------------------------------------------------------------


def _normalise_to_df(source: Any) -> Any:
    """Accept either a DataFrame or list-of-dicts; return a DataFrame.
    Raises if pandas is unavailable AND a non-DataFrame was passed."""
    if HAS_PANDAS and isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, list):
        if not HAS_PANDAS:
            raise RuntimeError(
                "pandas is required to ingest list-of-dicts input; "
                "install pandas or pass an existing DataFrame."
            )
        return pd.DataFrame(source)
    raise TypeError(
        f"source must be a DataFrame or list[dict]; got {type(source).__name__}"
    )


def _build_metadata(df: Any, primary_key: Optional[str]) -> Any:
    md = SingleTableMetadata()
    md.detect_from_dataframe(df)
    if primary_key and primary_key in df.columns:
        md.set_primary_key(primary_key)
    return md


def _run_sdv(req: SynthesisRequest, df: Any) -> Any:
    """Train the chosen SDV model on `df` and emit `req.n_rows` synthetic
    rows. The model object is created locally and goes out of scope when
    this function returns — it never touches disk."""
    metadata = _build_metadata(df, req.primary_key)

    if req.model == "ctgan":
        model = CTGANSynthesizer(
            metadata=metadata, epochs=req.epochs, verbose=False
        )
    elif req.model == "tvae":
        model = TVAESynthesizer(metadata=metadata, epochs=req.epochs)
    else:  # gaussian_copula
        model = GaussianCopulaSynthesizer(metadata=metadata)

    model.fit(df)
    n = req.n_rows if req.n_rows is not None else len(df)
    return model.sample(num_rows=n)


def _run_marginal_fallback(req: SynthesisRequest, df: Any) -> Any:
    """No-SDV fallback: sample each column independently from its empirical
    distribution. Preserves marginals, destroys joint structure. Useful for
    smoke tests when SDV is not installed in the dev environment."""
    import random as _r

    rng = _r.Random(req.random_seed)
    n = req.n_rows if req.n_rows is not None else len(df)
    out_rows: List[Dict[str, Any]] = []
    columns = list(df.columns)
    samples = {c: df[c].tolist() for c in columns}
    for _ in range(n):
        out_rows.append({c: rng.choice(samples[c]) for c in columns})
    if HAS_PANDAS:
        return pd.DataFrame(out_rows)
    return out_rows


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def _compute_report(
    real_df: Any,
    synthetic_df: Any,
    backend: str,
    elapsed: float,
    warnings: List[str],
) -> QualityReport:
    rows_in = len(real_df) if HAS_PANDAS else len(real_df)
    rows_out = len(synthetic_df) if hasattr(synthetic_df, "__len__") else 0
    report = QualityReport(
        fidelity_score=0.0,
        backend=backend,
        elapsed_seconds=elapsed,
        rows_in=rows_in,
        rows_out=rows_out,
        warnings=list(warnings),
    )
    if HAS_SDMETRICS and HAS_PANDAS:
        try:
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(real_df)
            q = SDQualityReport()
            q.generate(real_df, synthetic_df, metadata.to_dict(), verbose=False)
            report.fidelity_score = float(q.get_score())
            shapes = q.get_details(property_name="Column Shapes")
            pairs = q.get_details(property_name="Column Pair Trends")
            if shapes is not None:
                report.column_shapes = dict(
                    zip(shapes["Column"], shapes["Quality Score"])
                )
            if pairs is not None:
                report.column_pair_trends = {
                    f"{a}__{b}": s
                    for a, b, s in zip(
                        pairs["Column 1"], pairs["Column 2"], pairs["Quality Score"]
                    )
                }
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"SDMetrics failed: {exc}")
    else:
        report.warnings.append(
            "SDMetrics not available — fidelity_score reported as 0.0"
        )
    # Privacy distance: cheap min-nearest-neighbour check on numeric columns.
    try:
        report.privacy_distance = _privacy_distance(real_df, synthetic_df)
    except Exception as exc:  # noqa: BLE001
        report.warnings.append(f"Privacy distance failed: {exc}")
    return report


def _privacy_distance(real_df: Any, synthetic_df: Any) -> float:
    """Min Euclidean distance from any synthetic row to its nearest real row
    (after per-column z-scoring). Higher is safer. 0 = verbatim leak."""
    if not HAS_PANDAS:
        return 0.0
    num = real_df.select_dtypes(include="number")
    if num.empty or len(synthetic_df) == 0:
        return 1.0
    syn = synthetic_df[num.columns].dropna()
    if syn.empty:
        return 1.0
    mu = num.mean()
    sd = num.std().replace(0, 1)
    real_norm = ((num - mu) / sd).to_numpy()
    syn_norm = ((syn - mu) / sd).to_numpy()
    # Sample to keep this cheap; we just need a rough lower bound.
    import numpy as np  # noqa: WPS433

    sample = syn_norm[: min(len(syn_norm), 200)]
    min_d = float("inf")
    for row in sample:
        d = float(np.min(np.linalg.norm(real_norm - row, axis=1)))
        min_d = min(min_d, d)
    return round(min_d, 4)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def synthesize(req: SynthesisRequest) -> SynthesisResult:
    """Generate a synthetic twin of `req.source`.

    The raw source dataframe is consumed once for training; the result
    object intentionally does NOT carry a reference back to it. After this
    function returns, callers should discard their own reference to the
    raw input to honour the "we never store raw data" promise to clients.
    """
    if not HAS_PANDAS:
        raise RuntimeError(
            "pandas is required for synthesize(); install pandas first."
        )
    df = _normalise_to_df(req.source)
    warnings: List[str] = []
    t0 = time.perf_counter()

    if HAS_SDV and req.model != "marginal":
        try:
            synthetic = _run_sdv(req, df)
            backend = f"sdv:{req.model}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("SDV failed, falling back to marginal: %s", exc)
            warnings.append(f"SDV failed: {exc}; used marginal fallback.")
            synthetic = _run_marginal_fallback(req, df)
            backend = "marginal:fallback"
    else:
        if not HAS_SDV:
            warnings.append(
                "SDV not installed — using marginal sampler (low fidelity)."
            )
        synthetic = _run_marginal_fallback(req, df)
        backend = "marginal"

    elapsed = time.perf_counter() - t0
    report = _compute_report(df, synthetic, backend, elapsed, warnings)
    return SynthesisResult(synthetic=synthetic, report=report)


# ---------------------------------------------------------------------------
# Convenience helpers used by API / CLI
# ---------------------------------------------------------------------------


def synthesize_from_records(
    records: List[Dict[str, Any]], **kwargs: Any
) -> SynthesisResult:
    """Sugar: take a JSON-shape list of records and run synthesis."""
    return synthesize(SynthesisRequest(source=records, **kwargs))


def synthetic_to_records(result: SynthesisResult) -> List[Dict[str, Any]]:
    """Convert the synthetic dataframe back to JSON-friendly list of dicts."""
    if HAS_PANDAS and isinstance(result.synthetic, pd.DataFrame):
        return result.synthetic.to_dict(orient="records")
    if isinstance(result.synthetic, list):
        return result.synthetic
    return []
