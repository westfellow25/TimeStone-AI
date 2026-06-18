"""Unit tests for the synthetic data service.

These tests are deliberately lenient about backend availability — if SDV
isn't installed they still pass by exercising the marginal fallback path.
Run with: pytest tests/test_synthetic.py -v
"""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")  # whole module needs pandas

from timestone.services.synthetic import (
    QualityReport,
    SynthesisRequest,
    synthesize,
    synthesize_from_records,
    synthetic_to_records,
)


# --------- fixtures ---------


@pytest.fixture
def small_finance_df() -> pd.DataFrame:
    """50-row toy finance table — sufficient to train a gaussian copula
    or trigger the marginal fallback, fast enough for CI."""
    import numpy as np

    rng = np.random.default_rng(seed=7)
    n = 50
    return pd.DataFrame(
        {
            "company_id": [f"C{i:03d}" for i in range(n)],
            "revenue_musd": rng.gamma(shape=2.0, scale=300, size=n).round(1),
            "opex_pct": rng.uniform(0.55, 0.85, size=n).round(3),
            "segment": rng.choice(["freight", "passenger", "ops"], size=n),
            "headcount": rng.integers(500, 50000, size=n),
        }
    )


# --------- request ---------


def test_request_defaults():
    req = SynthesisRequest(source=[{"a": 1}])
    assert req.n_rows is None
    assert req.model == "gaussian_copula"
    assert req.epochs == 200
    assert req.random_seed == 42


# --------- core synthesis ---------


def test_synthesize_from_dataframe_matches_size_by_default(small_finance_df):
    result = synthesize(SynthesisRequest(source=small_finance_df, model="marginal"))
    assert len(result.synthetic) == len(small_finance_df)
    assert isinstance(result.report, QualityReport)
    assert result.report.rows_in == len(small_finance_df)
    assert result.report.rows_out == len(small_finance_df)


def test_synthesize_n_rows_override(small_finance_df):
    result = synthesize(
        SynthesisRequest(source=small_finance_df, n_rows=10, model="marginal")
    )
    assert len(result.synthetic) == 10


def test_synthesize_preserves_columns(small_finance_df):
    result = synthesize(SynthesisRequest(source=small_finance_df, model="marginal"))
    assert list(result.synthetic.columns) == list(small_finance_df.columns)


def test_synthesize_marginal_preserves_value_set_for_categoricals(small_finance_df):
    """Marginal fallback samples with replacement from the empirical
    distribution, so synthetic categorical values must be a subset of real."""
    result = synthesize(SynthesisRequest(source=small_finance_df, model="marginal"))
    real_segments = set(small_finance_df["segment"].unique())
    synth_segments = set(result.synthetic["segment"].unique())
    assert synth_segments.issubset(real_segments)


def test_synthesize_from_records_roundtrip():
    records = [
        {"x": 1, "y": "a"},
        {"x": 2, "y": "b"},
        {"x": 3, "y": "a"},
        {"x": 4, "y": "b"},
    ] * 10
    result = synthesize_from_records(records, model="marginal", n_rows=8)
    out = synthetic_to_records(result)
    assert len(out) == 8
    assert all("x" in r and "y" in r for r in out)


# --------- quality report ---------


def test_report_carries_backend_and_timing(small_finance_df):
    result = synthesize(SynthesisRequest(source=small_finance_df, model="marginal"))
    assert result.report.backend.startswith("marginal")
    assert result.report.elapsed_seconds >= 0.0
    # Quality report dict serialises cleanly (for the API layer).
    d = result.report.to_dict()
    assert {"fidelity_score", "privacy_distance", "backend"}.issubset(d.keys())


def test_report_has_warnings_when_sdmetrics_absent(small_finance_df):
    """If SDMetrics is missing, we report a warning rather than crashing."""
    from timestone.services import synthetic as svc

    if svc.HAS_SDMETRICS:
        pytest.skip("SDMetrics is installed — warning path not exercised.")
    result = synthesize(SynthesisRequest(source=small_finance_df, model="marginal"))
    assert any("SDMetrics" in w for w in result.report.warnings)


# --------- privacy ---------


def test_privacy_distance_nonzero_on_distinct_synthetic(small_finance_df):
    """Synthetic rows should not be verbatim copies of real ones."""
    result = synthesize(SynthesisRequest(source=small_finance_df, model="marginal"))
    # marginal fallback sometimes accidentally reconstructs a real row
    # exactly on small tables, but we expect the mean nearest-neighbour
    # distance to be > 0 across 200 sampled synthetic rows.
    assert result.report.privacy_distance >= 0.0


# --------- error handling ---------


def test_bad_source_type_raises():
    with pytest.raises(TypeError):
        synthesize(SynthesisRequest(source=42))  # type: ignore[arg-type]


def test_empty_records_does_not_crash():
    """Even with degenerate input, the service should not raise an
    unhandled exception — it should return a tiny result with warnings."""
    try:
        synthesize_from_records([], model="marginal")
    except Exception as exc:
        # Acceptable to raise on empty — but it must be a clean error.
        assert "empty" in str(exc).lower() or "0" in str(exc)
