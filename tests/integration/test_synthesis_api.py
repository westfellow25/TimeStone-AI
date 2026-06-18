"""Integration tests for the /synthesize FastAPI endpoints.

Run with: pytest tests/integration/test_synthesis_api.py -v
"""
from __future__ import annotations

import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pandas")

from fastapi.testclient import TestClient

from timestone.interfaces.api.app import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


def test_health_endpoint(client: TestClient) -> None:
    r = client.get("/synthesize/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "sdv_available" in body
    assert "fallback_in_use" in body


def test_records_endpoint_roundtrip(client: TestClient) -> None:
    payload = {
        "records": [
            {"co": f"C{i}", "rev": 100 + i * 7, "seg": ["a", "b", "c"][i % 3]}
            for i in range(30)
        ],
        "n_rows": 12,
        "model": "marginal",
        "random_seed": 42,
    }
    r = client.post("/synthesize/records", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["synthetic"]) == 12
    assert {"co", "rev", "seg"}.issubset(body["synthetic"][0].keys())
    rep = body["report"]
    assert rep["backend"].startswith("marginal")
    assert rep["rows_in"] == 30
    assert rep["rows_out"] == 12
    assert "privacy_distance" in rep
    assert "fidelity_score" in rep


def test_csv_endpoint_roundtrip(client: TestClient) -> None:
    csv_text = "co,rev,seg\n" + "\n".join(
        f"C{i},{100 + i * 7},{['a','b','c'][i % 3]}" for i in range(30)
    )
    files = {"file": ("input.csv", io.BytesIO(csv_text.encode()), "text/csv")}
    data = {"model": "marginal", "n_rows": "10", "random_seed": "42"}
    r = client.post("/synthesize/csv", files=files, data=data)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv")
    assert "X-TimeStone-Backend" in r.headers
    assert r.headers["X-TimeStone-RowsIn"] == "30"
    assert r.headers["X-TimeStone-RowsOut"] == "10"

    import pandas as pd
    out = pd.read_csv(io.StringIO(r.text))
    assert len(out) == 10
    assert list(out.columns) == ["co", "rev", "seg"]


def test_records_unknown_model_falls_back_gracefully(client: TestClient) -> None:
    """Unknown backend name should fall back to marginal with a warning,
    rather than crashing. This is the documented graceful-degradation
    contract."""
    payload = {
        "records": [
            {"x": 1, "y": "a"}, {"x": 2, "y": "b"}, {"x": 3, "y": "a"}
        ] * 5,
        "model": "definitely_not_a_real_backend",
    }
    r = client.post("/synthesize/records", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["report"]["backend"].startswith("marginal")
    warnings = body["report"]["warnings"]
    assert any("marginal" in w.lower() or "sdv" in w.lower() for w in warnings)


def test_csv_endpoint_rejects_empty_upload(client: TestClient) -> None:
    files = {"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    r = client.post("/synthesize/csv", files=files)
    assert r.status_code == 400
    assert "empty" in r.text.lower()
