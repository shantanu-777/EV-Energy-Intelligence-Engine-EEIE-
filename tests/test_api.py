"""FastAPI smoke tests.

We exercise the routes that don't require trained models or a populated
database: health and root. Full endpoint integration tests are marked
`integration` and live elsewhere.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from eeie.api.main import app


def test_healthz():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert "version" in payload
    assert "feature_version" in payload


def test_root():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    payload = r.json()
    assert payload["service"] == "EEIE"


def test_openapi_lists_intelligence_endpoints():
    client = TestClient(app)
    spec = client.get("/openapi.json").json()
    paths = set(spec["paths"].keys())
    assert {
        "/predict_range",
        "/optimize_charging",
        "/battery_health",
        "/cost_analysis",
        "/behavior_analysis",
    }.issubset(paths)
