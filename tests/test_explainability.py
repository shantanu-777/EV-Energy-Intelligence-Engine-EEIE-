"""Explainability smoke tests."""

from __future__ import annotations

import pandas as pd
import pytest

from eeie.explainability import (
    Insight,
    partial_dependence,
    range_insight,
    shap_summary_for_tree,
)
from eeie.features import build_range_features
from eeie.models.range.xgb import RangeXGB


def _joined(sim) -> pd.DataFrame:
    df = sim.telemetry.merge(
        sim.vehicles[
            ["vehicle_id", "battery_capacity_kwh", "nominal_efficiency_kwh_per_km", "tariff_id"]
        ],
        on="vehicle_id",
    )
    df["driver_profile_id"] = "moderate"
    df["archetype_id"] = "mid_sedan"
    df = df.merge(sim.weather[["ts", "humidity", "wind_speed_kmh", "precipitation_mm"]], on="ts")
    df = df.merge(
        sim.tariffs[["ts", "tariff_id", "rate_eur_per_kwh", "tier"]],
        on=["ts", "tariff_id"],
    )
    return df


def test_shap_for_tree(small_simulation):
    df = _joined(small_simulation)
    X, y = build_range_features(df)
    if len(X) < 50:
        pytest.skip("Not enough rows for stable SHAP.")
    model = RangeXGB()
    model.fit(X, y, n_estimators=40)
    top = shap_summary_for_tree(model.model, X.head(64), top_k=3, sample=64)
    assert len(top) == 3
    assert all(isinstance(t[0], str) for t in top)


def test_partial_dependence(small_simulation):
    df = _joined(small_simulation)
    X, y = build_range_features(df)
    if len(X) < 50:
        pytest.skip("Not enough rows.")
    model = RangeXGB()
    model.fit(X, y, n_estimators=40)
    pdp = partial_dependence(model, X.head(100), "ambient_temp_c", grid_resolution=8, sample=50)
    assert "x" in pdp.columns and "mean_pred" in pdp.columns
    assert len(pdp) == 8


def test_insight_serializes():
    ins = range_insight(
        feature_contributions={"ambient_temp_c": 0.2, "soc": 0.1},
        confidence=0.8,
        days_until_recharge=3.5,
        undercharge_risk=0.2,
    )
    assert isinstance(ins, Insight)
    payload = ins.model_dump()
    assert payload["confidence"] == 0.8
    assert payload["top_factors"]
