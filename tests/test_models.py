"""Round-trip smoke tests for the ML engines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeie.features import (
    build_behavior_features,
    build_demand_features,
    build_range_features,
)
from eeie.models.battery import predict_battery_health
from eeie.models.battery.empirical import annual_degradation_pct
from eeie.models.behavior.cluster import BehaviorClusterer
from eeie.models.behavior.consumption import ConsumptionRegressor
from eeie.models.demand.xgb import DemandXGB
from eeie.models.range.xgb import RangeXGB


def _joined(sim) -> pd.DataFrame:
    df = sim.telemetry.merge(
        sim.vehicles[
            ["vehicle_id", "battery_capacity_kwh", "nominal_efficiency_kwh_per_km", "tariff_id"]
        ],
        on="vehicle_id",
        how="left",
    )
    df["driver_profile_id"] = "moderate"
    df["archetype_id"] = "mid_sedan"
    df = df.merge(
        sim.weather[["ts", "humidity", "wind_speed_kmh", "precipitation_mm"]],
        on="ts",
        how="left",
    )
    df = df.merge(
        sim.tariffs[["ts", "tariff_id", "rate_eur_per_kwh", "tier"]],
        on=["ts", "tariff_id"],
        how="left",
    )
    return df


def test_range_xgb_roundtrip(small_simulation, tmp_path):
    df = _joined(small_simulation)
    X, y = build_range_features(df)
    model = RangeXGB()
    model.fit(X, y, n_estimators=50)
    path = tmp_path / "range.json"
    model.save(path)
    loaded = RangeXGB.load(path)
    preds = loaded.predict(X.head(10))
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()


def test_demand_xgb_roundtrip(small_simulation, tmp_path):
    df = _joined(small_simulation)
    daily = build_demand_features(df)
    if daily.empty:
        pytest.skip("Not enough days in mini simulation.")
    model = DemandXGB()
    model.fit(daily, n_estimators=80)
    path = tmp_path / "demand.json"
    model.save(path)
    loaded = DemandXGB.load(path)
    preds = loaded.predict(daily.head(5))
    assert preds.shape == (5,)


def test_battery_empirical_runs():
    pct = annual_degradation_pct(
        initial_soh=0.95,
        annual_eq_cycles=120.0,
        mean_soc=0.60,
        mean_temp_c=22.0,
        dc_fast_fraction=0.20,
        deep_cycle_fraction=0.30,
    )
    assert pct > 0


def test_battery_predict_shape():
    p = predict_battery_health(
        initial_soh=0.95,
        mean_soc=0.55,
        mean_temp_c=22.0,
        dc_fast_fraction=0.10,
        annual_eq_cycles=80.0,
        energy_per_100km=18.0,
        aggressive_events_per_drive_hr=1.0,
        soc_in_mid_band_pct=0.60,
    )
    assert 0.0 < p.projected_soh_5y <= 1.0
    assert len(p.trajectory) > 0


def test_behavior_cluster_and_consumption(small_simulation, tmp_path):
    df = _joined(small_simulation)
    beh = build_behavior_features(df)
    if beh.empty:
        pytest.skip("Too few vehicles in mini simulation.")
    cl = BehaviorClusterer(n_clusters=min(3, len(beh)))
    cl.fit(beh)
    cl_path = tmp_path / "cluster.joblib"
    cl.save(cl_path)
    out = BehaviorClusterer.load(cl_path).predict(beh)
    assert "cluster" in out.columns and "label" in out.columns

    reg = ConsumptionRegressor()
    reg.fit(beh)
    reg_path = tmp_path / "consumption.json"
    reg.save(reg_path)
    loaded = ConsumptionRegressor.load(reg_path)
    preds = loaded.predict(beh)
    assert len(preds) == len(beh)
