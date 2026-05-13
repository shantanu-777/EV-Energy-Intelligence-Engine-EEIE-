"""Feature builder smoke tests."""

from __future__ import annotations

import pandas as pd

from eeie.features import (
    build_behavior_features,
    build_demand_features,
    build_hourly_features,
    build_range_features,
)


def _joined(sim) -> pd.DataFrame:
    tel = sim.telemetry.copy()
    veh = sim.vehicles[
        ["vehicle_id", "battery_capacity_kwh", "nominal_efficiency_kwh_per_km", "tariff_id"]
    ].copy()
    veh["driver_profile_id"] = "moderate"
    veh["archetype_id"] = "mid_sedan"
    df = tel.merge(veh, on="vehicle_id", how="left")
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


def test_hourly_features_nonempty(small_simulation):
    df = _joined(small_simulation)
    feats = build_hourly_features(df)
    assert not feats.empty
    assert "soc_lag_1h" in feats.columns
    assert "temp_factor" in feats.columns


def test_range_features_have_target(small_simulation):
    df = _joined(small_simulation)
    X, y = build_range_features(df)
    assert len(X) == len(y) and len(X) > 0
    assert "ambient_temp_c" in X.columns


def test_demand_features_have_target(small_simulation):
    df = _joined(small_simulation)
    daily = build_demand_features(df)
    assert not daily.empty
    assert "target_kwh_tomorrow" in daily.columns


def test_behavior_features_shape(small_simulation):
    df = _joined(small_simulation)
    beh = build_behavior_features(df)
    assert not beh.empty
    assert "energy_per_100km" in beh.columns
