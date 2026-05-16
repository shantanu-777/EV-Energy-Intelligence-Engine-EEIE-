"""Feature builders consumed by the ML engines.

Three feature views are produced from the joined hourly telemetry+weather+
tariff frame:

- `range`     : hourly + 24h-ahead target SOC depletion delta
- `demand`    : daily next-day energy demand (kWh)
- `behavior`  : per-vehicle aggregated efficiency / stress

The build* functions accept already-joined data; `load_features_from_db`
provides the canonical join from SQLAlchemy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from eeie.db.models import Tariff, Telemetry, Vehicle, Weather

FEATURE_VERSION = "1.0.0"


# ---------- DB join ----------


def load_features_from_db(
    session: Session,
    *,
    vehicle_id: str | None = None,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
    region_id: str = "central_eu",
) -> pd.DataFrame:
    """Return the hourly base table joined across telemetry + weather + tariff."""
    stmt = select(
        Telemetry.ts,
        Telemetry.vehicle_id,
        Telemetry.soc,
        Telemetry.odometer_km,
        Telemetry.km_driven_hour,
        Telemetry.energy_consumed_kwh,
        Telemetry.energy_charged_kwh,
        Telemetry.battery_temp_c,
        Telemetry.ambient_temp_c,
        Telemetry.avg_speed_kmh,
        Telemetry.aggressive_event_count,
        Telemetry.is_charging,
        Telemetry.is_driving,
        Telemetry.soh,
        Telemetry.data_source,
        Vehicle.battery_capacity_kwh,
        Vehicle.nominal_efficiency_kwh_per_km,
        Vehicle.driver_profile_id,
        Vehicle.archetype_id,
        Vehicle.tariff_id,
    ).join(Vehicle, Vehicle.vehicle_id == Telemetry.vehicle_id)

    if vehicle_id is not None:
        stmt = stmt.where(Telemetry.vehicle_id == vehicle_id)
    if start_ts is not None:
        stmt = stmt.where(Telemetry.ts >= start_ts)
    if end_ts is not None:
        stmt = stmt.where(Telemetry.ts <= end_ts)

    df = pd.DataFrame(session.execute(stmt).all())
    if df.empty:
        return df

    weather_stmt = select(
        Weather.ts,
        Weather.region_id,
        Weather.humidity,
        Weather.wind_speed_kmh,
        Weather.precipitation_mm,
    ).where(Weather.region_id == region_id)
    if start_ts is not None:
        weather_stmt = weather_stmt.where(Weather.ts >= start_ts)
    if end_ts is not None:
        weather_stmt = weather_stmt.where(Weather.ts <= end_ts)
    w = pd.DataFrame(session.execute(weather_stmt).all())

    tariff_stmt = select(
        Tariff.ts,
        Tariff.tariff_id,
        Tariff.rate_eur_per_kwh,
        Tariff.tier,
    )
    if start_ts is not None:
        tariff_stmt = tariff_stmt.where(Tariff.ts >= start_ts)
    if end_ts is not None:
        tariff_stmt = tariff_stmt.where(Tariff.ts <= end_ts)
    t = pd.DataFrame(session.execute(tariff_stmt).all())

    if not w.empty:
        df = df.merge(w.drop(columns=["region_id"]), on="ts", how="left")
    if not t.empty:
        df = df.merge(t, on=["ts", "tariff_id"], how="left")
    return df


# ---------- Common helpers ----------


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic hour/day features and is_weekend flag."""
    ts = pd.to_datetime(df["ts"], utc=True)
    df = df.copy()
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(np.int8)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    return df


def build_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the canonical hourly feature frame used by Range engine."""
    df = df.sort_values(["vehicle_id", "ts"]).reset_index(drop=True)
    df = _add_time_features(df)

    grouped = df.groupby("vehicle_id", group_keys=False)
    df["soc_lag_1h"] = grouped["soc"].shift(1)
    df["soc_lag_24h"] = grouped["soc"].shift(24)
    df["km_rolling_24h"] = grouped["km_driven_hour"].transform(
        lambda s: s.rolling(24, min_periods=1).sum()
    )
    df["km_rolling_7d"] = grouped["km_driven_hour"].transform(
        lambda s: s.rolling(168, min_periods=24).mean()
    )
    df["energy_consumed_rolling_24h"] = grouped["energy_consumed_kwh"].transform(
        lambda s: s.rolling(24, min_periods=1).sum()
    )
    df["dc_fast_ratio_30d"] = grouped["energy_charged_kwh"].transform(
        lambda s: s.rolling(720, min_periods=24).sum()
    )

    df["temp_factor"] = 1.0 + np.maximum(0.0, 5.0 - df["ambient_temp_c"]) * 0.020
    df["battery_stress"] = (
        np.maximum(0.0, df["battery_temp_c"] - 35.0) * 0.02
        + np.maximum(0.0, 10.0 - df["battery_temp_c"]) * 0.015
    )
    df["soc_band_mid"] = ((df["soc"] >= 0.20) & (df["soc"] <= 0.80)).astype(np.int8)

    return df.dropna(subset=["soc_lag_1h"]).reset_index(drop=True)


# ---------- Range features ----------


def build_range_features(
    df: pd.DataFrame,
    *,
    horizon_hours: int = 24,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and target y for the Range engine.

    Target: SOC depletion over the next `horizon_hours` hours (positive => SOC drop).
    """
    feats = build_hourly_features(df)
    feats = feats.sort_values(["vehicle_id", "ts"]).reset_index(drop=True)
    grouped = feats.groupby("vehicle_id", group_keys=False)
    feats["soc_future"] = grouped["soc"].shift(-horizon_hours)
    feats = feats.dropna(subset=["soc_future"]).copy()
    feats["target_soc_drop"] = feats["soc"] - feats["soc_future"]

    feature_cols = [
        "soc",
        "soc_lag_1h",
        "soc_lag_24h",
        "ambient_temp_c",
        "battery_temp_c",
        "humidity",
        "wind_speed_kmh",
        "km_rolling_24h",
        "km_rolling_7d",
        "energy_consumed_rolling_24h",
        "temp_factor",
        "battery_stress",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
        "battery_capacity_kwh",
        "nominal_efficiency_kwh_per_km",
    ]
    return feats[feature_cols].copy(), feats["target_soc_drop"].copy()


# ---------- Demand features ----------


def build_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Daily next-day energy demand features and target.

    Returns a long-form dataframe with one row per (vehicle, day) and the
    columns expected by both the XGBoost demand baseline and pytorch-
    forecasting's TimeSeriesDataSet:

        vehicle_id, date, time_idx, kwh_today, kwh_yesterday, ..., target_kwh_tomorrow
    """
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"] = df["ts"].dt.floor("D")
    daily = (
        df.groupby(["vehicle_id", "date"], as_index=False)
        .agg(
            kwh_today=("energy_consumed_kwh", "sum"),
            km_today=("km_driven_hour", "sum"),
            mean_temp=("ambient_temp_c", "mean"),
            max_temp=("ambient_temp_c", "max"),
            min_temp=("ambient_temp_c", "min"),
            aggr_events=("aggressive_event_count", "sum"),
            mean_soc=("soc", "mean"),
        )
        .sort_values(["vehicle_id", "date"])
    )
    g = daily.groupby("vehicle_id", group_keys=False)
    daily["kwh_yesterday"] = g["kwh_today"].shift(1)
    daily["km_yesterday"] = g["km_today"].shift(1)
    daily["kwh_7d_mean"] = g["kwh_today"].transform(lambda s: s.rolling(7, min_periods=2).mean())
    daily["kwh_28d_mean"] = g["kwh_today"].transform(lambda s: s.rolling(28, min_periods=4).mean())
    daily["target_kwh_tomorrow"] = g["kwh_today"].shift(-1)
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(np.int8)
    daily["time_idx"] = daily.groupby("vehicle_id").cumcount().astype(np.int32)

    # Backfill rolling means within each vehicle so pytorch-forecasting (TFT)
    # and downstream consumers don't see NaN in the first 7-28 days.
    for col in ("kwh_7d_mean", "kwh_28d_mean"):
        daily[col] = (
            daily.groupby("vehicle_id", group_keys=False)[col]
            .apply(lambda s: s.bfill())
            .fillna(daily["kwh_today"])
        )

    return daily.dropna(subset=["target_kwh_tomorrow", "kwh_yesterday"]).reset_index(drop=True)


# ---------- Behavior features ----------


def build_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-vehicle aggregated efficiency, stress, and consumption."""
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    only_driving = df[df["is_driving"]].copy()
    drive_agg = (
        only_driving.groupby("vehicle_id")
        .agg(
            mean_speed=("avg_speed_kmh", "mean"),
            aggressive_events_per_drive_hr=("aggressive_event_count", "mean"),
            total_km=("km_driven_hour", "sum"),
            total_kwh=("energy_consumed_kwh", "sum"),
            mean_temp=("ambient_temp_c", "mean"),
        )
        .reset_index()
    )
    drive_agg["energy_per_100km"] = np.where(
        drive_agg["total_km"] > 0,
        drive_agg["total_kwh"] / drive_agg["total_km"] * 100.0,
        np.nan,
    )

    overall = (
        df.groupby("vehicle_id")
        .agg(
            mean_soc=("soc", "mean"),
            soc_in_mid_band_pct=("soc", lambda s: float(((s >= 0.20) & (s <= 0.80)).mean())),
            dc_fast_energy=("energy_charged_kwh", "sum"),
        )
        .reset_index()
    )
    feats = drive_agg.merge(overall, on="vehicle_id", how="inner")
    if "battery_capacity_kwh" in df.columns:
        per_vehicle = df.groupby("vehicle_id")["battery_capacity_kwh"].first().reset_index()
        feats = feats.merge(per_vehicle, on="vehicle_id", how="left")
    return feats.dropna(subset=["energy_per_100km"]).reset_index(drop=True)
