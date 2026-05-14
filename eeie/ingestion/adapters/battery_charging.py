"""Kaggle EV Battery Charging telemetry → telemetry + VehicleRecord."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from eeie.config.vehicles import ARCHETYPES_BY_ID
from eeie.ingestion.adapters._common import (
    DEFAULT_HOME_LAT,
    DEFAULT_HOME_LON,
    REAL_TARIFF_ID,
    CuratedFrame,
    project_columns,
    validate_curated,
)
from eeie.ingestion.schemas import TelemetryRecord, VehicleRecord

SLUG = "battery_charging"
VEHICLE_ID = "bc_unit_01"
ARCHETYPE_ID = "mid_sedan"
DRIVER_PROFILE_ID = "efficient"
ASSUMED_PACK_KWH = 75.0
START_TS = pd.Timestamp("2024-01-01", tz="UTC")


def _build_telemetry(raw: pd.DataFrame) -> pd.DataFrame:
    n = len(raw)
    ts = START_TS + pd.to_timedelta(np.arange(n), unit="h")

    soc = raw["SOC"].astype(float).clip(lower=0.0, upper=1.0)
    soh = raw["SOH"].astype(float).clip(lower=0.0, upper=1.0)

    soc_delta = raw.get("soc_delta")
    if soc_delta is None:
        soc_delta = soc.diff().fillna(0.0)
    soc_delta = soc_delta.astype(float)

    charging = raw["action_current"].astype(float) > 0.0

    return pd.DataFrame(
        {
            "ts": ts,
            "vehicle_id": VEHICLE_ID,
            "soc": soc,
            "odometer_km": 0.0,
            "km_driven_hour": 0.0,
            "energy_consumed_kwh": 0.0,
            "energy_charged_kwh": (soc_delta.clip(lower=0.0) * ASSUMED_PACK_KWH).astype(float),
            "battery_temp_c": raw["battery_temp"].astype(float),
            "ambient_temp_c": raw["ambient_temp"].astype(float),
            "avg_speed_kmh": 0.0,
            "aggressive_event_count": 0,
            "is_charging": charging,
            "is_driving": pd.Series(False, index=charging.index),
            "soh": soh,
        }
    )


def _build_vehicle(initial_soh: float) -> pd.DataFrame:
    a = ARCHETYPES_BY_ID[ARCHETYPE_ID]
    row = {
        "vehicle_id": VEHICLE_ID,
        "archetype_id": a.archetype_id,
        "driver_profile_id": DRIVER_PROFILE_ID,
        "battery_capacity_kwh": ASSUMED_PACK_KWH,
        "nominal_efficiency_kwh_per_km": a.nominal_efficiency_kwh_per_km,
        "max_ac_charge_kw": a.max_ac_charge_kw,
        "max_dc_charge_kw": a.max_dc_charge_kw,
        "initial_soh": float(np.clip(initial_soh, 0.0, 1.0)),
        "home_lat": DEFAULT_HOME_LAT,
        "home_lon": DEFAULT_HOME_LON,
        "tariff_id": REAL_TARIFF_ID,
        "created_at": datetime.now(tz=UTC),
    }
    return pd.DataFrame([row])


def _coerce_native_bools(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "is_charging" in df.columns:
        df["is_charging"] = df["is_charging"].astype(bool)
    if "is_driving" in df.columns:
        df["is_driving"] = df["is_driving"].astype(bool)
    return df


def to_canonical(raw_path: Path) -> CuratedFrame:
    logger.info("Reading {} from {}", SLUG, raw_path)
    raw = pd.read_csv(raw_path).dropna(subset=["SOC", "SOH", "battery_temp", "ambient_temp"])

    telemetry = project_columns(_coerce_native_bools(_build_telemetry(raw)), TelemetryRecord)
    initial_soh = float(telemetry["soh"].iloc[0]) if len(telemetry) else 1.0
    vehicles = project_columns(_build_vehicle(initial_soh), VehicleRecord)

    frame = CuratedFrame(slug=SLUG, vehicles=vehicles, telemetry=telemetry)
    validate_curated(frame)
    logger.info("{} curated: {}", SLUG, frame.summary())
    return frame
