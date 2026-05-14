"""Kaggle Electric Vehicle Charging Patterns → vehicles + charging_events."""

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
    parse_utc,
    pct_to_fraction,
    project_columns,
    usd_to_eur,
    validate_curated,
)
from eeie.ingestion.schemas import ChargingEventRecord, VehicleRecord

SLUG = "charging_patterns"

_USER_TO_PROFILE = {
    "Commuter": "moderate",
    "Casual Driver": "efficient",
    "Long-Distance Traveler": "aggressive",
}
_DEFAULT_PROFILE = "moderate"


def _bucket_archetype(capacity_kwh: float) -> str:
    if capacity_kwh < 50.0:
        return "city_compact"
    if capacity_kwh < 70.0:
        return "mid_sedan"
    if capacity_kwh < 90.0:
        return "long_range_suv"
    return "performance"


def _vehicle_id(user_id: str) -> str:
    return f"cp_{user_id}"


def _build_charging_events(raw: pd.DataFrame) -> pd.DataFrame:
    charger = raw["Charger Type"].astype(str).str.strip()
    df = pd.DataFrame(
        {
            "event_id": [f"cp_{i:05d}" for i in range(len(raw))],
            "start_ts": parse_utc(raw["Charging Start Time"]),
            "end_ts": parse_utc(raw["Charging End Time"]),
            "vehicle_id": raw["User ID"].astype(str).map(_vehicle_id),
            "start_soc": pct_to_fraction(raw["State of Charge (Start %)"].astype(float)),
            "end_soc": pct_to_fraction(raw["State of Charge (End %)"].astype(float)),
            "energy_kwh": raw["Energy Consumed (kWh)"].astype(float).clip(lower=0.0),
            "avg_power_kw": raw["Charging Rate (kW)"].astype(float).clip(lower=0.0),
            "is_dc_fast": charger.str.casefold().eq("dc fast charger").astype(bool),
            "cost_eur": usd_to_eur(raw["Charging Cost (USD)"].astype(float)).clip(lower=0.0),
        }
    )
    required = [
        "start_ts",
        "end_ts",
        "start_soc",
        "end_soc",
        "energy_kwh",
        "avg_power_kw",
        "cost_eur",
    ]
    df = df.dropna(subset=required).reset_index(drop=True)
    return df.loc[df["end_ts"] > df["start_ts"]].reset_index(drop=True)


def _build_vehicles(raw: pd.DataFrame) -> pd.DataFrame:
    created_at = datetime.now(tz=UTC)
    rows: list[dict] = []
    for user_id, group in raw.groupby("User ID", sort=True):
        capacity = float(np.nanmedian(group["Battery Capacity (kWh)"].astype(float)))
        if not np.isfinite(capacity) or capacity <= 0:
            continue
        archetype = ARCHETYPES_BY_ID[_bucket_archetype(capacity)]
        observed_max_rate = float(group["Charging Rate (kW)"].astype(float).max())
        max_dc = max(archetype.max_dc_charge_kw, observed_max_rate)
        ut_raw = group["User Type"].dropna().astype(str).str.strip()
        modes = ut_raw.mode()
        user_type = str(modes.iloc[0]) if len(modes) else ""
        profile_id = _USER_TO_PROFILE.get(user_type, _DEFAULT_PROFILE)
        rows.append(
            {
                "vehicle_id": _vehicle_id(str(user_id)),
                "archetype_id": archetype.archetype_id,
                "driver_profile_id": profile_id,
                "battery_capacity_kwh": capacity,
                "nominal_efficiency_kwh_per_km": archetype.nominal_efficiency_kwh_per_km,
                "max_ac_charge_kw": archetype.max_ac_charge_kw,
                "max_dc_charge_kw": max_dc,
                "initial_soh": 1.0,
                "home_lat": DEFAULT_HOME_LAT,
                "home_lon": DEFAULT_HOME_LON,
                "tariff_id": REAL_TARIFF_ID,
                "created_at": created_at,
            }
        )
    return pd.DataFrame(rows)


def to_canonical(raw_path: Path) -> CuratedFrame:
    logger.info("Reading {} from {}", SLUG, raw_path)
    raw = pd.read_csv(raw_path)

    events = project_columns(_build_charging_events(raw), ChargingEventRecord)
    vehicles = project_columns(_build_vehicles(raw), VehicleRecord)

    events = events[events["vehicle_id"].isin(vehicles["vehicle_id"])].reset_index(drop=True)

    frame = CuratedFrame(slug=SLUG, vehicles=vehicles, charging_events=events)
    validate_curated(frame)
    logger.info("{} curated: {}", SLUG, frame.summary())
    return frame
