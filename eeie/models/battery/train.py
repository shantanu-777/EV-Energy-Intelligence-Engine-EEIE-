"""Training entrypoint for the battery residual-correction model.

Builds (per-vehicle) features and an empirical SOH prediction, then trains
an XGBoost residual to match the simulated end-of-horizon SOH.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from eeie.config import get_settings
from eeie.db.models import Vehicle
from eeie.features import build_behavior_features, load_features_from_db
from eeie.models.battery.correction import BatterySOHCorrection
from eeie.models.battery.empirical import (
    annual_degradation_pct,
)
from eeie.models.battery.predict import CHECKPOINT


def _build_battery_dataset(session: Session) -> pd.DataFrame:
    base = load_features_from_db(session)
    if base.empty:
        raise RuntimeError("No telemetry; run the simulator first.")
    beh = build_behavior_features(base)

    span_days = (
        base.groupby("vehicle_id")["ts"]
        .agg(lambda s: (pd.to_datetime(s, utc=True).max() - pd.to_datetime(s, utc=True).min()).days)
        .reset_index(name="span_days")
    )
    final_soh = (
        base.sort_values("ts").groupby("vehicle_id")["soh"].last().reset_index(name="final_soh")
    )
    fast_events = (
        base[base["is_charging"]]
        .assign(is_dc=lambda d: d["energy_charged_kwh"] >= 5.0)
        .groupby("vehicle_id")["is_dc"]
        .mean()
        .reset_index(name="dc_fast_fraction")
    )

    veh = pd.DataFrame(
        session.execute(select(Vehicle.vehicle_id, Vehicle.initial_soh)).all()
    )
    df = (
        beh.merge(span_days, on="vehicle_id")
        .merge(final_soh, on="vehicle_id")
        .merge(fast_events, on="vehicle_id", how="left")
        .merge(veh, on="vehicle_id", how="left")
    )
    df["dc_fast_fraction"] = df["dc_fast_fraction"].fillna(0.0)
    df["years"] = df["span_days"] / 365.0
    df["annual_eq_cycles"] = (
        df["total_kwh"] / df["battery_capacity_kwh"] / df["years"].replace(0, np.nan)
    ).fillna(0.0)
    df["deep_cycle_fraction"] = 1.0 - df["soc_in_mid_band_pct"]

    df["empirical_annual_pct"] = df.apply(
        lambda r: annual_degradation_pct(
            initial_soh=r["initial_soh"],
            annual_eq_cycles=r["annual_eq_cycles"],
            mean_soc=r["mean_soc"],
            mean_temp_c=r["mean_temp"],
            dc_fast_fraction=r["dc_fast_fraction"],
            deep_cycle_fraction=r["deep_cycle_fraction"],
        ),
        axis=1,
    )
    df["observed_annual_pct"] = (
        (df["initial_soh"] - df["final_soh"]) / df["years"].replace(0, np.nan) * 100.0
    ).fillna(0.0)
    df["residual"] = (df["observed_annual_pct"] - df["empirical_annual_pct"]) / 100.0
    return df


def train_battery_correction(session: Session) -> dict[str, float]:
    df = _build_battery_dataset(session)
    model = BatterySOHCorrection()
    metrics = model.fit(df, df["residual"])
    target = get_settings().checkpoint_dir / CHECKPOINT
    model.save(target)
    logger.info("Saved BatterySOHCorrection to {}", target)
    return metrics
