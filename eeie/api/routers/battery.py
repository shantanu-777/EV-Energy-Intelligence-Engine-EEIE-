"""POST /battery_health."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from eeie.api.deps import db, load_vehicle
from eeie.api.schemas import BatteryHealthRequest, BatteryHealthResponse
from eeie.db.models import ChargingEvent
from eeie.explainability.insight import battery_insight
from eeie.features import build_behavior_features, load_features_from_db
from eeie.models.battery import predict_battery_health

router = APIRouter()


@router.post("/battery_health", response_model=BatteryHealthResponse, tags=["intelligence"])
def battery_health_endpoint(
    req: BatteryHealthRequest, session: Session = Depends(db)
) -> BatteryHealthResponse:
    vehicle = load_vehicle(session, req.vehicle_id)
    df = load_features_from_db(session, vehicle_id=req.vehicle_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="No telemetry available.")

    beh = build_behavior_features(df)
    if beh.empty:
        raise HTTPException(status_code=422, detail="Insufficient driving history.")
    row = beh.iloc[0]

    horizon_days = (
        pd.to_datetime(df["ts"], utc=True).max() - pd.to_datetime(df["ts"], utc=True).min()
    ).days
    horizon_years = max(horizon_days / 365.0, 1e-3)
    capacity = float(vehicle.battery_capacity_kwh)
    annual_eq_cycles = float(row["total_kwh"]) / capacity / horizon_years

    events = session.execute(
        select(ChargingEvent.is_dc_fast, ChargingEvent.energy_kwh).where(
            ChargingEvent.vehicle_id == req.vehicle_id
        )
    ).all()
    if events:
        total_energy = sum(float(e.energy_kwh) for e in events)
        dc_energy = sum(float(e.energy_kwh) for e in events if e.is_dc_fast)
        dc_fast_fraction = (dc_energy / total_energy) if total_energy > 0 else 0.0
    else:
        dc_fast_fraction = 0.0

    current_soh = float(df.sort_values("ts").iloc[-1]["soh"])

    pred = predict_battery_health(
        initial_soh=current_soh,
        mean_soc=float(row["mean_soc"]),
        mean_temp_c=float(row["mean_temp"]),
        dc_fast_fraction=float(dc_fast_fraction),
        annual_eq_cycles=float(annual_eq_cycles),
        energy_per_100km=float(row["energy_per_100km"]),
        aggressive_events_per_drive_hr=float(row["aggressive_events_per_drive_hr"]),
        soc_in_mid_band_pct=float(row["soc_in_mid_band_pct"]),
        horizon_years=float(req.horizon_years),
    )

    insight = battery_insight(
        contributing_factors=pred.contributing_factors,
        annual_degradation_pct=pred.annual_degradation_pct,
        projected_soh_5y=pred.projected_soh_5y,
        confidence=pred.confidence,
    )
    return BatteryHealthResponse(
        vehicle_id=req.vehicle_id,
        current_soh=pred.current_soh,
        annual_degradation_pct=pred.annual_degradation_pct,
        projected_soh_3y=pred.projected_soh_3y,
        projected_soh_5y=pred.projected_soh_5y,
        trajectory=pred.trajectory,
        recommended_soc_band=pred.recommended_soc_band,
        recommended_dc_fast_cap_pct=pred.recommended_dc_fast_cap_pct,
        insight=insight,
    )
