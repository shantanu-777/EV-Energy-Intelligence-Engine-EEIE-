"""POST /optimize_charging."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from eeie.api.deps import db, load_vehicle
from eeie.api.schemas import OptimizeChargingRequest, OptimizeChargingResponse
from eeie.config.tariffs import TARIFFS_BY_ID
from eeie.db.models import Tariff, Telemetry
from eeie.explainability.insight import optimization_insight
from eeie.models.optimization import OptimizationInputs, optimize_charging
from eeie.models.optimization.optimize import naive_plan

router = APIRouter()


def _next_hours_tariff(
    session: Session, tariff_id: str, start_ts, periods: int
) -> tuple[list[float], list[int]]:
    end_ts = start_ts + timedelta(hours=periods)
    rows = session.execute(
        select(Tariff.ts, Tariff.rate_eur_per_kwh, Tariff.tier)
        .where(Tariff.tariff_id == tariff_id)
        .where(Tariff.ts >= start_ts)
        .where(Tariff.ts < end_ts)
        .order_by(Tariff.ts.asc())
    ).all()
    if len(rows) >= periods:
        rates = [float(r.rate_eur_per_kwh) for r in rows[:periods]]
        peaks = [int(r.tier == "peak") for r in rows[:periods]]
        return rates, peaks

    sched = TARIFFS_BY_ID.get(tariff_id) or next(iter(TARIFFS_BY_ID.values()))
    rates = []
    peaks = []
    cur = start_ts
    for _ in range(periods):
        ts = pd.Timestamp(cur).tz_convert("UTC")
        r, tier = sched.rate_at(int(ts.hour), bool(ts.dayofweek >= 5))
        rates.append(float(r))
        peaks.append(int(tier == "peak"))
        cur = cur + timedelta(hours=1)
    return rates, peaks


router_router = router


@router.post("/optimize_charging", response_model=OptimizeChargingResponse, tags=["intelligence"])
def optimize_charging_endpoint(
    req: OptimizeChargingRequest, session: Session = Depends(db)
) -> OptimizeChargingResponse:
    vehicle = load_vehicle(session, req.vehicle_id)

    latest = session.execute(
        select(Telemetry)
        .where(Telemetry.vehicle_id == req.vehicle_id)
        .order_by(Telemetry.ts.desc())
        .limit(1)
    ).scalar_one_or_none()
    if latest is None:
        raise HTTPException(status_code=404, detail="No telemetry available.")

    initial_soc = req.initial_soc if req.initial_soc is not None else float(latest.soc)
    start_ts = latest.ts
    rates, peaks = _next_hours_tariff(session, vehicle.tariff_id, start_ts, req.horizon_hours)
    battery_temps = [float(latest.battery_temp_c)] * req.horizon_hours

    inputs = OptimizationInputs(
        horizon_hours=req.horizon_hours,
        initial_soc=initial_soc,
        target_soc=req.target_soc,
        soc_min=0.10,
        soc_max=0.95,
        capacity_kwh=float(vehicle.battery_capacity_kwh),
        max_power_kw=float(vehicle.max_ac_charge_kw),
        charging_efficiency=0.93,
        hourly_rates=rates,
        peak_flags=peaks,
        battery_temp_c=battery_temps,
        lambda_wear=req.lambda_wear,
        lambda_peak=req.lambda_peak,
    )

    plan = optimize_charging(inputs, optimizer=req.optimizer)
    naive = naive_plan(inputs)
    savings_eur = max(0.0, naive.total_cost_eur - plan.total_cost_eur)
    savings_pct = (savings_eur / naive.total_cost_eur) if naive.total_cost_eur > 0 else 0.0

    insight = optimization_insight(
        feature_contributions=plan.feature_contributions,
        total_cost_eur=plan.total_cost_eur,
        naive_cost_eur=naive.total_cost_eur,
        confidence=plan.confidence,
    )
    return OptimizeChargingResponse(
        vehicle_id=req.vehicle_id,
        optimizer=plan.optimizer,
        feasible=plan.feasible,
        power_per_hour_kw=plan.power_per_hour,
        soc_per_hour=plan.soc_per_hour,
        total_cost_eur=plan.total_cost_eur,
        naive_cost_eur=naive.total_cost_eur,
        savings_vs_naive_eur=savings_eur,
        savings_vs_naive_pct=savings_pct,
        chosen_window_start_hour=plan.chosen_window_start_hour,
        chosen_window_end_hour=plan.chosen_window_end_hour,
        chosen_avg_rate_kw=plan.chosen_avg_rate_kw,
        insight=insight,
    )
