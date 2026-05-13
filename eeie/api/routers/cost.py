"""POST /cost_analysis: isolate EV electricity cost and project annual savings."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from eeie.api.deps import db, load_vehicle
from eeie.api.schemas import CostAnalysisRequest, CostAnalysisResponse
from eeie.db.models import ChargingEvent
from eeie.explainability.insight import cost_insight

router = APIRouter()


@router.post("/cost_analysis", response_model=CostAnalysisResponse, tags=["intelligence"])
def cost_analysis_endpoint(
    req: CostAnalysisRequest, session: Session = Depends(db)
) -> CostAnalysisResponse:
    vehicle = load_vehicle(session, req.vehicle_id)

    conds = [ChargingEvent.vehicle_id == req.vehicle_id]
    if req.start_ts is not None:
        conds.append(ChargingEvent.start_ts >= req.start_ts)
    if req.end_ts is not None:
        conds.append(ChargingEvent.end_ts <= req.end_ts)

    rows = session.execute(
        select(
            ChargingEvent.start_ts,
            ChargingEvent.end_ts,
            ChargingEvent.energy_kwh,
            ChargingEvent.cost_eur,
        ).where(and_(*conds))
    ).all()
    if not rows:
        raise HTTPException(status_code=404, detail="No charging events found.")

    starts = pd.to_datetime([r.start_ts for r in rows], utc=True)
    energies = np.array([float(r.energy_kwh) for r in rows])
    costs = np.array([float(r.cost_eur) for r in rows])

    span_days = max((starts.max() - starts.min()).days, 1)
    total_kwh = float(energies.sum())
    total_cost = float(costs.sum())
    monthly_cost = total_cost / span_days * 30.0
    annual_cost = total_cost / span_days * 365.0

    peak_hours_set = {7, 8, 9, 17, 18, 19, 20}
    peak_mask = starts.hour.isin(list(peak_hours_set))
    peak_kwh = float(energies[peak_mask].sum())
    peak_pct = (peak_kwh / total_kwh * 100.0) if total_kwh > 0 else 0.0

    # ICE comparison: kWh -> km via vehicle efficiency, then liters via input cons.
    annual_kwh = total_kwh / span_days * 365.0
    km_per_year = annual_kwh / float(vehicle.nominal_efficiency_kwh_per_km)
    ice_liters = km_per_year * (req.ice_consumption_l_per_100km / 100.0)
    ice_annual = float(ice_liters * req.ice_price_eur_per_liter)
    annual_savings = max(0.0, ice_annual - annual_cost)

    insight = cost_insight(
        monthly_eur=monthly_cost,
        annual_eur=annual_cost,
        ice_comparison_eur=ice_annual,
        confidence=0.9,
    )
    return CostAnalysisResponse(
        vehicle_id=req.vehicle_id,
        period_days=int(span_days),
        total_kwh_charged=total_kwh,
        total_cost_eur=total_cost,
        monthly_cost_eur=monthly_cost,
        annual_cost_eur=annual_cost,
        peak_hour_share_pct=peak_pct,
        ice_annual_cost_eur=ice_annual,
        annual_savings_vs_ice_eur=annual_savings,
        insight=insight,
    )
