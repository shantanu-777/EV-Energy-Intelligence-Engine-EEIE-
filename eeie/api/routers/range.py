"""POST /predict_range."""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from eeie.api.deps import db, load_vehicle
from eeie.api.schemas import PredictRangeRequest, PredictRangeResponse
from eeie.db.models import Telemetry
from eeie.explainability.insight import range_insight
from eeie.features import build_range_features, load_features_from_db
from eeie.models.range.predict import predict_range

router = APIRouter()


@router.post("/predict_range", response_model=PredictRangeResponse, tags=["intelligence"])
def predict_range_endpoint(
    req: PredictRangeRequest, session: Session = Depends(db)
) -> PredictRangeResponse:
    load_vehicle(session, req.vehicle_id)

    latest_ts = session.execute(
        select(Telemetry.ts)
        .where(Telemetry.vehicle_id == req.vehicle_id)
        .order_by(Telemetry.ts.desc())
        .limit(1)
    ).scalar_one_or_none()
    if latest_ts is None:
        raise HTTPException(status_code=404, detail="No telemetry available.")

    start_ts = latest_ts - timedelta(hours=req.lookback_hours)
    base = load_features_from_db(
        session, vehicle_id=req.vehicle_id, start_ts=start_ts, end_ts=latest_ts
    )
    if base.empty:
        raise HTTPException(status_code=404, detail="Telemetry window is empty.")

    X, _ = build_range_features(base)
    if X.empty:
        raise HTTPException(status_code=422, detail="Not enough history to build features.")

    current_soc = float(base.sort_values("ts").iloc[-1]["soc"])
    pred = predict_range(X, current_soc=current_soc, model=req.model)
    insight = range_insight(
        feature_contributions=pred.feature_contributions,
        confidence=pred.confidence,
        days_until_recharge=pred.days_until_recharge,
        undercharge_risk=pred.undercharge_risk,
    )
    return PredictRangeResponse(
        vehicle_id=req.vehicle_id,
        current_soc=current_soc,
        soc_drop_24h=pred.soc_drop_24h,
        soc_drop_24h_lower=pred.soc_drop_24h_lower,
        soc_drop_24h_upper=pred.soc_drop_24h_upper,
        days_until_recharge=pred.days_until_recharge,
        undercharge_risk=pred.undercharge_risk,
        model_used=pred.model_used,
        insight=insight,
    )
