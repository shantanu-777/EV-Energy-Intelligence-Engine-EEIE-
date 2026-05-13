"""POST /behavior_analysis."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from eeie.api.deps import db, load_vehicle
from eeie.api.schemas import BehaviorAnalysisRequest, BehaviorAnalysisResponse
from eeie.explainability.insight import behavior_insight
from eeie.features import build_behavior_features, load_features_from_db
from eeie.models.behavior import analyze_behavior

router = APIRouter()


@router.post("/behavior_analysis", response_model=BehaviorAnalysisResponse, tags=["intelligence"])
def behavior_analysis_endpoint(
    req: BehaviorAnalysisRequest, session: Session = Depends(db)
) -> BehaviorAnalysisResponse:
    load_vehicle(session, req.vehicle_id)
    df = load_features_from_db(session, vehicle_id=req.vehicle_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="No telemetry available.")

    feats = build_behavior_features(df)
    if feats.empty:
        raise HTTPException(status_code=422, detail="Insufficient driving history.")

    analysis = analyze_behavior(feats)
    insight = behavior_insight(
        feature_contributions=analysis.feature_contributions,
        potential_savings_pct=analysis.potential_savings_pct,
        cluster_label=analysis.cluster_label,
        confidence=analysis.confidence,
    )
    return BehaviorAnalysisResponse(
        vehicle_id=req.vehicle_id,
        cluster_id=analysis.cluster_id,
        cluster_label=analysis.cluster_label,
        predicted_kwh_per_100km=analysis.predicted_kwh_per_100km,
        efficient_kwh_per_100km=analysis.efficient_kwh_per_100km,
        potential_savings_kwh_per_100km=analysis.potential_savings_kwh_per_100km,
        potential_savings_pct=analysis.potential_savings_pct,
        insight=insight,
    )
