"""Unified Behavior engine inference + recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from eeie.config import get_settings
from eeie.models.behavior.cluster import BehaviorClusterer
from eeie.models.behavior.consumption import (
    CONSUMPTION_FEATURES,
    ConsumptionRegressor,
)

CLUSTER_CHECKPOINT = "behavior/cluster.joblib"
CONSUMPTION_CHECKPOINT = "behavior/consumption.json"


@dataclass
class BehaviorAnalysis:
    cluster_label: str
    cluster_id: int
    predicted_kwh_per_100km: float
    efficient_kwh_per_100km: float
    potential_savings_kwh_per_100km: float
    potential_savings_pct: float
    confidence: float
    feature_contributions: dict[str, float] = field(default_factory=dict)


def analyze_behavior(vehicle_features: pd.DataFrame) -> BehaviorAnalysis:
    """Return cluster label + consumption forecast + improvement potential."""
    if vehicle_features.empty:
        raise ValueError("vehicle_features must contain at least one row.")

    settings = get_settings()
    cluster_path = settings.checkpoint_dir / CLUSTER_CHECKPOINT
    consumption_path = settings.checkpoint_dir / CONSUMPTION_CHECKPOINT

    clusterer = BehaviorClusterer.load(cluster_path)
    regressor = ConsumptionRegressor.load(consumption_path)

    cluster_pred = clusterer.predict(vehicle_features)
    label = str(cluster_pred["label"].iloc[0])
    cluster_id = int(cluster_pred["cluster"].iloc[0])

    pred_kwh = float(regressor.predict(vehicle_features)[0])

    efficient_row = vehicle_features.copy()
    efficient_row["aggressive_events_per_drive_hr"] = (
        efficient_row["aggressive_events_per_drive_hr"] * 0.3
    )
    efficient_row["mean_speed"] = np.clip(efficient_row["mean_speed"] * 0.92, 25, 110)
    efficient_kwh = float(regressor.predict(efficient_row)[0])
    savings = max(0.0, pred_kwh - efficient_kwh)
    savings_pct = float(np.clip(savings / pred_kwh, 0.0, 1.0)) if pred_kwh > 0 else 0.0

    importances: dict[str, float] = {}
    if regressor.model is not None:
        importances = dict(
            zip(
                CONSUMPTION_FEATURES,
                regressor.model.feature_importances_,
                strict=True,
            )
        )

    return BehaviorAnalysis(
        cluster_label=label,
        cluster_id=cluster_id,
        predicted_kwh_per_100km=pred_kwh,
        efficient_kwh_per_100km=efficient_kwh,
        potential_savings_kwh_per_100km=savings,
        potential_savings_pct=savings_pct,
        confidence=0.80,
        feature_contributions=dict(sorted(importances.items(), key=lambda x: -x[1])[:5]),
    )
