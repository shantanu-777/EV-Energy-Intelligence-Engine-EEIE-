"""Unified Demand engine inference surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eeie.config import get_settings
from eeie.models.demand.xgb import DEMAND_FEATURES, DemandXGB

XGB_CHECKPOINT = "demand/xgb.json"
TFT_CHECKPOINT = "demand/tft.pt"


@dataclass
class DemandPrediction:
    expected_kwh_tomorrow: float
    lower_kwh: float
    upper_kwh: float
    model_used: str
    confidence: float
    feature_contributions: dict[str, float]


def _xgb() -> DemandXGB:
    return DemandXGB.load(get_settings().checkpoint_dir / XGB_CHECKPOINT)


def predict_demand(daily_features: pd.DataFrame, *, model: str = "xgb") -> DemandPrediction:
    """Predict next-day kWh demand for the latest row in `daily_features`."""
    if daily_features.empty:
        raise ValueError("daily_features is empty.")

    row = daily_features.tail(1)
    if model == "xgb":
        m = _xgb()
        pred = float(m.predict(row)[0])
        sigma = max(2.0, abs(pred) * 0.10)
        importances: dict[str, float] = {}
        if m.model is not None:
            importances = dict(
                zip(
                    DEMAND_FEATURES,
                    m.model.feature_importances_.tolist(),
                    strict=True,
                )
            )
    elif model == "tft":
        from eeie.models.demand.tft import DemandTFT

        tft = DemandTFT.load(get_settings().checkpoint_dir / TFT_CHECKPOINT, daily_features)
        pred = float(tft.predict(daily_features)[-1])
        sigma = max(2.0, abs(pred) * 0.12)
        importances = {}
    else:
        raise ValueError(f"Unknown model '{model}'. Use 'xgb' or 'tft'.")

    lower = max(0.0, pred - 1.96 * sigma)
    upper = pred + 1.96 * sigma
    confidence = float(np.clip(1.0 - (upper - lower) / max(abs(pred) * 2.0, 4.0), 0.0, 1.0))
    contribs = dict(sorted(importances.items(), key=lambda x: -x[1])[:5]) if importances else {}

    return DemandPrediction(
        expected_kwh_tomorrow=pred,
        lower_kwh=lower,
        upper_kwh=upper,
        model_used=model,
        confidence=confidence,
        feature_contributions=contribs,
    )
