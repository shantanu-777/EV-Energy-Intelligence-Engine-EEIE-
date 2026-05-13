"""Unified Range engine inference surface.

The API hits `predict_range(...)` rather than the individual XGBoost / LSTM
classes; selection between baseline and advanced model is controlled here
so the upstream consumer is model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from eeie.config import get_settings
from eeie.models.range.lstm import RangeLSTM
from eeie.models.range.xgb import RangeXGB

XGB_CHECKPOINT = "range/xgb.json"
LSTM_CHECKPOINT = "range/lstm.pt"


@dataclass
class RangePrediction:
    soc_drop_24h: float
    soc_drop_24h_lower: float
    soc_drop_24h_upper: float
    days_until_recharge: float
    undercharge_risk: float
    model_used: str
    confidence: float
    feature_contributions: dict[str, float]


def _resolve_path(rel: str) -> Path:
    return get_settings().checkpoint_dir / rel


def _xgb() -> RangeXGB:
    return RangeXGB.load(_resolve_path(XGB_CHECKPOINT))


def _lstm() -> RangeLSTM:
    return RangeLSTM.load(_resolve_path(LSTM_CHECKPOINT))


def predict_range(
    features: pd.DataFrame,
    *,
    current_soc: float,
    min_safe_soc: float = 0.20,
    model: str = "xgb",
) -> RangePrediction:
    """Predict next-24h SOC drop, days-to-recharge, and undercharge risk."""
    if features.empty:
        raise ValueError("`features` must contain at least one feature row.")

    importances: dict[str, float] = {}
    if model == "xgb":
        xgb_model = _xgb()
        point, lo, hi = xgb_model.predict_with_interval(features.tail(1))
        soc_drop = float(np.clip(point[0], 0.0, 1.0))
        soc_drop_lo = float(np.clip(lo[0], 0.0, 1.0))
        soc_drop_hi = float(np.clip(hi[0], 0.0, 1.0))
        if xgb_model.model is not None:
            try:
                importances = dict(
                    zip(
                        xgb_model.feature_names,
                        xgb_model.model.feature_importances_.tolist(),
                        strict=True,
                    )
                )
            except Exception:  # pragma: no cover - defensive
                importances = {}
    elif model == "lstm":
        lstm_model = _lstm()
        preds = lstm_model.predict(features)
        soc_drop = float(np.clip(preds[-1], 0.0, 1.0))
        sigma = 0.04
        soc_drop_lo = max(0.0, soc_drop - 1.96 * sigma)
        soc_drop_hi = min(1.0, soc_drop + 1.96 * sigma)
    else:
        raise ValueError(f"Unknown model '{model}', expected 'xgb' or 'lstm'.")

    days_until_recharge = (
        ((current_soc - min_safe_soc) / max(soc_drop, 1e-6)) if soc_drop > 0 else 30.0
    )
    days_until_recharge = float(np.clip(days_until_recharge, 0.0, 30.0))

    expected_soc = current_soc - soc_drop
    undercharge_risk = float(np.clip(1.0 - (expected_soc - min_safe_soc) / 0.30, 0.0, 1.0))

    confidence = float(np.clip(1.0 - (soc_drop_hi - soc_drop_lo) / 0.5, 0.0, 1.0))

    contribs = dict(sorted(importances.items(), key=lambda x: -x[1])[:5]) if importances else {}

    logger.debug(
        "predict_range model={} drop={:.3f} days={:.2f} risk={:.2f}",
        model,
        soc_drop,
        days_until_recharge,
        undercharge_risk,
    )

    return RangePrediction(
        soc_drop_24h=soc_drop,
        soc_drop_24h_lower=soc_drop_lo,
        soc_drop_24h_upper=soc_drop_hi,
        days_until_recharge=days_until_recharge,
        undercharge_risk=undercharge_risk,
        model_used=model,
        confidence=confidence,
        feature_contributions=contribs,
    )
