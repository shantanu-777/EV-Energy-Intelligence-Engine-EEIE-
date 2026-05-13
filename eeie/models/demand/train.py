"""Training entrypoints for the Demand engine."""

from __future__ import annotations

import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from eeie.config import get_settings
from eeie.features import build_demand_features, load_features_from_db
from eeie.models.demand.predict import TFT_CHECKPOINT, XGB_CHECKPOINT
from eeie.models.demand.xgb import DemandXGB


def _load_daily(session: Session) -> pd.DataFrame:
    df = load_features_from_db(session)
    if df.empty:
        raise RuntimeError("No telemetry; run the simulator first.")
    return build_demand_features(df)


def train_demand_xgb(session: Session) -> dict[str, float]:
    daily = _load_daily(session)
    model = DemandXGB()
    metrics = model.fit(daily)
    settings = get_settings()
    target = settings.checkpoint_dir / XGB_CHECKPOINT
    model.save(target)
    logger.info("Saved DemandXGB to {}", target)
    return metrics


def train_demand_tft(session: Session, *, max_epochs: int = 3) -> dict[str, float]:
    from eeie.models.demand.tft import DemandTFT

    daily = _load_daily(session)
    model = DemandTFT()
    metrics = model.fit(daily, max_epochs=max_epochs)
    settings = get_settings()
    target = settings.checkpoint_dir / TFT_CHECKPOINT
    model.save(target)
    logger.info("Saved DemandTFT to {}", target)
    return metrics
