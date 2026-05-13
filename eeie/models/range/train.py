"""Training entrypoints for the Range engine."""

from __future__ import annotations

import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from eeie.config import get_settings
from eeie.features import build_range_features, load_features_from_db
from eeie.models.range.lstm import RangeLSTM
from eeie.models.range.predict import LSTM_CHECKPOINT, XGB_CHECKPOINT
from eeie.models.range.xgb import RangeXGB


def _load_or_panic(session: Session) -> pd.DataFrame:
    df = load_features_from_db(session)
    if df.empty:
        raise RuntimeError("No telemetry found. Run `python -m eeie.simulation.run` first.")
    return df


def train_range_xgb(session: Session) -> dict[str, float]:
    df = _load_or_panic(session)
    X, y = build_range_features(df)
    model = RangeXGB()
    metrics = model.fit(X, y)
    settings = get_settings()
    model.save(settings.checkpoint_dir / XGB_CHECKPOINT)
    logger.info("Saved RangeXGB to {}", settings.checkpoint_dir / XGB_CHECKPOINT)
    return metrics


def train_range_lstm(session: Session, *, epochs: int = 4) -> dict[str, float]:
    df = _load_or_panic(session)
    X, y = build_range_features(df)
    model = RangeLSTM()
    metrics = model.fit(X, y, epochs=epochs)
    settings = get_settings()
    target = settings.checkpoint_dir / LSTM_CHECKPOINT
    model.save(target)
    logger.info("Saved RangeLSTM to {}", target)
    return metrics
