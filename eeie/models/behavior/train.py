"""Training entrypoints for the Behavior engine."""

from __future__ import annotations

from loguru import logger
from sqlalchemy.orm import Session

from eeie.config import get_settings
from eeie.features import build_behavior_features, load_features_from_db
from eeie.models.behavior.analyze import (
    CLUSTER_CHECKPOINT,
    CONSUMPTION_CHECKPOINT,
)
from eeie.models.behavior.cluster import BehaviorClusterer
from eeie.models.behavior.consumption import ConsumptionRegressor


def train_behavior(session: Session) -> dict[str, dict[str, float]]:
    df = load_features_from_db(session)
    if df.empty:
        raise RuntimeError("No telemetry; run the simulator first.")
    feats = build_behavior_features(df)
    if feats.empty:
        raise RuntimeError("Behavior features are empty.")

    settings = get_settings()
    clusterer = BehaviorClusterer()
    cluster_metrics = clusterer.fit(feats)
    clusterer.save(settings.checkpoint_dir / CLUSTER_CHECKPOINT)
    logger.info("Saved BehaviorClusterer.")

    regressor = ConsumptionRegressor()
    cons_metrics = regressor.fit(feats)
    regressor.save(settings.checkpoint_dir / CONSUMPTION_CHECKPOINT)
    logger.info("Saved ConsumptionRegressor.")

    return {"cluster": cluster_metrics, "consumption": cons_metrics}
