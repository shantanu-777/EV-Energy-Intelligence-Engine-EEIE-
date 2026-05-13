"""ML residual correction on top of the empirical aging model.

The empirical model gives a coarse prior; this layer learns the residual
(true SOH - empirical SOH) against the simulated ground truth. Inputs are
the aggregated per-vehicle features that drive aging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

CORRECTION_FEATURES = [
    "mean_soc",
    "soc_in_mid_band_pct",
    "mean_temp",
    "energy_per_100km",
    "aggressive_events_per_drive_hr",
    "dc_fast_fraction",
    "annual_eq_cycles",
]


@dataclass
class BatterySOHCorrection:
    """Residual XGBoost regressor on top of the empirical SOH model."""

    model: xgb.XGBRegressor | None = None
    feature_names: list[str] = field(default_factory=lambda: list(CORRECTION_FEATURES))

    def fit(
        self,
        X: pd.DataFrame,
        residual: pd.Series,
        *,
        random_state: int = 42,
    ) -> dict[str, float]:
        X_train, X_val, y_train, y_val = train_test_split(
            X[self.feature_names], residual, test_size=0.2, random_state=random_state
        )
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_state=random_state,
            early_stopping_rounds=20,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = self.model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))
        logger.info("BatterySOHCorrection: RMSE={:.5f} MAE={:.5f}", rmse, mae)
        return {"rmse": rmse, "mae": mae}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(X), dtype=np.float32)
        return self.model.predict(X[self.feature_names])

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Untrained correction model.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> BatterySOHCorrection:
        model = xgb.XGBRegressor()
        model.load_model(str(path))
        return cls(model=model)
