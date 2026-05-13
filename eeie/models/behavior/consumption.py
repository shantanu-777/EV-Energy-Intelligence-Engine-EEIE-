"""XGBoost regression for energy consumption per 100 km."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

CONSUMPTION_FEATURES = [
    "mean_speed",
    "aggressive_events_per_drive_hr",
    "mean_temp",
    "mean_soc",
    "soc_in_mid_band_pct",
    "battery_capacity_kwh",
]


@dataclass
class ConsumptionRegressor:
    """Predicts kWh/100km per vehicle from behavior features."""

    model: xgb.XGBRegressor | None = None
    feature_names: list[str] = field(default_factory=lambda: list(CONSUMPTION_FEATURES))

    def fit(self, df: pd.DataFrame, *, target_col: str = "energy_per_100km") -> dict[str, float]:
        X = df[self.feature_names].copy()
        y = df[target_col].copy()
        if len(X) < 10:
            self.model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=1)
            self.model.fit(X, y, verbose=False)
            preds = self.model.predict(X)
            rmse = float(np.sqrt(mean_squared_error(y, preds)))
            mae = float(mean_absolute_error(y, preds))
            return {"rmse": rmse, "mae": mae}

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_state=42,
            early_stopping_rounds=20,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = self.model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))
        logger.info("ConsumptionRegressor: RMSE={:.3f} MAE={:.3f}", rmse, mae)
        return {"rmse": rmse, "mae": mae}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ConsumptionRegressor not trained.")
        return self.model.predict(X[self.feature_names])

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> ConsumptionRegressor:
        m = xgb.XGBRegressor()
        m.load_model(str(path))
        return cls(model=m)
