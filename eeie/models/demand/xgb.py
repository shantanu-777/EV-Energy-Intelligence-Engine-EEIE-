"""XGBoost baseline for next-day energy demand."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

DEMAND_FEATURES = [
    "kwh_today",
    "km_today",
    "mean_temp",
    "max_temp",
    "min_temp",
    "aggr_events",
    "mean_soc",
    "kwh_yesterday",
    "km_yesterday",
    "kwh_7d_mean",
    "kwh_28d_mean",
    "day_of_week",
    "is_weekend",
]


@dataclass
class DemandXGB:
    """Tomorrow's energy-consumption forecast in kWh."""

    model: xgb.XGBRegressor | None = None
    feature_names: list[str] = field(default_factory=lambda: list(DEMAND_FEATURES))

    def fit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "target_kwh_tomorrow",
        n_estimators: int = 350,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, float]:
        X = df[self.feature_names].copy()
        y = df[target_col].copy()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=random_state,
            early_stopping_rounds=25,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = self.model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))
        logger.info("DemandXGB validation: RMSE={:.3f} kWh MAE={:.3f} kWh", rmse, mae)
        return {"rmse": rmse, "mae": mae}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DemandXGB not trained.")
        return self.model.predict(X[self.feature_names])

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> DemandXGB:
        model = xgb.XGBRegressor()
        model.load_model(str(path))
        return cls(model=model)
