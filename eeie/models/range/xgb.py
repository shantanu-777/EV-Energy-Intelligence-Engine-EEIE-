"""XGBoost baseline for range / SOC-depletion forecasting.

Target: SOC drop over the next 24 hours, given the current hourly feature
row. Wraps `xgboost.XGBRegressor` in a small persistence-aware class so the
training script and the API serve from a single artifact.
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


@dataclass
class RangeXGB:
    """XGBoost regressor for `target_soc_drop`."""

    model: xgb.XGBRegressor | None = None
    feature_names: list[str] = field(default_factory=list)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
        test_size: float = 0.2,
    ) -> dict[str, float]:
        self.feature_names = list(X.columns)
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
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = self.model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))
        logger.info("RangeXGB validation: RMSE={:.4f} MAE={:.4f}", rmse, mae)
        return {"rmse": rmse, "mae": mae}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RangeXGB is not trained. Call .fit() first.")
        return self.model.predict(X[self.feature_names])

    def predict_with_interval(
        self, X: pd.DataFrame, *, sigma: float = 0.03
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (point, lower_95, upper_95) using a constant sigma proxy."""
        point = self.predict(X)
        return point, point - 1.96 * sigma, point + 1.96 * sigma

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save: model is untrained.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        (path.with_suffix(".features.json")).write_text(
            pd.Series(self.feature_names).to_json(orient="values")
        )

    @classmethod
    def load(cls, path: Path) -> RangeXGB:
        model = xgb.XGBRegressor()
        model.load_model(str(path))
        features = pd.read_json(path.with_suffix(".features.json"), typ="series").tolist()
        return cls(model=model, feature_names=features)
