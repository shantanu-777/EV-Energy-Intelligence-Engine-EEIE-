"""Temporal Fusion Transformer for daily demand forecasting.

Wraps `pytorch_forecasting.TemporalFusionTransformer` in a small adapter.
The TFT lives behind the same `predict()` surface as the XGBoost baseline,
so the API can flip the backend without touching upstream code.

Phase 1 trains for a small number of epochs to keep training cheap; the
intent is to prove the path end-to-end. Tuning is post-Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_forecasting import (
    GroupNormalizer,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.metrics import QuantileLoss

_MAX_PREDICTION_LENGTH = 1
_MAX_ENCODER_LENGTH = 14


def _make_dataset(df: pd.DataFrame, *, training: bool = True) -> TimeSeriesDataSet:
    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target_kwh_tomorrow",
        group_ids=["vehicle_id"],
        min_encoder_length=2,
        max_encoder_length=_MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=_MAX_PREDICTION_LENGTH,
        static_categoricals=["vehicle_id"],
        time_varying_known_reals=[
            "time_idx",
            "day_of_week",
            "is_weekend",
            "mean_temp",
            "max_temp",
            "min_temp",
        ],
        time_varying_unknown_reals=[
            "kwh_today",
            "kwh_yesterday",
            "kwh_7d_mean",
            "kwh_28d_mean",
            "km_today",
            "aggr_events",
            "mean_soc",
        ],
        target_normalizer=GroupNormalizer(groups=["vehicle_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


@dataclass
class DemandTFT:
    """Thin wrapper around pytorch-forecasting's TFT."""

    model: TemporalFusionTransformer | None = None
    dataset_params: dict | None = None

    def fit(
        self,
        df: pd.DataFrame,
        *,
        max_epochs: int = 3,
        batch_size: int = 64,
        learning_rate: float = 3e-3,
        hidden_size: int = 16,
        attn_heads: int = 1,
    ) -> dict[str, float]:
        training_ds = _make_dataset(df, training=True)
        validation_ds = TimeSeriesDataSet.from_dataset(
            training_ds, df, predict=True, stop_randomization=True
        )
        train_loader = training_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_loader = validation_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

        self.model = TemporalFusionTransformer.from_dataset(
            training_ds,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attn_heads,
            dropout=0.10,
            loss=QuantileLoss(),
            log_interval=0,
            reduce_on_plateau_patience=3,
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
            gradient_clip_val=0.1,
        )
        trainer.fit(self.model, train_loader, val_loader)
        self.dataset_params = training_ds.get_parameters()

        try:
            preds = self.model.predict(val_loader, mode="prediction")
            targets = torch.cat([y[0] for _, y in iter(val_loader)])
            mse = float(((preds.flatten() - targets.flatten()) ** 2).mean().item())
            rmse = float(np.sqrt(mse))
            logger.info("DemandTFT validation RMSE={:.3f} kWh", rmse)
            return {"rmse": rmse}
        except Exception as exc:  # pragma: no cover - eval is best-effort
            logger.warning("DemandTFT eval failed: {}", exc)
            return {"rmse": float("nan")}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.dataset_params is None:
            raise RuntimeError("DemandTFT is not trained.")
        ds = TimeSeriesDataSet.from_parameters(self.dataset_params, df, predict=True)
        loader = ds.to_dataloader(train=False, batch_size=64, num_workers=0)
        with torch.no_grad():
            preds = self.model.predict(loader, mode="prediction")
        return preds.cpu().numpy().reshape(-1)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hparams": dict(self.model.hparams),
                "dataset_params": self.dataset_params,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, df_for_ds_init: pd.DataFrame) -> DemandTFT:
        """Restore from disk. Needs a dataframe to recreate the dataset spec."""
        payload = torch.load(path, map_location="cpu")
        ds = TimeSeriesDataSet.from_parameters(payload["dataset_params"], df_for_ds_init)
        model = TemporalFusionTransformer.from_dataset(ds, **payload["hparams"])
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return cls(model=model, dataset_params=payload["dataset_params"])
