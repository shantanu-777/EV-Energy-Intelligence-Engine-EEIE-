"""PyTorch LSTM sequence model for SOC depletion forecasting.

Given a window of the most recent feature rows, predict the SOC depletion
over the next 24 hours. The model is intentionally small so it can train
quickly on CPU in Phase 1; tune later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

SEQ_LEN = 24


class _LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.10 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def _build_windows(X: pd.DataFrame, y: pd.Series, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    X_np = X.to_numpy(dtype=np.float32, copy=False)
    y_np = y.to_numpy(dtype=np.float32, copy=False)
    n = len(X_np) - seq_len
    if n <= 0:
        raise ValueError(f"Need at least {seq_len + 1} rows, got {len(X_np)}.")
    xs = np.lib.stride_tricks.sliding_window_view(X_np, window_shape=(seq_len, X_np.shape[1]))
    xs = xs.reshape(-1, seq_len, X_np.shape[1])[:n]
    ys = y_np[seq_len - 1 : seq_len - 1 + n]
    return xs, ys


@dataclass
class RangeLSTM:
    """LSTM trainer/predictor wrapper."""

    feature_names: list[str] = field(default_factory=list)
    seq_len: int = SEQ_LEN
    hidden: int = 64
    num_layers: int = 2
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    model: _LSTMRegressor | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        epochs: int = 30,
        batch_size: int = 128,
        lr: float = 1e-3,
        device: str | None = None,
    ) -> dict[str, float]:
        self.feature_names = list(X.columns)
        device_ = device or ("cuda" if torch.cuda.is_available() else "cpu")

        merged = pd.concat([X.reset_index(drop=True), y.rename("_y_target_")], axis=1)
        merged = merged.replace([np.inf, -np.inf], np.nan)
        num_cols = list(X.columns)
        merged[num_cols] = merged[num_cols].fillna(merged[num_cols].median(numeric_only=True))
        merged[num_cols] = merged[num_cols].fillna(0.0)
        merged["_y_target_"] = merged["_y_target_"].fillna(merged["_y_target_"].median())
        merged = merged.dropna(subset=["_y_target_"])
        merged = merged[np.isfinite(merged["_y_target_"]).to_numpy()]

        X_df = merged[num_cols]
        y_series = merged["_y_target_"]

        X_np = np.ascontiguousarray(X_df.to_numpy(dtype=np.float32, copy=True))
        self.mean_ = X_np.mean(axis=0)
        self.std_ = X_np.std(axis=0) + 1e-6
        X_scaled = (X_np - self.mean_) / self.std_
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)

        xs, ys = _build_windows(X_scaled_df, y_series, self.seq_len)
        if len(xs) < 2:
            raise ValueError(
                f"LSTM training needs sliding windows beyond seq_len={self.seq_len}; got {len(xs)}."
            )
        split = int(0.8 * len(xs))
        split = max(1, min(split, len(xs) - 1))
        x_train, y_train = xs[:split], ys[:split]
        x_val, y_val = xs[split:], ys[split:]

        x_train = np.ascontiguousarray(x_train.astype(np.float32, copy=True))
        y_train = np.ascontiguousarray(y_train.astype(np.float32, copy=True))
        x_val = np.ascontiguousarray(x_val.astype(np.float32, copy=True))
        y_val = np.ascontiguousarray(y_val.astype(np.float32, copy=True))

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
            batch_size=batch_size,
            shuffle=False,
        )

        self.model = _LSTMRegressor(
            n_features=X_np.shape[1], hidden=self.hidden, num_layers=self.num_layers
        ).to(device_)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(device_)
                yb = yb.to(device_)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                val_losses: list[float] = []
                for xb, yb in val_loader:
                    xb = xb.to(device_)
                    yb = yb.to(device_)
                    pred = self.model(xb)
                    batch_loss = loss_fn(pred, yb).item()
                    if np.isfinite(batch_loss):
                        val_losses.append(batch_loss)
                val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            logger.info("RangeLSTM epoch {}: val_loss={:.5f}", epoch + 1, val_loss)

        return {
            "val_mse": val_loss,
            "rmse": float(np.sqrt(val_loss)) if val_loss == val_loss else float("nan"),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("RangeLSTM is not trained.")
        device_ = next(self.model.parameters()).device
        X_np = X[self.feature_names].to_numpy(dtype=np.float32)
        X_scaled = (X_np - self.mean_) / self.std_

        if len(X_scaled) < self.seq_len:
            pad = np.repeat(X_scaled[:1], self.seq_len - len(X_scaled), axis=0)
            X_scaled = np.concatenate([pad, X_scaled], axis=0)

        windows = np.lib.stride_tricks.sliding_window_view(
            X_scaled, window_shape=(self.seq_len, X_scaled.shape[1])
        )
        windows = windows.reshape(-1, self.seq_len, X_scaled.shape[1])
        windows = np.ascontiguousarray(windows.astype(np.float32, copy=True))
        with torch.no_grad():
            self.model.eval()
            preds = self.model(torch.from_numpy(windows).to(device_)).cpu().numpy()
        return preds

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save: model is untrained.")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "feature_names": self.feature_names,
            "seq_len": self.seq_len,
            "hidden": self.hidden,
            "num_layers": self.num_layers,
            "mean": self.mean_,
            "std": self.std_,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> RangeLSTM:
        payload = torch.load(path, map_location=device or "cpu")
        model = _LSTMRegressor(
            n_features=len(payload["feature_names"]),
            hidden=payload["hidden"],
            num_layers=payload["num_layers"],
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return cls(
            feature_names=payload["feature_names"],
            seq_len=payload["seq_len"],
            hidden=payload["hidden"],
            num_layers=payload["num_layers"],
            mean_=payload["mean"],
            std_=payload["std"],
            model=model,
        )
