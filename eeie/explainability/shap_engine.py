"""SHAP value extraction utilities.

Two narrow helpers cover the Phase 1 needs:

- `shap_summary_for_tree`: works for any sklearn-API tree model
  (XGBoost, RandomForest, etc.) using TreeExplainer.
- `shap_summary_for_torch`: KernelExplainer fallback for the LSTM / TFT.
  Slow on raw networks — sampled small.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


def shap_summary_for_tree(
    model: Any,
    X: pd.DataFrame,
    *,
    top_k: int = 5,
    sample: int | None = 256,
) -> list[tuple[str, float]]:
    """Return [(feature, mean_abs_shap)] sorted desc, truncated to top_k."""
    X_use = X.sample(n=sample, random_state=0) if sample is not None and len(X) > sample else X
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_use)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(
        zip(X_use.columns.tolist(), mean_abs.tolist(), strict=True),
        key=lambda x: -x[1],
    )
    return ranked[:top_k]


def shap_summary_for_torch(
    predict_fn,
    background: np.ndarray,
    sample: np.ndarray,
    feature_names: list[str],
    *,
    top_k: int = 5,
    nsamples: int = 100,
) -> list[tuple[str, float]]:
    """KernelExplainer-based SHAP for a generic PyTorch (or any) callable.

    `predict_fn` must accept a 2D numpy array (n, n_features) and return a
    1D array of predictions. Background must be a small representative
    sample (e.g. k=50 rows).
    """
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(sample, nsamples=nsamples)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs.tolist(), strict=True), key=lambda x: -x[1])
    return ranked[:top_k]
