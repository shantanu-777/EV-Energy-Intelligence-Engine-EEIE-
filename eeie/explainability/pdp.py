"""Partial dependence utility.

Sweeps a single feature across a grid and records the model's mean
prediction at each grid point, holding the rest of the inputs at the
provided baseline distribution. Returns a tidy DataFrame ready for plotting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def partial_dependence(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    *,
    grid_resolution: int = 20,
    sample: int | None = 200,
) -> pd.DataFrame:
    """Return DataFrame with columns ['x', 'mean_pred']."""
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' missing from X.")
    base = X.sample(n=min(sample or len(X), len(X)), random_state=0).copy()
    lo = float(np.nanquantile(X[feature], 0.05))
    hi = float(np.nanquantile(X[feature], 0.95))
    grid = np.linspace(lo, hi, grid_resolution)

    means = []
    for v in grid:
        b = base.copy()
        b[feature] = v
        preds = model.predict(b)
        means.append(float(np.mean(preds)))
    return pd.DataFrame({"x": grid, "mean_pred": means})
