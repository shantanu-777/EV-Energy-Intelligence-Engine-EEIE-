"""Numeric metrics used across the evaluation harness."""

from __future__ import annotations

import numpy as np


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    return {"rmse": rmse, "mae": mae}


def calibration_error(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of true values inside [lower, upper] minus nominal 0.95."""
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    return coverage - 0.95


def cost_reduction_pct(baseline_cost: float, optimized_cost: float) -> float:
    if baseline_cost <= 0:
        return 0.0
    return float(max(0.0, (baseline_cost - optimized_cost) / baseline_cost * 100.0))


def peak_avoidance_pct(baseline_peak_kwh: float, optimized_peak_kwh: float) -> float:
    if baseline_peak_kwh <= 0:
        return 0.0
    return float(max(0.0, (baseline_peak_kwh - optimized_peak_kwh) / baseline_peak_kwh * 100.0))


def degradation_reduction_pct(baseline_wear: float, optimized_wear: float) -> float:
    if baseline_wear <= 0:
        return 0.0
    return float(max(0.0, (baseline_wear - optimized_wear) / baseline_wear * 100.0))
