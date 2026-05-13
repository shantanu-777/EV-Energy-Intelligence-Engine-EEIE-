"""Evaluation: metrics + naive/rule-based/EEIE ablation."""

from eeie.evaluation.ablation import AblationReport, run_charging_ablation
from eeie.evaluation.metrics import (
    calibration_error,
    cost_reduction_pct,
    peak_avoidance_pct,
    rmse_mae,
)

__all__ = [
    "AblationReport",
    "calibration_error",
    "cost_reduction_pct",
    "peak_avoidance_pct",
    "rmse_mae",
    "run_charging_ablation",
]
