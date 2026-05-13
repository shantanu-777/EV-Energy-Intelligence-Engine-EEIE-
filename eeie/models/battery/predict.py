"""Unified Battery engine inference surface."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from eeie.config import get_settings
from eeie.models.battery.correction import BatterySOHCorrection
from eeie.models.battery.empirical import (
    annual_degradation_pct,
    predict_soh_trajectory_empirical,
)

CHECKPOINT = "battery/correction.json"


@dataclass
class BatteryPrediction:
    current_soh: float
    annual_degradation_pct: float
    projected_soh_3y: float
    projected_soh_5y: float
    trajectory: list[dict[str, float]] = field(default_factory=list)
    recommended_soc_band: tuple[float, float] = (0.20, 0.80)
    recommended_dc_fast_cap_pct: float = 0.25
    confidence: float = 0.7
    contributing_factors: dict[str, float] = field(default_factory=dict)


def _maybe_load_correction() -> BatterySOHCorrection | None:
    path = get_settings().checkpoint_dir / CHECKPOINT
    if not path.exists():
        return None
    try:
        return BatterySOHCorrection.load(path)
    except Exception:
        return None


def predict_battery_health(
    *,
    initial_soh: float,
    mean_soc: float,
    mean_temp_c: float,
    dc_fast_fraction: float,
    annual_eq_cycles: float,
    energy_per_100km: float,
    aggressive_events_per_drive_hr: float,
    soc_in_mid_band_pct: float,
    horizon_years: float = 5.0,
) -> BatteryPrediction:
    """Predict SOH trajectory + behavior-band recommendations."""
    deep_cycle_fraction = float(np.clip(1.0 - soc_in_mid_band_pct, 0.0, 1.0))

    base_annual = annual_degradation_pct(
        initial_soh=initial_soh,
        annual_eq_cycles=annual_eq_cycles,
        mean_soc=mean_soc,
        mean_temp_c=mean_temp_c,
        dc_fast_fraction=dc_fast_fraction,
        deep_cycle_fraction=deep_cycle_fraction,
    )

    correction = _maybe_load_correction()
    feature_row = pd.DataFrame(
        [
            {
                "mean_soc": mean_soc,
                "soc_in_mid_band_pct": soc_in_mid_band_pct,
                "mean_temp": mean_temp_c,
                "energy_per_100km": energy_per_100km,
                "aggressive_events_per_drive_hr": aggressive_events_per_drive_hr,
                "dc_fast_fraction": dc_fast_fraction,
                "annual_eq_cycles": annual_eq_cycles,
            }
        ]
    )

    correction_pct = (
        float(correction.predict(feature_row)[0]) * 100.0 if correction is not None else 0.0
    )
    annual_pct = max(0.0, base_annual + correction_pct)

    trajectory_df = predict_soh_trajectory_empirical(
        initial_soh=initial_soh,
        years=horizon_years,
        annual_eq_cycles=annual_eq_cycles,
        mean_soc=mean_soc,
        mean_temp_c=mean_temp_c,
        dc_fast_fraction=dc_fast_fraction,
        deep_cycle_fraction=deep_cycle_fraction,
    )
    if correction_pct != 0.0:
        trajectory_df["soh"] = np.clip(
            trajectory_df["soh"] - (correction_pct / 100.0) * trajectory_df["years"],
            0.5,
            1.0,
        )

    def _soh_at(year: float) -> float:
        idx = int(np.argmin(np.abs(trajectory_df["years"].to_numpy() - year)))
        return float(trajectory_df["soh"].iloc[idx])

    contributing = {
        "mean_soc": float(mean_soc),
        "mean_temp_c": float(mean_temp_c),
        "dc_fast_fraction": float(dc_fast_fraction),
        "deep_cycle_fraction": float(deep_cycle_fraction),
        "annual_eq_cycles": float(annual_eq_cycles),
    }

    confidence = 0.85 if correction is not None else 0.65

    return BatteryPrediction(
        current_soh=float(initial_soh),
        annual_degradation_pct=annual_pct,
        projected_soh_3y=_soh_at(3.0),
        projected_soh_5y=_soh_at(min(5.0, horizon_years)),
        trajectory=trajectory_df.to_dict(orient="records"),
        recommended_soc_band=(0.20, 0.80),
        recommended_dc_fast_cap_pct=0.25,
        confidence=confidence,
        contributing_factors=contributing,
    )
