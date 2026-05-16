"""Fit lightweight priors for the synthetic fleet from real curated tables.

``charging_events`` rows drive DC-fast mix, session energy (proxy for daily
km), and typical plug-in SOC. Optional ``telemetry`` with ``soh`` tightens
initial SOH sampling (e.g. battery lab CSVs).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from eeie.ingestion.adapters import CuratedFrame

_DEFAULT_DRIVER_PROBS = (0.35, 0.45, 0.20)
_FAST_PREFS = (0.10, 0.25, 0.50)
_NOMINAL_KWH_PER_KM = 0.175


@dataclass(frozen=True)
class CalibrationProfile:
    """Priors passed into ``sample_fleet`` / ``_simulate_vehicle``."""

    driver_profile_probs: tuple[float, float, float]
    weekday_km_mean: float
    weekend_km_mean: float
    initial_soh_low: float
    initial_soh_high: float
    median_plug_in_soc: float

    @staticmethod
    def defaults() -> CalibrationProfile:
        return CalibrationProfile(
            driver_profile_probs=_DEFAULT_DRIVER_PROBS,
            weekday_km_mean=45.0,
            weekend_km_mean=30.0,
            initial_soh_low=0.92,
            initial_soh_high=1.0,
            median_plug_in_soc=0.30,
        )


def _normalize_probs(p: tuple[float, float, float]) -> tuple[float, float, float]:
    arr = np.clip(np.array(p, dtype=float), 0.0, None)
    s = float(arr.sum())
    if s < 1e-9:
        return _DEFAULT_DRIVER_PROBS
    normed = arr / s
    return float(normed[0]), float(normed[1]), float(normed[2])


def driver_probs_for_dc_share(dc_share: float) -> tuple[float, float, float]:
    """Mix driver archetypes so mean ``fast_charge_preference`` matches ``dc_share``."""
    t = float(np.clip(dc_share, _FAST_PREFS[0], _FAST_PREFS[2]))
    for lo in range(2):
        hi = lo + 1
        flo, fhi = _FAST_PREFS[lo], _FAST_PREFS[hi]
        if flo <= t <= fhi:
            denom = fhi - flo
            w_hi = (t - flo) / denom
            p = [0.0, 0.0, 0.0]
            p[lo] = 1.0 - w_hi
            p[hi] = w_hi
            return _normalize_probs((p[0], p[1], p[2]))
    return _DEFAULT_DRIVER_PROBS


def fit_from_charging_events(df: pd.DataFrame) -> CalibrationProfile:
    if df.empty:
        return CalibrationProfile.defaults()
    dc = float(df["is_dc_fast"].astype(bool).mean())
    probs = driver_probs_for_dc_share(dc)
    med_e = float(df["energy_kwh"].astype(float).median())
    med_km = float(np.clip(med_e / _NOMINAL_KWH_PER_KM, 18.0, 180.0))
    weekend_km = float(np.clip(med_km * (30.0 / 45.0), 10.0, 150.0))
    med_soc = float(np.clip(df["start_soc"].astype(float).median(), 0.12, 0.85))
    return CalibrationProfile(
        driver_profile_probs=probs,
        weekday_km_mean=med_km,
        weekend_km_mean=weekend_km,
        initial_soh_low=0.92,
        initial_soh_high=1.0,
        median_plug_in_soc=med_soc,
    )


def _soh_band_from_telemetry(soh: pd.Series) -> tuple[float, float]:
    s = soh.astype(float)
    lo = float(max(0.85, s.quantile(0.05)))
    hi = float(min(1.0, s.quantile(0.95)))
    if hi <= lo + 0.01:
        hi = min(1.0, lo + 0.02)
    return lo, hi


def fit_from_curated(frame: CuratedFrame) -> CalibrationProfile | None:
    """Return a profile if any signal exists; otherwise ``None``."""
    any_signal = False
    profile = CalibrationProfile.defaults()

    if not frame.charging_events.empty:
        profile = fit_from_charging_events(frame.charging_events)
        any_signal = True

    if not frame.telemetry.empty and "soh" in frame.telemetry.columns:
        lo, hi = _soh_band_from_telemetry(frame.telemetry["soh"])
        profile = replace(profile, initial_soh_low=lo, initial_soh_high=hi)
        any_signal = True

    return profile if any_signal else None
