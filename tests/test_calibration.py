"""Synthetic fleet calibration from curated real-derived tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eeie.ingestion.adapters import CuratedFrame
from eeie.simulation.calibration import (
    CalibrationProfile,
    driver_probs_for_dc_share,
    fit_from_charging_events,
    fit_from_curated,
)
from eeie.simulation.engine import SimulationConfig, simulate_fleet


def test_driver_probs_min_dc_pure_efficient():
    probs = driver_probs_for_dc_share(0.10)
    assert probs[0] == pytest.approx(1.0)
    assert sum(probs) == pytest.approx(1.0)


def test_driver_probs_max_dc_pure_aggressive():
    probs = driver_probs_for_dc_share(0.50)
    assert probs[2] == pytest.approx(1.0)
    assert sum(probs) == pytest.approx(1.0)


def test_fit_from_charging_events_derives_medians():
    df = pd.DataFrame(
        {
            "energy_kwh": [10.0, 20.0, 35.2],
            "start_soc": [0.2, 0.25, 0.44],
            "is_dc_fast": [True, False, False],
        }
    )
    p = fit_from_charging_events(df)
    assert p.driver_profile_probs == pytest.approx(driver_probs_for_dc_share(1 / 3))
    assert np.isfinite(p.weekday_km_mean)
    assert p.weekend_km_mean <= p.weekday_km_mean
    assert p.median_plug_in_soc == pytest.approx(0.25)


def test_fit_from_curated_battery_soh_updates_band_only():
    base = CalibrationProfile.defaults()
    telem = pd.DataFrame({"soh": [0.88, 0.90, 0.94, 0.96] * 5})
    out = fit_from_curated(CuratedFrame(slug="battery_charging", telemetry=telem))
    assert out is not None
    assert out.weekday_km_mean == pytest.approx(base.weekday_km_mean)
    assert out.initial_soh_low < out.initial_soh_high <= 1.0


def test_fit_from_curated_empty_none():
    assert fit_from_curated(CuratedFrame(slug="empty")) is None


def test_simulate_fleet_accepts_calibration(small_simulation):
    ce = small_simulation.charging_events
    if ce.empty:
        pytest.skip("fixture produced no charging events")
    cal = fit_from_curated(CuratedFrame(slug="synth", charging_events=ce.copy()))
    assert cal is not None
    cfg = SimulationConfig(
        n_vehicles=3,
        months=1,
        seed=1,
        start_ts=pd.Timestamp("2025-01-01", tz="UTC"),
        calibration=cal,
    )
    res = simulate_fleet(cfg)
    assert len(res.telemetry) > 0
