"""Adapter tests for the EV Battery Charging dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eeie.ingestion.adapters import battery_charging

SAMPLE_CSV = """\
timestamp,SOC,SOH,terminal_voltage,battery_current,battery_temp,ambient_temp,internal_resistance,action_current,action_voltage,dT_dt,dV_dt,soc_delta,thermal_stress_index,aging_indicator,charging_efficiency,charging_time,cycle_degradation,over_temp_flag,over_voltage_flag,balancing_time
0,0.10,0.99,3.0,10.0,25.0,20.0,0.10,10.0,3.0,0.5,0.00,0.05,0.10,0.10,0.90,1200,0.0005,0,0,20.0
1,0.18,0.98,3.1,12.0,26.0,21.0,0.10,12.0,3.1,0.6,0.01,0.08,0.15,0.10,0.91,1300,0.0005,0,0,21.0
2,0.30,0.97,3.2,15.0,27.0,22.0,0.10,15.0,3.2,0.7,0.01,0.12,0.20,0.12,0.92,1400,0.0006,0,0,22.0
3,0.45,0.97,3.3,18.0,28.0,23.0,0.10,18.0,3.3,0.8,0.01,0.15,0.25,0.13,0.93,1500,0.0006,0,0,23.0
4,0.60,0.96,3.4,20.0,30.0,23.0,0.11,20.0,3.4,0.9,0.01,0.15,0.30,0.15,0.93,1600,0.0007,0,0,24.0
5,0.80,0.96,3.5,22.0,32.0,24.0,0.11,22.0,3.5,1.0,0.01,0.20,0.40,0.16,0.94,1700,0.0007,0,0,25.0
6,0.95,0.95,3.6,5.0,33.0,24.0,0.11,5.0,3.6,0.5,0.00,0.15,0.20,0.18,0.94,1800,0.0008,0,0,26.0
"""


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    path = tmp_path / "sample.csv"
    path.write_text(SAMPLE_CSV, encoding="utf-8")
    return path


def test_emits_single_vehicle_with_archetype_and_profile(sample_csv: Path) -> None:
    frame = battery_charging.to_canonical(sample_csv)
    assert len(frame.vehicles) == 1
    row = frame.vehicles.iloc[0]
    assert row["vehicle_id"] == battery_charging.VEHICLE_ID
    assert row["archetype_id"] == battery_charging.ARCHETYPE_ID
    assert row["driver_profile_id"] == battery_charging.DRIVER_PROFILE_ID
    assert row["battery_capacity_kwh"] == battery_charging.ASSUMED_PACK_KWH


def test_telemetry_is_hourly_and_anchored(sample_csv: Path) -> None:
    frame = battery_charging.to_canonical(sample_csv)
    assert len(frame.telemetry) == 7

    ts = frame.telemetry["ts"]
    assert ts.iloc[0] == battery_charging.START_TS
    diffs = ts.diff().dropna().unique()
    assert len(diffs) == 1
    assert diffs[0] == pd.Timedelta(hours=1)


def test_telemetry_values_are_in_range(sample_csv: Path) -> None:
    frame = battery_charging.to_canonical(sample_csv)
    tel = frame.telemetry
    assert (tel["soc"].between(0.0, 1.0)).all()
    assert (tel["soh"].between(0.0, 1.0)).all()
    assert (tel["energy_consumed_kwh"] == 0.0).all()
    assert (tel["km_driven_hour"] == 0.0).all()
    assert (tel["is_driving"] == False).all()  # noqa: E712
    assert (tel["data_source"] == "real").all()


def test_charging_flag_tracks_action_current(sample_csv: Path) -> None:
    frame = battery_charging.to_canonical(sample_csv)
    assert frame.telemetry["is_charging"].all()


def test_energy_charged_scales_with_pack_capacity(sample_csv: Path) -> None:
    frame = battery_charging.to_canonical(sample_csv)
    pack = battery_charging.ASSUMED_PACK_KWH
    expected = [v * pack for v in (0.05, 0.08, 0.12, 0.15, 0.15, 0.20, 0.15)]
    assert list(frame.telemetry["energy_charged_kwh"]) == pytest.approx(expected)


def test_other_tables_are_empty(sample_csv: Path) -> None:
    frame = battery_charging.to_canonical(sample_csv)
    assert frame.charging_events.empty
    assert frame.weather.empty
    assert frame.tariffs.empty
    assert frame.station_state.empty
