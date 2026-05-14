"""Adapter tests for the Electric Vehicle Charging Patterns dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from eeie.config.settings import get_settings
from eeie.ingestion.adapters import charging_patterns
from eeie.ingestion.adapters._common import REAL_TARIFF_ID

SAMPLE_CSV = """\
User ID,Vehicle Model,Battery Capacity (kWh),Charging Station ID,Charging Station Location,Charging Start Time,Charging End Time,Energy Consumed (kWh),Charging Duration (hours),Charging Rate (kW),Charging Cost (USD),Time of Day,Day of Week,State of Charge (Start %),State of Charge (End %),Distance Driven (since last charge) (km),Temperature (°C),Vehicle Age (years),Charger Type,User Type
User_1,BMW i3,42.0,Station_001,Berlin,2024-01-01 08:00:00,2024-01-01 09:00:00,15.0,1.0,15.0,5.0,Morning,Monday,20.0,55.0,100.0,5.0,2.0,Level 2,Commuter
User_1,BMW i3,42.0,Station_002,Berlin,2024-01-02 18:00:00,2024-01-02 18:30:00,12.0,0.5,40.0,8.0,Evening,Tuesday,30.0,60.0,80.0,3.0,2.0,DC Fast Charger,Commuter
User_2,Tesla Model 3,75.0,Station_003,Paris,2024-01-01 12:00:00,2024-01-01 12:45:00,30.0,0.75,50.0,12.0,Afternoon,Monday,15.0,55.0,200.0,10.0,1.0,DC Fast Charger,Long-Distance Traveler
User_3,Hyundai Kona,64.0,Station_004,Madrid,2024-01-03 22:00:00,2024-01-04 04:00:00,40.0,6.0,7.0,9.5,Night,Wednesday,10.0,80.0,250.0,8.0,3.0,Level 1,Casual Driver
User_3,Hyundai Kona,64.0,Station_005,Madrid,2024-01-05 21:00:00,2024-01-05 23:00:00,18.0,2.0,9.0,4.25,Night,Friday,40.0,70.0,90.0,6.0,3.0,Level 2,Casual Driver
"""


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    path = tmp_path / "sample.csv"
    path.write_text(SAMPLE_CSV, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _force_known_fx(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EEIE_USD_TO_EUR", "0.5")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_round_trip_produces_one_vehicle_per_user(sample_csv: Path) -> None:
    frame = charging_patterns.to_canonical(sample_csv)
    assert sorted(frame.vehicles["vehicle_id"]) == ["cp_User_1", "cp_User_2", "cp_User_3"]


def test_charging_events_have_canonical_columns_and_dtypes(sample_csv: Path) -> None:
    frame = charging_patterns.to_canonical(sample_csv)
    events = frame.charging_events

    assert len(events) == 5
    assert events["start_ts"].dt.tz is not None
    assert events["end_ts"].dt.tz is not None
    assert events["is_dc_fast"].dtype == bool

    assert (events["start_soc"] >= 0).all() and (events["start_soc"] <= 1).all()
    assert (events["end_soc"] >= 0).all() and (events["end_soc"] <= 1).all()
    assert (events["cost_eur"] >= 0).all()


def test_usd_is_converted_to_eur(sample_csv: Path) -> None:
    frame = charging_patterns.to_canonical(sample_csv)
    first = frame.charging_events.iloc[0]
    assert first["cost_eur"] == pytest.approx(5.0 * 0.5)


def test_charger_type_drives_dc_fast_flag(sample_csv: Path) -> None:
    frame = charging_patterns.to_canonical(sample_csv)
    by_id = frame.charging_events.set_index("event_id")
    assert by_id.loc["cp_00000", "is_dc_fast"] is False or bool(
        by_id.loc["cp_00000", "is_dc_fast"]
    ) is False
    assert bool(by_id.loc["cp_00001", "is_dc_fast"]) is True
    assert bool(by_id.loc["cp_00002", "is_dc_fast"]) is True


def test_vehicle_archetype_buckets_by_capacity(sample_csv: Path) -> None:
    frame = charging_patterns.to_canonical(sample_csv)
    by_id = frame.vehicles.set_index("vehicle_id")
    assert by_id.loc["cp_User_1", "archetype_id"] == "city_compact"
    assert by_id.loc["cp_User_2", "archetype_id"] == "long_range_suv"
    assert by_id.loc["cp_User_3", "archetype_id"] == "mid_sedan"


def test_user_type_maps_to_driver_profile(sample_csv: Path) -> None:
    frame = charging_patterns.to_canonical(sample_csv)
    by_id = frame.vehicles.set_index("vehicle_id")
    assert by_id.loc["cp_User_1", "driver_profile_id"] == "moderate"
    assert by_id.loc["cp_User_2", "driver_profile_id"] == "aggressive"
    assert by_id.loc["cp_User_3", "driver_profile_id"] == "efficient"


def test_real_tariff_marker_is_set(sample_csv: Path) -> None:
    frame = charging_patterns.to_canonical(sample_csv)
    assert (frame.vehicles["tariff_id"] == REAL_TARIFF_ID).all()


def test_event_vehicle_ids_are_all_known(sample_csv: Path) -> None:
    frame = charging_patterns.to_canonical(sample_csv)
    assert set(frame.charging_events["vehicle_id"]).issubset(set(frame.vehicles["vehicle_id"]))


def test_telemetry_and_weather_are_empty(sample_csv: Path) -> None:
    frame = charging_patterns.to_canonical(sample_csv)
    assert frame.telemetry.empty
    assert frame.weather.empty
    assert frame.tariffs.empty
