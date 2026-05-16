"""Adapter tests for the US station availability tracking CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eeie.ingestion.adapters.station_availability import REGION, to_canonical

HDR = """\
timestamp,station_id,station_name,network,city,state,latitude,longitude,location_type,charger_type,power_output_kw,amenities_nearby,ports_total,ports_available,ports_occupied,ports_out_of_service,utilization_rate,station_status,estimated_wait_time_mins,avg_session_duration_mins,current_price,pricing_type,temperature_f,precipitation_mm,weather_condition,gas_price_per_gallon,traffic_congestion_index,local_event,is_weekend,is_peak_hour,hour_of_day,day_of_week,month
"""


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    p = tmp_path / "ev_charging_station_data.csv"
    p.write_text(
        HDR
        + "2025-07-01 00:00:00,EVTEST1,Name1,N1,Los Angeles,CA,34.0,-118.25,Mall,Level 2,11.5,,8,8,0,0,0.1,online,5,35,0.24,per_kwh,82,0,humid,5,3,none,false,false,14,7,11\n"
        + "2025-07-01 01:00:00,EVTEST1,Name1,N1,Los Angeles,CA,34.0,-118.25,Mall,DC Fast Charge,90.0,,8,2,6,0,0.74,online,10,42,0.35,per_kwh,83,0,clear,5,5,none,false,true,22,12,13\n",
        encoding="utf-8",
    )
    return p


def test_availability_only_station_state(sample_csv: Path) -> None:
    frame = to_canonical(sample_csv)
    s = frame.summary()
    assert s["station_state"] == 2
    assert s["vehicles"] == 0


def test_region_and_hourly_resolution(sample_csv: Path) -> None:
    frame = to_canonical(sample_csv)
    st = frame.station_state
    assert (st["region_id"] == REGION).all()
    t0, t1 = st["ts"].iloc[0], st["ts"].iloc[1]
    assert getattr(t0, "tzinfo", None) is not None
    assert (t1 - t0) == pd.Timedelta(hours=1)


def test_ports_clipped(sample_csv: Path) -> None:
    frame = to_canonical(sample_csv)
    st = frame.station_state.iloc[1]
    assert int(st["n_connectors"]) == 8
    assert int(st["n_available"]) == 2


def test_dc_fast_from_charger_label(sample_csv: Path) -> None:
    frame = to_canonical(sample_csv)
    assert frame.station_state["is_dc_fast"].tolist() == [False, True]


@pytest.fixture
def bad_ports_csv(tmp_path: Path) -> Path:
    p = tmp_path / "bad.csv"
    p.write_text(
        HDR
        + "2025-08-01 12:00:00,EVX,Name1,N1,Los Angeles,CA,34.0,-118.25,Mall,Level 2,11.5,,-1,99,0,0,0.1,online,5,35,0.24,per_kwh,82,0,humid,5,3,none,false,false,14,7,11\n",
        encoding="utf-8",
    )
    return p


def test_ports_non_negative_even_if_raw_negative(bad_ports_csv: Path) -> None:
    frame = to_canonical(bad_ports_csv)
    st = frame.station_state.iloc[0]
    assert int(st["n_connectors"]) == 0
    assert int(st["n_available"]) == 0
