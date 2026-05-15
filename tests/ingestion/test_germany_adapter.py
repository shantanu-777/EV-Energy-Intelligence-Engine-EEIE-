"""Adapter tests for the German public charging registry CSV."""

from __future__ import annotations

from pathlib import Path

import pytest

from eeie.ingestion.adapters.germany_charging import REGION, STATIC_TS, to_canonical

HDR = """\
,betreiber,art_der_ladeeinrichtung,anzahl_ladepunkte,anschlussleistung,steckertypen1,steckertypen2,steckertypen3,steckertypen4,p1_kw,p2_kw,p3_kw,p4_kw,kreis_kreisfreie_stadt,ort,postleitzahl,strasse,hausnummer,adresszusatz,inbetriebnahmedatum,breitengrad,laengengrad
"""


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    p = tmp_path / "charging_data.csv"
    p.write_text(
        HDR
        + "9001,OpA,Normalladeeinrichtung,2,22.0,AC Steckdose Typ 2,,,,22.0,,,,Kreis,Burg,99999,Weg,1,,2019-01-01,50.0,9.12\n"
        + "9002,OpB,Schnellladeeinrichtung,4,150.0,DC Kupplung Combo,,,,150.0,,,,Kreis,Burg2,88888,Weg2,3,,2020-02-02,51.05,11.234\n",
        encoding="utf-8",
    )
    return p


def test_germany_only_populates_station_state(sample_csv: Path) -> None:
    frame = to_canonical(sample_csv)
    summary = frame.summary()
    assert summary["station_state"] == 2
    assert summary["vehicles"] == 0
    assert (frame.station_state["region_id"] == REGION).all()
    assert (frame.station_state["ts"] == STATIC_TS).all()
    assert frame.station_state["station_id"].tolist() == ["de_9001", "de_9002"]


def test_germany_dc_flag(sample_csv: Path) -> None:
    frame = to_canonical(sample_csv)
    assert frame.station_state["is_dc_fast"].tolist() == [False, True]


def test_germany_coordinates_parsed(sample_csv: Path) -> None:
    frame = to_canonical(sample_csv)
    lat = frame.station_state["lat"].tolist()
    lon = frame.station_state["lon"].tolist()
    assert lat[0] == pytest.approx(50.0)
    assert lat[1] == pytest.approx(51.05)
    assert lon[0] == pytest.approx(9.12)
    assert lon[1] == pytest.approx(11.234)


def test_germany_station_id_truncation(tmp_path: Path) -> None:
    long_rid = "x" * 200
    p = tmp_path / "charging_data.csv"
    p.write_text(
        HDR + f"{long_rid},Op,Z,1,4.6,Typ2,,,,4.6,,,,K,B,33333,W,9,,2020-01-01,48.5,13.44\n",
        encoding="utf-8",
    )
    frame = to_canonical(p)
    sid = frame.station_state["station_id"].iloc[0]
    assert len(sid) <= 128
    assert sid.startswith("de_")
