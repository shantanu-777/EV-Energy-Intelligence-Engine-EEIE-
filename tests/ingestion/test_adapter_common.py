"""Unit tests for the shared adapter helpers."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from eeie.config.settings import get_settings
from eeie.ingestion.adapters import _common as common
from eeie.ingestion.adapters._common import CuratedFrame
from eeie.ingestion.schemas import VehicleRecord


@pytest.fixture
def isolated_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    monkeypatch.setenv("EEIE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_usd_to_eur_uses_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EEIE_USD_TO_EUR", "0.5")
    get_settings.cache_clear()
    try:
        assert common.usd_to_eur(10.0) == pytest.approx(5.0)
        series = pd.Series([1.0, 2.0, 4.0])
        assert list(common.usd_to_eur(series)) == [0.5, 1.0, 2.0]
    finally:
        get_settings.cache_clear()


def test_pct_to_fraction_scalar_and_series() -> None:
    assert common.pct_to_fraction(50.0) == pytest.approx(0.5)
    assert common.pct_to_fraction(-10.0) == 0.0
    assert common.pct_to_fraction(150.0) == 1.0
    series = pd.Series([0.0, 50.0, 200.0, -5.0])
    assert list(common.pct_to_fraction(series)) == [0.0, 0.5, 1.0, 0.0]


def test_parse_utc_handles_mixed_tz() -> None:
    s = pd.Series(["2024-01-01 08:00:00", "2024-01-02 10:30:00"])
    parsed = common.parse_utc(s)
    assert str(parsed.dt.tz) == "UTC"
    assert parsed.iloc[0] == pd.Timestamp("2024-01-01 08:00:00", tz="UTC")


def test_project_columns_raises_on_missing() -> None:
    df = pd.DataFrame({"vehicle_id": ["a"]})
    with pytest.raises(ValueError, match="missing columns"):
        common.project_columns(df, VehicleRecord)


def test_project_columns_drops_extras() -> None:
    base = {
        "vehicle_id": "a",
        "archetype_id": "mid_sedan",
        "driver_profile_id": "moderate",
        "battery_capacity_kwh": 60.0,
        "nominal_efficiency_kwh_per_km": 0.18,
        "max_ac_charge_kw": 11.0,
        "max_dc_charge_kw": 150.0,
        "initial_soh": 1.0,
        "home_lat": 0.0,
        "home_lon": 0.0,
        "tariff_id": "real_kaggle",
        "created_at": datetime.now(tz=UTC),
        "extra_column": "should_be_dropped",
    }
    df = pd.DataFrame([base])
    projected = common.project_columns(df, VehicleRecord)
    assert "extra_column" not in projected.columns
    assert list(projected.columns) == list(VehicleRecord.model_fields)


def test_curated_dir_resolves_under_settings(isolated_data_dir: Path) -> None:
    assert common.curated_dir("charging_patterns") == isolated_data_dir / "curated" / "charging_patterns"


def test_write_curated_skips_empty_tables(isolated_data_dir: Path) -> None:
    frame = CuratedFrame(slug="empty")
    written = common.write_curated(frame)
    assert written == {}


def test_write_curated_writes_non_empty_parquet(isolated_data_dir: Path) -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    frame = CuratedFrame(slug="probe", vehicles=df)
    written = common.write_curated(frame)
    assert "vehicles" in written
    out = written["vehicles"]
    assert out.exists()
    assert pd.read_parquet(out).equals(df)
