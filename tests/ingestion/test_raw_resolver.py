"""Tests for the raw-dataset resolver and verifier."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from eeie.config.settings import get_settings
from eeie.ingestion import raw
from eeie.ingestion.datasets import get_dataset


@pytest.fixture
def isolated_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Redirect EEIE_DATA_DIR at the tmp_path and clear the settings cache."""
    monkeypatch.setenv("EEIE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)] + [",".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_raw_root_uses_settings_data_dir(isolated_data_dir: Path) -> None:
    assert raw.raw_root() == isolated_data_dir / "raw"
    assert raw.raw_dir("charging_patterns") == isolated_data_dir / "raw" / "charging_patterns"


def test_verify_returns_missing_when_directory_absent(isolated_data_dir: Path) -> None:
    status = raw.verify("charging_patterns")
    assert status.directory_exists is False
    assert status.found_files == ()
    assert "ev_charging_patterns.csv" in status.missing_expected
    assert status.primary_csv_path is None
    assert status.ok is False


def test_verify_returns_missing_when_csv_absent(isolated_data_dir: Path) -> None:
    raw.raw_dir("charging_patterns").mkdir(parents=True)
    status = raw.verify("charging_patterns")
    assert status.directory_exists is True
    assert status.found_files == ()
    assert "ev_charging_patterns.csv" in status.missing_expected
    assert status.ok is False


def test_verify_flags_wrong_header(isolated_data_dir: Path) -> None:
    spec = get_dataset("battery_charging")
    csv_path = raw.raw_dir(spec.slug) / spec.primary_csv
    _write_csv(csv_path, header=["foo", "bar"], rows=[["1", "2"]])

    status = raw.verify(spec.slug)
    assert status.directory_exists is True
    assert status.primary_csv_path == csv_path
    assert set(status.missing_sample_columns) == set(spec.sample_columns)
    assert status.ok is False


def test_verify_succeeds_when_header_matches(isolated_data_dir: Path) -> None:
    spec = get_dataset("battery_charging")
    csv_path = raw.raw_dir(spec.slug) / spec.primary_csv
    _write_csv(csv_path, header=list(spec.sample_columns), rows=[["0"] * len(spec.sample_columns)])

    status = raw.verify(spec.slug)
    assert status.ok is True
    assert status.missing_sample_columns == ()
    assert status.missing_expected == ()
    assert status.primary_csv_path == csv_path


def test_verify_all_returns_one_status_per_registered_dataset(isolated_data_dir: Path) -> None:
    statuses = raw.verify_all()
    assert {s.spec.slug for s in statuses} == {
        "battery_charging",
        "charging_patterns",
        "germany_charging",
        "station_availability",
    }
    assert all(s.ok is False for s in statuses)


def test_find_csv_returns_path_when_present(isolated_data_dir: Path) -> None:
    spec = get_dataset("germany_charging")
    csv_path = raw.raw_dir(spec.slug) / spec.primary_csv
    _write_csv(csv_path, header=["a"], rows=[["1"]])

    found = raw.find_csv(spec.slug)
    assert found == csv_path


def test_find_csv_raises_when_missing(isolated_data_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Raw file not found"):
        raw.find_csv("germany_charging")


def test_status_to_lines_is_human_readable(isolated_data_dir: Path) -> None:
    status = raw.verify("charging_patterns")
    lines = status.to_lines()
    assert any(line.startswith("[MISSING] charging_patterns") for line in lines)
    assert any("target tables" in line for line in lines)
