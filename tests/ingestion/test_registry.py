"""Tests for the real-dataset registry."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eeie.ingestion import datasets

EXPECTED_SLUGS = {
    "battery_charging",
    "charging_patterns",
    "germany_charging",
    "station_availability",
}


def test_registry_contains_all_phase1_datasets() -> None:
    assert set(datasets.REGISTRY) == EXPECTED_SLUGS


@pytest.mark.parametrize("slug", sorted(EXPECTED_SLUGS))
def test_spec_is_fully_populated(slug: str) -> None:
    spec = datasets.get_dataset(slug)
    assert spec.slug == slug
    assert spec.title
    assert spec.kaggle_url.startswith("https://www.kaggle.com/datasets/")
    assert spec.primary_csv.endswith(".csv")
    assert spec.primary_csv in spec.expected_files
    assert spec.sample_columns, "sample_columns is used by the verifier; must be non-empty"
    assert spec.target_canonical, "every dataset must declare a target canonical table"
    assert spec.description


def test_spec_is_immutable() -> None:
    spec = datasets.get_dataset("battery_charging")
    with pytest.raises(ValidationError):
        spec.slug = "mutated"  # type: ignore[misc]


def test_get_dataset_raises_on_unknown_slug() -> None:
    with pytest.raises(ValueError, match="Unknown dataset slug"):
        datasets.get_dataset("does_not_exist")


def test_list_datasets_is_sorted_by_slug() -> None:
    slugs = [s.slug for s in datasets.list_datasets()]
    assert slugs == sorted(EXPECTED_SLUGS)


@pytest.mark.parametrize(
    "slug,must_target",
    [
        ("charging_patterns", "charging_events"),
        ("battery_charging", "telemetry"),
        ("station_availability", "station_state"),
        ("germany_charging", "station_state"),
    ],
)
def test_target_canonical_routing(slug: str, must_target: str) -> None:
    spec = datasets.get_dataset(slug)
    assert must_target in spec.target_canonical


def test_every_registered_slug_has_adapter() -> None:
    from eeie.ingestion.adapters import ADAPTERS

    assert set(ADAPTERS) == EXPECTED_SLUGS
