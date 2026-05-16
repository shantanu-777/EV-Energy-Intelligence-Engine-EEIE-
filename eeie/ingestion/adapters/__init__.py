"""Dataset-specific CSV → canonical table projections."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from eeie.ingestion.adapters import (
    battery_charging,
    charging_patterns,
    germany_charging,
    station_availability,
)
from eeie.ingestion.adapters._common import (
    CuratedFrame,
    curated_dir,
    validate_curated,
    write_curated,
)

AdapterFn = Callable[[Path], CuratedFrame]

ADAPTERS: dict[str, AdapterFn] = {
    charging_patterns.SLUG: charging_patterns.to_canonical,
    battery_charging.SLUG: battery_charging.to_canonical,
    station_availability.SLUG: station_availability.to_canonical,
    germany_charging.SLUG: germany_charging.to_canonical,
}


def get_adapter(slug: str) -> AdapterFn:
    """Return the adapter callable for `slug`, raising if none is registered."""
    try:
        return ADAPTERS[slug]
    except KeyError as exc:
        known = ", ".join(sorted(ADAPTERS)) or "(none)"
        raise ValueError(
            f"No adapter registered for slug {slug!r}. Available: {known}."
        ) from exc


def available_slugs() -> list[str]:
    """Return the slugs that currently have an adapter implementation."""
    return sorted(ADAPTERS)


__all__ = [
    "ADAPTERS",
    "AdapterFn",
    "CuratedFrame",
    "available_slugs",
    "curated_dir",
    "get_adapter",
    "validate_curated",
    "write_curated",
]
