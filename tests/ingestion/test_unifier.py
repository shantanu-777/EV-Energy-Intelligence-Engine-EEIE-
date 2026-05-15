"""Merging curated frames with simulated output."""

from __future__ import annotations

from eeie.ingestion import CuratedFrame, unify_pipeline
from eeie.ingestion.adapters._common import validate_curated


def test_unify_synthetic_only(small_simulation) -> None:
    merged = unify_pipeline([], small_simulation)
    assert merged.slug == "hybrid"
    assert len(merged.telemetry) == len(small_simulation.telemetry)
    assert (merged.telemetry["data_source"] == "synthetic").all()


def test_unify_duplicate_telemetry_prefers_curated_rows(small_simulation) -> None:
    dup = small_simulation.telemetry.iloc[:4].copy()
    dup["data_source"] = "real"
    injected = CuratedFrame(slug="inject", telemetry=dup)
    validate_curated(injected)
    merged = unify_pipeline([injected], small_simulation)

    keyed = merged.telemetry.merge(dup[["ts", "vehicle_id"]], on=["ts", "vehicle_id"])
    assert len(keyed) == len(dup)
    assert keyed["data_source"].tolist() == ["real"] * len(dup)
