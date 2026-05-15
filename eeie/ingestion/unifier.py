"""Merge curated adapter outputs with an optional ``SimulationResult``."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from eeie.ingestion.adapters._common import CuratedFrame, validate_curated
from eeie.simulation.engine import SimulationResult


def _concat_dedupe(parts: Sequence[pd.DataFrame], subset: list[str]) -> pd.DataFrame:
    nonempty = [p for p in parts if not p.empty]
    if not nonempty:
        return pd.DataFrame()
    merged = pd.concat(nonempty, ignore_index=True)
    return merged.drop_duplicates(subset=subset, keep="first")


def _pick_weather_tariffs(curated: Sequence[CuratedFrame], synthetic: SimulationResult | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulator weather/tariffs win when merging hybrid data (same region/time grid)."""

    if synthetic is not None and not synthetic.weather.empty:
        return synthetic.weather, synthetic.tariffs
    weather = pd.DataFrame()
    tariffs = pd.DataFrame()
    for c in curated:
        if weather.empty and not c.weather.empty:
            weather = c.weather
        if tariffs.empty and not c.tariffs.empty:
            tariffs = c.tariffs
    return weather, tariffs


def unify_pipeline(
    curated: Sequence[CuratedFrame],
    synthetic: SimulationResult | None = None,
    *,
    slug: str = "hybrid",
    prefer_real_rows: bool = True,
) -> CuratedFrame:
    """Concatenate overlapping tables.

    Telemetry and charging-events rows are keyed by their natural primary
    keys; when duplicates appear, ordering is ``[*curated_flat, synth]``
    unless ``prefer_real_rows`` is false (reverse). Weather and tariffs come
    from the simulator whenever it is supplied so timestamps stay aligned
    with the synthetic grid.
    """
    curated = tuple(curated)
    parts_tel_cur = [c.telemetry for c in curated]
    syn_tel = synthetic.telemetry if synthetic is not None else pd.DataFrame()
    telemetry = _concat_dedupe(
        [*parts_tel_cur, syn_tel] if prefer_real_rows else [syn_tel, *parts_tel_cur],
        ["ts", "vehicle_id"],
    )

    parts_ev_cur = [c.charging_events for c in curated]
    syn_ev = synthetic.charging_events if synthetic is not None else pd.DataFrame()
    charging_events = _concat_dedupe(
        [*parts_ev_cur, syn_ev] if prefer_real_rows else [syn_ev, *parts_ev_cur],
        ["event_id", "start_ts"],
    )

    parts_veh_cur = [c.vehicles for c in curated]
    syn_v = synthetic.vehicles if synthetic is not None else pd.DataFrame()
    vehicles = _concat_dedupe(
        [*parts_veh_cur, syn_v] if prefer_real_rows else [syn_v, *parts_veh_cur],
        ["vehicle_id"],
    )

    parts_st_cur = [c.station_state for c in curated]
    station_state = _concat_dedupe(parts_st_cur, ["ts", "station_id"])
    weather, tariffs = _pick_weather_tariffs(curated, synthetic)

    frame = CuratedFrame(
        slug=slug,
        vehicles=vehicles,
        telemetry=telemetry,
        charging_events=charging_events,
        weather=weather,
        tariffs=tariffs,
        station_state=station_state,
    )
    validate_curated(frame)
    return frame
