"""US station availability tracking CSV → station_state time series."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from eeie.ingestion.adapters._common import (
    CuratedFrame,
    parse_utc,
    project_columns,
    validate_curated,
)
from eeie.ingestion.schemas import StationStateRecord

SLUG = "station_availability"
REGION = "US"


def _build_station_state(raw: pd.DataFrame) -> pd.DataFrame:
    ts = parse_utc(raw["timestamp"])
    sid = raw["station_id"].astype(str).str.strip().str.slice(0, 128)
    lat = pd.to_numeric(raw["latitude"], errors="coerce")
    lon = pd.to_numeric(raw["longitude"], errors="coerce")
    n_conn = (
        pd.to_numeric(raw["ports_total"], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(lower=0)
    )
    n_avail = (
        pd.to_numeric(raw["ports_available"], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(lower=0)
    )
    n_avail = pd.concat([n_avail, n_conn], axis=1).min(axis=1).astype(int)

    ch = raw["charger_type"].fillna("").astype(str).str.lower()
    is_dc = (ch.str.contains("dc fast", regex=False) | ch.str.contains("dc_fast", regex=False)).astype(
        bool
    )

    op = raw["network"].fillna(raw["station_name"]).fillna("").astype(str).str.strip().str.slice(0, 256)
    op = op.mask(op == "", "unknown")

    df = pd.DataFrame(
        {
            "ts": ts,
            "station_id": sid,
            "region_id": REGION,
            "lat": lat,
            "lon": lon,
            "n_connectors": n_conn,
            "n_available": n_avail,
            "is_dc_fast": is_dc,
            "operator": op,
            "tariff_id": pd.Series([None] * len(raw), dtype=object),
        }
    )
    req = ["ts", "station_id", "lat", "lon"]
    return df.dropna(subset=req).reset_index(drop=True)


def to_canonical(raw_path: Path) -> CuratedFrame:
    logger.info("Reading {} from {}", SLUG, raw_path)
    raw = pd.read_csv(raw_path)
    states = project_columns(_build_station_state(raw), StationStateRecord)
    frame = CuratedFrame(slug=SLUG, station_state=states)
    validate_curated(frame)
    logger.info("{} curated: {}", SLUG, frame.summary())
    return frame
