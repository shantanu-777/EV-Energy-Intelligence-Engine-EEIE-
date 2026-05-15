"""Bundesnetzagentur-style German charging registry → station_state (static ts)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from eeie.ingestion.adapters._common import CuratedFrame, project_columns, validate_curated
from eeie.ingestion.schemas import StationStateRecord

SLUG = "germany_charging"
REGION = "DE"
STATIC_TS = pd.Timestamp("1970-01-01", tz="UTC")


def _is_dc_fast(raw: pd.DataFrame) -> pd.Series:
    blob = raw["art_der_ladeeinrichtung"].fillna("").astype(str).str.lower()
    for i in range(1, 5):
        col = f"steckertypen{i}"
        if col in raw.columns:
            blob = blob + " " + raw[col].fillna("").astype(str).str.lower()
    return (
        blob.str.contains("schnell", regex=False)
        | blob.str.contains("dc", regex=False)
        | blob.str.contains("combo", regex=False)
    )


def _build_station_state(raw: pd.DataFrame) -> pd.DataFrame:
    key_col = raw.columns[0]
    lat = pd.to_numeric(raw["breitengrad"], errors="coerce")
    lon = pd.to_numeric(
        raw["laengengrad"].astype(str).str.replace(",", ".").str.strip(),
        errors="coerce",
    )
    n_ports = (
        pd.to_numeric(raw["anzahl_ladepunkte"], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(lower=0)
    )
    op = raw["betreiber"].fillna("").astype(str).str.strip().str.slice(0, 256)
    op = op.mask(op == "", "unknown")

    ids = "de_" + raw[key_col].astype(str).str.strip()
    df = pd.DataFrame(
        {
            "ts": STATIC_TS,
            "station_id": ids.str.slice(0, 128),
            "region_id": REGION,
            "lat": lat,
            "lon": lon,
            "n_connectors": n_ports,
            "n_available": n_ports,
            "is_dc_fast": _is_dc_fast(raw).astype(bool),
            "operator": op,
            "tariff_id": pd.Series([None] * len(raw), dtype=object),
        }
    )
    return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def to_canonical(raw_path: Path) -> CuratedFrame:
    logger.info("Reading {} from {}", SLUG, raw_path)
    raw = pd.read_csv(raw_path)
    states = project_columns(_build_station_state(raw), StationStateRecord)
    frame = CuratedFrame(slug=SLUG, station_state=states)
    validate_curated(frame)
    logger.info("{} curated: {}", SLUG, frame.summary())
    return frame
