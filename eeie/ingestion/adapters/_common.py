"""Helpers shared by ingestion adapters (currency, parquet, pydantic validation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import overload

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from eeie.config.settings import get_settings
from eeie.ingestion.schemas import (
    ChargingEventRecord,
    StationStateRecord,
    TariffRecord,
    TelemetryRecord,
    VehicleRecord,
    WeatherRecord,
)

REAL_TARIFF_ID = "real_kaggle"

DEFAULT_HOME_LAT = 0.0
DEFAULT_HOME_LON = 0.0


@dataclass(frozen=True)
class CuratedFrame:
    """One raw source mapped onto the simulator table shapes."""

    slug: str
    vehicles: pd.DataFrame = field(default_factory=pd.DataFrame)
    telemetry: pd.DataFrame = field(default_factory=pd.DataFrame)
    charging_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    weather: pd.DataFrame = field(default_factory=pd.DataFrame)
    tariffs: pd.DataFrame = field(default_factory=pd.DataFrame)
    station_state: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> dict[str, int]:
        return {
            "vehicles": len(self.vehicles),
            "telemetry": len(self.telemetry),
            "charging_events": len(self.charging_events),
            "weather": len(self.weather),
            "tariffs": len(self.tariffs),
            "station_state": len(self.station_state),
        }


_SCHEMA_BY_TABLE: dict[str, type[BaseModel]] = {
    "vehicles": VehicleRecord,
    "telemetry": TelemetryRecord,
    "charging_events": ChargingEventRecord,
    "weather": WeatherRecord,
    "tariffs": TariffRecord,
    "station_state": StationStateRecord,
}


@overload
def usd_to_eur(amount: float) -> float: ...
@overload
def usd_to_eur(amount: pd.Series) -> pd.Series: ...
def usd_to_eur(amount: float | pd.Series) -> float | pd.Series:
    """Convert a USD value (or column) to EUR using the configured rate."""
    return amount * get_settings().usd_to_eur


@overload
def pct_to_fraction(value: float) -> float: ...
@overload
def pct_to_fraction(value: pd.Series) -> pd.Series: ...
def pct_to_fraction(value: float | pd.Series) -> float | pd.Series:
    """Percent (0 through 100) to fraction clipped to ``[0, 1]``."""
    if isinstance(value, pd.Series):
        return (value / 100.0).clip(lower=0.0, upper=1.0)
    return max(0.0, min(1.0, float(value) / 100.0))


def parse_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def project_columns(df: pd.DataFrame, schema: type[BaseModel]) -> pd.DataFrame:
    """Reorder / subset ``df`` to match ``schema`` field order exactly."""
    expected = list(schema.model_fields)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"Adapter output missing columns for {schema.__name__}: {missing}"
        )
    return df[expected].copy()


def validate_records(df: pd.DataFrame, schema: type[BaseModel]) -> None:
    """Raise if any row does not satisfy ``schema``."""
    if df.empty:
        return

    def _cell(v):
        if v is None:
            return None
        try:
            if pd.api.types.is_scalar(v) and pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        return v

    for record in df.to_dict(orient="records"):
        cleaned = {k: _cell(v) for k, v in record.items()}
        schema.model_validate(cleaned)


def validate_curated(frame: CuratedFrame) -> None:
    for table, schema in _SCHEMA_BY_TABLE.items():
        validate_records(getattr(frame, table), schema)


def curated_dir(slug: str) -> Path:
    return get_settings().data_dir / "curated" / slug


def write_curated(frame: CuratedFrame, *, output_dir: Path | None = None) -> dict[str, Path]:
    target = output_dir or curated_dir(frame.slug)
    target.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for table in _SCHEMA_BY_TABLE:
        df = getattr(frame, table)
        if df.empty:
            continue
        path = target / f"{table}.parquet"
        df.to_parquet(path, index=False)
        written[table] = path
        logger.info("Wrote {} rows to {}", len(df), path)
    return written
