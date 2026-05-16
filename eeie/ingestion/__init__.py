"""Ingestion layer: source-to-DB adapters, pydantic schemas, real-dataset
registry / on-disk resolver, and per-source adapters."""

from eeie.ingestion.adapters import (
    ADAPTERS,
    CuratedFrame,
    available_slugs,
    curated_dir,
    get_adapter,
    validate_curated,
    write_curated,
)
from eeie.ingestion.datasets import (
    REGISTRY,
    DatasetSpec,
    get_dataset,
    list_datasets,
)
from eeie.ingestion.loader import (
    load_charging_events,
    load_simulation_result,
    load_station_state,
    load_telemetry,
    load_vehicles,
    load_weather,
)
from eeie.ingestion.raw import (
    RawDatasetStatus,
    find_csv,
    raw_dir,
    raw_root,
    verify,
    verify_all,
)
from eeie.ingestion.schemas import (
    ChargingEventRecord,
    DataSource,
    StationStateRecord,
    TariffRecord,
    TelemetryRecord,
    VehicleRecord,
    WeatherRecord,
)
from eeie.ingestion.unifier import unify_pipeline

__all__ = [
    "ADAPTERS",
    "REGISTRY",
    "ChargingEventRecord",
    "CuratedFrame",
    "DataSource",
    "DatasetSpec",
    "RawDatasetStatus",
    "StationStateRecord",
    "TariffRecord",
    "TelemetryRecord",
    "VehicleRecord",
    "WeatherRecord",
    "available_slugs",
    "curated_dir",
    "find_csv",
    "get_adapter",
    "get_dataset",
    "list_datasets",
    "load_charging_events",
    "load_simulation_result",
    "load_station_state",
    "load_telemetry",
    "load_vehicles",
    "load_weather",
    "raw_dir",
    "raw_root",
    "unify_pipeline",
    "validate_curated",
    "verify",
    "verify_all",
    "write_curated",
]
