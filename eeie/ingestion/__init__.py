"""Ingestion layer: source-to-DB adapters, pydantic schemas, and the
real-dataset registry / on-disk resolver used by Phase 2+ adapters."""

from eeie.ingestion.datasets import (
    REGISTRY,
    DatasetSpec,
    get_dataset,
    list_datasets,
)
from eeie.ingestion.loader import (
    load_charging_events,
    load_simulation_result,
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
    TariffRecord,
    TelemetryRecord,
    VehicleRecord,
    WeatherRecord,
)

__all__ = [
    "REGISTRY",
    "ChargingEventRecord",
    "DatasetSpec",
    "RawDatasetStatus",
    "TariffRecord",
    "TelemetryRecord",
    "VehicleRecord",
    "WeatherRecord",
    "find_csv",
    "get_dataset",
    "list_datasets",
    "load_charging_events",
    "load_simulation_result",
    "load_telemetry",
    "load_vehicles",
    "load_weather",
    "raw_dir",
    "raw_root",
    "verify",
    "verify_all",
]
