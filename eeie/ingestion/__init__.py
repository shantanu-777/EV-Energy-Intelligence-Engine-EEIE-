"""Ingestion layer: source-to-DB adapters and pydantic schemas."""

from eeie.ingestion.loader import (
    load_charging_events,
    load_simulation_result,
    load_telemetry,
    load_vehicles,
    load_weather,
)
from eeie.ingestion.schemas import (
    ChargingEventRecord,
    TariffRecord,
    TelemetryRecord,
    VehicleRecord,
    WeatherRecord,
)

__all__ = [
    "ChargingEventRecord",
    "TariffRecord",
    "TelemetryRecord",
    "VehicleRecord",
    "WeatherRecord",
    "load_charging_events",
    "load_simulation_result",
    "load_telemetry",
    "load_vehicles",
    "load_weather",
]
