"""Pydantic schemas mirroring the SQLAlchemy ORM models.

Used at adapter boundaries (e.g., a real-vehicle telemetry adapter would
validate its payloads against `TelemetryRecord` before persisting).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class VehicleRecord(_Base):
    vehicle_id: str
    archetype_id: str
    driver_profile_id: str
    battery_capacity_kwh: float = Field(gt=0)
    nominal_efficiency_kwh_per_km: float = Field(gt=0)
    max_ac_charge_kw: float = Field(gt=0)
    max_dc_charge_kw: float = Field(gt=0)
    initial_soh: float = Field(ge=0, le=1)
    home_lat: float
    home_lon: float
    tariff_id: str
    created_at: datetime


class TelemetryRecord(_Base):
    ts: datetime
    vehicle_id: str
    soc: float = Field(ge=0, le=1)
    odometer_km: float = Field(ge=0)
    km_driven_hour: float = Field(ge=0)
    energy_consumed_kwh: float = Field(ge=0)
    energy_charged_kwh: float = Field(ge=0)
    battery_temp_c: float
    ambient_temp_c: float
    avg_speed_kmh: float = Field(ge=0)
    aggressive_event_count: int = Field(ge=0)
    is_charging: bool
    is_driving: bool
    soh: float = Field(ge=0, le=1)


class WeatherRecord(_Base):
    ts: datetime
    region_id: str
    temperature_c: float
    humidity: float = Field(ge=0, le=1)
    wind_speed_kmh: float = Field(ge=0)
    precipitation_mm: float = Field(ge=0)


class TariffRecord(_Base):
    ts: datetime
    tariff_id: str
    rate_eur_per_kwh: float = Field(ge=0)
    tier: str


class ChargingEventRecord(_Base):
    event_id: str
    start_ts: datetime
    end_ts: datetime
    vehicle_id: str
    start_soc: float = Field(ge=0, le=1)
    end_soc: float = Field(ge=0, le=1)
    energy_kwh: float = Field(ge=0)
    avg_power_kw: float = Field(ge=0)
    is_dc_fast: bool
    cost_eur: float = Field(ge=0)


class StationStateRecord(_Base):
    ts: datetime
    station_id: str
    region_id: str
    lat: float
    lon: float
    n_connectors: int = Field(ge=0)
    n_available: int = Field(ge=0)
    is_dc_fast: bool
    operator: str
    tariff_id: str | None = None
