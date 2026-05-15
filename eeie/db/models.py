"""SQLAlchemy 2.x ORM models for the EEIE schema.

Telemetry, weather, tariffs, charging events, and public charging
snapshots (`station_state`) use their own tables. In production deployments
time-series tables (`telemetry`, `weather`, `charging_events`,
`station_state`) are TimescaleDB hypertables (see
`eeie.db.hypertables.bootstrap_hypertables`); on plain Postgres they stay
ordinary tables so local runs and CI stay lightweight.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base."""


class Vehicle(Base):
    __tablename__ = "vehicles"

    vehicle_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    archetype_id: Mapped[str] = mapped_column(String(64), nullable=False)
    driver_profile_id: Mapped[str] = mapped_column(String(64), nullable=False)
    battery_capacity_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    nominal_efficiency_kwh_per_km: Mapped[float] = mapped_column(Float, nullable=False)
    max_ac_charge_kw: Mapped[float] = mapped_column(Float, nullable=False)
    max_dc_charge_kw: Mapped[float] = mapped_column(Float, nullable=False)
    initial_soh: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    home_lat: Mapped[float] = mapped_column(Float, nullable=False)
    home_lon: Mapped[float] = mapped_column(Float, nullable=False)
    tariff_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    telemetry: Mapped[list[Telemetry]] = relationship(back_populates="vehicle")
    charging_events: Mapped[list[ChargingEvent]] = relationship(back_populates="vehicle")


class Telemetry(Base):
    """Hourly vehicle state. Hypertable on `ts` in TimescaleDB."""

    __tablename__ = "telemetry"

    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    vehicle_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("vehicles.vehicle_id"), primary_key=True
    )
    soc: Mapped[float] = mapped_column(Float, nullable=False)
    odometer_km: Mapped[float] = mapped_column(Float, nullable=False)
    km_driven_hour: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    energy_consumed_kwh: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    energy_charged_kwh: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    battery_temp_c: Mapped[float] = mapped_column(Float, nullable=False)
    ambient_temp_c: Mapped[float] = mapped_column(Float, nullable=False)
    avg_speed_kmh: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    aggressive_event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_charging: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_driving: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    soh: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    vehicle: Mapped[Vehicle] = relationship(back_populates="telemetry")

    __table_args__ = (Index("ix_telemetry_vehicle_ts", "vehicle_id", "ts"),)


class Weather(Base):
    """Per-location hourly weather. Hypertable on `ts`."""

    __tablename__ = "weather"

    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    region_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    temperature_c: Mapped[float] = mapped_column(Float, nullable=False)
    humidity: Mapped[float] = mapped_column(Float, nullable=False)
    wind_speed_kmh: Mapped[float] = mapped_column(Float, nullable=False)
    precipitation_mm: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class Tariff(Base):
    """Materialized hourly tariff. Not a hypertable (small)."""

    __tablename__ = "tariffs"

    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    tariff_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    rate_eur_per_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    tier: Mapped[str] = mapped_column(String(16), nullable=False)


class ChargingEvent(Base):
    """A discrete charging session. Hypertable on `start_ts`."""

    __tablename__ = "charging_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    start_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    end_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    vehicle_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("vehicles.vehicle_id"), nullable=False
    )
    start_soc: Mapped[float] = mapped_column(Float, nullable=False)
    end_soc: Mapped[float] = mapped_column(Float, nullable=False)
    energy_kwh: Mapped[float] = mapped_column(Float, nullable=False)
    avg_power_kw: Mapped[float] = mapped_column(Float, nullable=False)
    is_dc_fast: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    cost_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    vehicle: Mapped[Vehicle] = relationship(back_populates="charging_events")

    __table_args__ = (Index("ix_charging_vehicle_start", "vehicle_id", "start_ts"),)


class StationState(Base):
    """Public charging point availability or catalog row (time + station)."""

    __tablename__ = "station_state"

    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    station_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    region_id: Mapped[str] = mapped_column(String(64), nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    n_connectors: Mapped[int] = mapped_column(Integer, nullable=False)
    n_available: Mapped[int] = mapped_column(Integer, nullable=False)
    is_dc_fast: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    operator: Mapped[str] = mapped_column(String(256), nullable=False)
    tariff_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (Index("ix_station_state_region_ts", "region_id", "ts"),)
