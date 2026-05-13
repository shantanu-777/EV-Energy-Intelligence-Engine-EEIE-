"""Initial schema: vehicles, telemetry, weather, tariffs, charging_events.

Revision ID: 0001
Revises:
Create Date: 2026-01-01 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Best-effort enable TimescaleDB; safe to no-op on plain Postgres.
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    op.create_table(
        "vehicles",
        sa.Column("vehicle_id", sa.String(64), primary_key=True),
        sa.Column("archetype_id", sa.String(64), nullable=False),
        sa.Column("driver_profile_id", sa.String(64), nullable=False),
        sa.Column("battery_capacity_kwh", sa.Float, nullable=False),
        sa.Column("nominal_efficiency_kwh_per_km", sa.Float, nullable=False),
        sa.Column("max_ac_charge_kw", sa.Float, nullable=False),
        sa.Column("max_dc_charge_kw", sa.Float, nullable=False),
        sa.Column("initial_soh", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("home_lat", sa.Float, nullable=False),
        sa.Column("home_lon", sa.Float, nullable=False),
        sa.Column("tariff_id", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "telemetry",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "vehicle_id",
            sa.String(64),
            sa.ForeignKey("vehicles.vehicle_id"),
            nullable=False,
        ),
        sa.Column("soc", sa.Float, nullable=False),
        sa.Column("odometer_km", sa.Float, nullable=False),
        sa.Column("km_driven_hour", sa.Float, nullable=False, server_default="0"),
        sa.Column("energy_consumed_kwh", sa.Float, nullable=False, server_default="0"),
        sa.Column("energy_charged_kwh", sa.Float, nullable=False, server_default="0"),
        sa.Column("battery_temp_c", sa.Float, nullable=False),
        sa.Column("ambient_temp_c", sa.Float, nullable=False),
        sa.Column("avg_speed_kmh", sa.Float, nullable=False, server_default="0"),
        sa.Column("aggressive_event_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("is_charging", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("is_driving", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("soh", sa.Float, nullable=False, server_default="1.0"),
        sa.PrimaryKeyConstraint("ts", "vehicle_id"),
    )
    op.create_index("ix_telemetry_vehicle_ts", "telemetry", ["vehicle_id", "ts"])

    op.create_table(
        "weather",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("region_id", sa.String(64), nullable=False),
        sa.Column("temperature_c", sa.Float, nullable=False),
        sa.Column("humidity", sa.Float, nullable=False),
        sa.Column("wind_speed_kmh", sa.Float, nullable=False),
        sa.Column("precipitation_mm", sa.Float, nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("ts", "region_id"),
    )

    op.create_table(
        "tariffs",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("tariff_id", sa.String(64), nullable=False),
        sa.Column("rate_eur_per_kwh", sa.Float, nullable=False),
        sa.Column("tier", sa.String(16), nullable=False),
        sa.PrimaryKeyConstraint("ts", "tariff_id"),
    )

    op.create_table(
        "charging_events",
        sa.Column("event_id", sa.String(64), nullable=False),
        sa.Column("start_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "vehicle_id",
            sa.String(64),
            sa.ForeignKey("vehicles.vehicle_id"),
            nullable=False,
        ),
        sa.Column("start_soc", sa.Float, nullable=False),
        sa.Column("end_soc", sa.Float, nullable=False),
        sa.Column("energy_kwh", sa.Float, nullable=False),
        sa.Column("avg_power_kw", sa.Float, nullable=False),
        sa.Column("is_dc_fast", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("cost_eur", sa.Float, nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("event_id", "start_ts"),
    )
    op.create_index("ix_charging_vehicle_start", "charging_events", ["vehicle_id", "start_ts"])


def downgrade() -> None:
    op.drop_index("ix_charging_vehicle_start", table_name="charging_events")
    op.drop_table("charging_events")
    op.drop_table("tariffs")
    op.drop_table("weather")
    op.drop_index("ix_telemetry_vehicle_ts", table_name="telemetry")
    op.drop_table("telemetry")
    op.drop_table("vehicles")
