"""Add data_source to telemetry and charging_events.

Revision ID: 0003
Revises: 0002
Create Date: 2026-02-14 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "telemetry",
        sa.Column(
            "data_source",
            sa.String(16),
            nullable=False,
            server_default=sa.text("'synthetic'"),
        ),
    )
    op.add_column(
        "charging_events",
        sa.Column(
            "data_source",
            sa.String(16),
            nullable=False,
            server_default=sa.text("'synthetic'"),
        ),
    )
    op.create_index(
        "ix_telemetry_data_source_ts",
        "telemetry",
        ["data_source", "ts"],
    )
    op.create_index(
        "ix_charging_events_data_source_start",
        "charging_events",
        ["data_source", "start_ts"],
    )


def downgrade() -> None:
    op.drop_index("ix_charging_events_data_source_start", table_name="charging_events")
    op.drop_index("ix_telemetry_data_source_ts", table_name="telemetry")
    op.drop_column("charging_events", "data_source")
    op.drop_column("telemetry", "data_source")
