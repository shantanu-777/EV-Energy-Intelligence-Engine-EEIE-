"""station_state: regional charging infrastructure snapshots.

Revision ID: 0002
Revises: 0001
Create Date: 2026-02-01 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "station_state",
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("station_id", sa.String(128), nullable=False),
        sa.Column("region_id", sa.String(64), nullable=False),
        sa.Column("lat", sa.Float, nullable=False),
        sa.Column("lon", sa.Float, nullable=False),
        sa.Column("n_connectors", sa.Integer, nullable=False),
        sa.Column("n_available", sa.Integer, nullable=False),
        sa.Column("is_dc_fast", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("operator", sa.String(256), nullable=False),
        sa.Column("tariff_id", sa.String(64), nullable=True),
        sa.PrimaryKeyConstraint("ts", "station_id"),
    )
    op.create_index(
        "ix_station_state_region_ts",
        "station_state",
        ["region_id", "ts"],
    )


def downgrade() -> None:
    op.drop_index("ix_station_state_region_ts", table_name="station_state")
    op.drop_table("station_state")
