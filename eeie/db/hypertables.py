"""TimescaleDB hypertable bootstrap.

Run once after `alembic upgrade head` to promote time-series tables to
hypertables. Idempotent: `create_hypertable(..., if_not_exists => TRUE)`.
"""

from __future__ import annotations

from loguru import logger
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DatabaseError

HYPERTABLES: list[tuple[str, str, str]] = [
    # (table, time column, chunk interval)
    ("telemetry", "ts", "7 days"),
    ("weather", "ts", "30 days"),
    ("charging_events", "start_ts", "30 days"),
    ("station_state", "ts", "30 days"),
]


def is_timescaledb(engine: Engine) -> bool:
    """Detect whether the connected database has the TimescaleDB extension."""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
            ).scalar()
            return row == 1
    except DatabaseError:
        return False


def bootstrap_hypertables(engine: Engine) -> None:
    """Promote known tables to hypertables. No-ops if TimescaleDB is absent."""
    if not is_timescaledb(engine):
        logger.info("TimescaleDB extension not present; skipping hypertable bootstrap.")
        return

    with engine.begin() as conn:
        for table, time_col, chunk in HYPERTABLES:
            stmt = text(
                f"""
                SELECT create_hypertable(
                    '{table}', '{time_col}',
                    chunk_time_interval => INTERVAL '{chunk}',
                    if_not_exists => TRUE,
                    migrate_data => TRUE
                );
                """
            )
            try:
                conn.execute(stmt)
                logger.info("Promoted '{}' to hypertable on '{}'.", table, time_col)
            except DatabaseError as exc:
                logger.warning("Hypertable bootstrap for '{}' failed: {}", table, exc)
