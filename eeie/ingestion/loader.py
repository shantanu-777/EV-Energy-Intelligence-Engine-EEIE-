"""DataFrame-to-DB bulk loaders.

These adapters are reused by the simulator and by any future real-data
adapter. They issue INSERT ... ON CONFLICT DO NOTHING in chunks, so reruns
on the same horizon are idempotent.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger
from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from eeie.db.models import (
    ChargingEvent,
    Tariff,
    Telemetry,
    Vehicle,
    Weather,
)

if TYPE_CHECKING:
    from eeie.simulation.engine import SimulationResult


# Postgres caps bind parameters at 65535 per statement. Our widest row
# (telemetry) has ~14 columns, so 1000 rows = ~14k params -- well under the
# limit and gives a comfortable safety margin for future columns.
_CHUNK = 1_000


def _records(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")


def _bulk_insert(
    session: Session, table_class, rows: Iterable[dict], conflict_cols: list[str]
) -> int:
    """Insert with `ON CONFLICT DO NOTHING` if running on Postgres."""
    rows = list(rows)
    if not rows:
        return 0

    dialect = session.bind.dialect.name if session.bind is not None else ""

    def _pg_stmt(chunk: list[dict]):
        return (
            pg_insert(table_class.__table__)
            .values(chunk)
            .on_conflict_do_nothing(index_elements=conflict_cols)
        )

    def _generic_stmt(chunk: list[dict]):
        return insert(table_class.__table__).values(chunk)

    make_stmt = _pg_stmt if dialect == "postgresql" else _generic_stmt

    total = 0
    for i in range(0, len(rows), _CHUNK):
        chunk = rows[i : i + _CHUNK]
        session.execute(make_stmt(chunk))
        total += len(chunk)
    return total


def load_vehicles(session: Session, df: pd.DataFrame) -> int:
    n = _bulk_insert(session, Vehicle, _records(df), ["vehicle_id"])
    logger.info("Inserted {} vehicle rows.", n)
    return n


def load_telemetry(session: Session, df: pd.DataFrame) -> int:
    n = _bulk_insert(session, Telemetry, _records(df), ["ts", "vehicle_id"])
    logger.info("Inserted {} telemetry rows.", n)
    return n


def load_weather(session: Session, df: pd.DataFrame) -> int:
    n = _bulk_insert(session, Weather, _records(df), ["ts", "region_id"])
    logger.info("Inserted {} weather rows.", n)
    return n


def load_tariffs(session: Session, df: pd.DataFrame) -> int:
    n = _bulk_insert(session, Tariff, _records(df), ["ts", "tariff_id"])
    logger.info("Inserted {} tariff rows.", n)
    return n


def load_charging_events(session: Session, df: pd.DataFrame) -> int:
    n = _bulk_insert(session, ChargingEvent, _records(df), ["event_id", "start_ts"])
    logger.info("Inserted {} charging events.", n)
    return n


def load_simulation_result(session: Session, result: SimulationResult) -> None:
    """Persist all five tables of a simulation result."""
    load_vehicles(session, result.vehicles)
    load_weather(session, result.weather)
    load_tariffs(session, result.tariffs)
    load_telemetry(session, result.telemetry)
    load_charging_events(session, result.charging_events)
