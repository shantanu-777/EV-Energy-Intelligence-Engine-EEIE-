"""FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from eeie.db import get_session
from eeie.db.models import Vehicle


def db() -> Iterator[Session]:
    yield from get_session()


def load_vehicle(session: Session, vehicle_id: str) -> Vehicle:
    v = session.execute(
        select(Vehicle).where(Vehicle.vehicle_id == vehicle_id)
    ).scalar_one_or_none()
    if v is None:
        raise HTTPException(status_code=404, detail=f"Unknown vehicle '{vehicle_id}'")
    return v
