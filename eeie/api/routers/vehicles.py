"""GET /vehicles for the UI to list available simulated vehicles."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from eeie.api.deps import db
from eeie.db.models import Vehicle

router = APIRouter()


@router.get("/vehicles", tags=["catalog"])
def list_vehicles(session: Session = Depends(db)) -> list[dict]:
    rows = session.execute(
        select(
            Vehicle.vehicle_id,
            Vehicle.archetype_id,
            Vehicle.driver_profile_id,
            Vehicle.battery_capacity_kwh,
            Vehicle.tariff_id,
        ).order_by(Vehicle.vehicle_id.asc())
    ).all()
    return [
        {
            "vehicle_id": r.vehicle_id,
            "archetype_id": r.archetype_id,
            "driver_profile_id": r.driver_profile_id,
            "battery_capacity_kwh": float(r.battery_capacity_kwh),
            "tariff_id": r.tariff_id,
        }
        for r in rows
    ]
