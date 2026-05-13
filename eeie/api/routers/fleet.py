"""Fleet-wide aggregate endpoints used by the executive-summary dashboard.

These endpoints intentionally return small, plotting-ready payloads. The UI is
a pure HTTP client and must not touch the database directly.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from eeie.api.deps import db
from eeie.db.models import ChargingEvent, Telemetry, Vehicle

router = APIRouter(prefix="/fleet", tags=["fleet"])


_PEAK_HOURS: set[int] = {7, 8, 9, 17, 18, 19, 20}


@router.get("/stats")
def fleet_stats(session: Session = Depends(db)) -> dict[str, Any]:
    """Return headline fleet KPIs in a single round-trip."""
    vehicles = session.execute(
        select(
            Vehicle.vehicle_id,
            Vehicle.archetype_id,
            Vehicle.driver_profile_id,
            Vehicle.battery_capacity_kwh,
            Vehicle.nominal_efficiency_kwh_per_km,
        )
    ).all()

    n_vehicles = len(vehicles)
    archetype_mix: dict[str, int] = defaultdict(int)
    driver_mix: dict[str, int] = defaultdict(int)
    total_capacity = 0.0
    for v in vehicles:
        archetype_mix[v.archetype_id] += 1
        driver_mix[v.driver_profile_id] += 1
        total_capacity += float(v.battery_capacity_kwh)

    energy_total, cost_total, n_events = session.execute(
        select(
            func.coalesce(func.sum(ChargingEvent.energy_kwh), 0.0),
            func.coalesce(func.sum(ChargingEvent.cost_eur), 0.0),
            func.count(ChargingEvent.event_id),
        )
    ).one()

    peak_kwh = 0.0
    off_peak_kwh = 0.0
    rows = session.execute(select(ChargingEvent.start_ts, ChargingEvent.energy_kwh)).all()
    for r in rows:
        ts: datetime = r.start_ts
        if ts.hour in _PEAK_HOURS:
            peak_kwh += float(r.energy_kwh)
        else:
            off_peak_kwh += float(r.energy_kwh)

    sub = (
        select(Telemetry.vehicle_id, func.max(Telemetry.ts).label("ts"))
        .group_by(Telemetry.vehicle_id)
        .subquery()
    )
    latest_soh = (
        session.execute(
            select(func.avg(Telemetry.soh)).join(
                sub, (Telemetry.vehicle_id == sub.c.vehicle_id) & (Telemetry.ts == sub.c.ts)
            )
        ).scalar()
        or 1.0
    )

    date_range = session.execute(select(func.min(Telemetry.ts), func.max(Telemetry.ts))).one()
    start_ts, end_ts = date_range
    span_days = 0
    if start_ts is not None and end_ts is not None:
        span_days = max((end_ts - start_ts).days, 1)

    total_energy = float(energy_total or 0.0)
    total_cost = float(cost_total or 0.0)
    annual_cost = (total_cost / span_days * 365.0) if span_days > 0 else 0.0
    annual_energy = (total_energy / span_days * 365.0) if span_days > 0 else 0.0
    peak_share = (peak_kwh / total_energy * 100.0) if total_energy > 0 else 0.0

    return {
        "n_vehicles": n_vehicles,
        "n_charging_events": int(n_events or 0),
        "total_battery_capacity_kwh": float(total_capacity),
        "total_energy_kwh": total_energy,
        "total_cost_eur": total_cost,
        "annual_energy_kwh": annual_energy,
        "annual_cost_eur": annual_cost,
        "avg_soh": float(latest_soh),
        "peak_kwh": peak_kwh,
        "off_peak_kwh": off_peak_kwh,
        "peak_share_pct": peak_share,
        "archetype_mix": dict(archetype_mix),
        "driver_mix": dict(driver_mix),
        "period_days": int(span_days),
        "period_start": start_ts.isoformat() if start_ts else None,
        "period_end": end_ts.isoformat() if end_ts else None,
    }


@router.get("/timeseries")
def fleet_timeseries(
    session: Session = Depends(db),
    bucket: str = Query("day", pattern="^(day|week|month)$"),
) -> list[dict[str, Any]]:
    """Daily/weekly/monthly aggregates of charging energy + cost (whole fleet)."""
    rows = session.execute(
        select(
            ChargingEvent.start_ts,
            ChargingEvent.energy_kwh,
            ChargingEvent.cost_eur,
        ).order_by(ChargingEvent.start_ts.asc())
    ).all()

    def _bucket_key(ts: datetime) -> str:
        if bucket == "day":
            return ts.strftime("%Y-%m-%d")
        if bucket == "week":
            iso = ts.isocalendar()
            return f"{iso.year}-W{iso.week:02d}"
        return ts.strftime("%Y-%m")

    agg: dict[str, dict[str, float]] = defaultdict(
        lambda: {"energy_kwh": 0.0, "cost_eur": 0.0, "peak_kwh": 0.0, "off_peak_kwh": 0.0}
    )
    for r in rows:
        key = _bucket_key(r.start_ts)
        bin_ = agg[key]
        bin_["energy_kwh"] += float(r.energy_kwh)
        bin_["cost_eur"] += float(r.cost_eur)
        if r.start_ts.hour in _PEAK_HOURS:
            bin_["peak_kwh"] += float(r.energy_kwh)
        else:
            bin_["off_peak_kwh"] += float(r.energy_kwh)

    out = [{"bucket": key, **vals} for key, vals in sorted(agg.items())]
    return out
