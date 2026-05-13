"""Helpers to construct ablation scenarios from the simulated dataset."""

from __future__ import annotations

from datetime import timedelta
from random import Random

from sqlalchemy import select
from sqlalchemy.orm import Session

from eeie.config.tariffs import TARIFFS_BY_ID
from eeie.db.models import Telemetry, Vehicle
from eeie.models.optimization import OptimizationInputs


def build_scenarios_from_db(
    session: Session,
    *,
    n: int = 30,
    horizon_hours: int = 24,
    seed: int = 0,
    target_soc: float = 0.80,
) -> list[OptimizationInputs]:
    """Sample N vehicles and build a charging optimization scenario for each.

    Skips vehicles already at or above target SOC (charging not needed) and
    samples a fresh starting SOC for each remaining vehicle so the harness
    actually exercises the "needs charging" decision space.
    """
    rng = Random(seed)
    vehicles = session.execute(select(Vehicle)).scalars().all()
    if not vehicles:
        return []

    sampled = rng.sample(list(vehicles), k=min(n, len(vehicles)))
    scenarios: list[OptimizationInputs] = []

    for vehicle in sampled:
        latest = session.execute(
            select(Telemetry)
            .where(Telemetry.vehicle_id == vehicle.vehicle_id)
            .order_by(Telemetry.ts.desc())
            .limit(1)
        ).scalar_one_or_none()
        if latest is None:
            continue

        # Force a realistic "needs charge" start state. The simulator's
        # last-hour SOC is often already full, which makes the optimizer
        # trivially answer 'do nothing' and the ablation uninformative.
        initial_soc = rng.uniform(0.20, target_soc - 0.05)

        sched = TARIFFS_BY_ID.get(vehicle.tariff_id) or next(iter(TARIFFS_BY_ID.values()))
        rates: list[float] = []
        peaks: list[int] = []
        for h in range(horizon_hours):
            ts = latest.ts + timedelta(hours=h)
            rate, tier = sched.rate_at(int(ts.hour), bool(ts.weekday() >= 5))
            rates.append(float(rate))
            peaks.append(int(tier == "peak"))

        scenarios.append(
            OptimizationInputs(
                horizon_hours=horizon_hours,
                initial_soc=initial_soc,
                target_soc=target_soc,
                soc_min=0.10,
                soc_max=0.95,
                capacity_kwh=float(vehicle.battery_capacity_kwh),
                max_power_kw=float(vehicle.max_ac_charge_kw),
                charging_efficiency=0.93,
                hourly_rates=rates,
                peak_flags=peaks,
                battery_temp_c=[float(latest.battery_temp_c)] * horizon_hours,
            )
        )

    return scenarios
