"""Vehicle fleet sampling.

Mixes the static archetype catalog with per-vehicle jitter (battery
capacity, baseline efficiency, commuter distance, initial SOH) so the
resulting fleet has realistic heterogeneity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from eeie.config.tariffs import DEFAULT_TARIFF
from eeie.config.vehicles import (
    DRIVER_PROFILES,
    VEHICLE_ARCHETYPES,
    DriverProfile,
    VehicleArchetype,
)
from eeie.simulation.calibration import CalibrationProfile


@dataclass(frozen=True)
class SampledVehicle:
    """A concrete simulated vehicle."""

    vehicle_id: str
    archetype: VehicleArchetype
    driver: DriverProfile
    battery_capacity_kwh: float
    nominal_efficiency_kwh_per_km: float
    max_ac_charge_kw: float
    max_dc_charge_kw: float
    initial_soh: float
    weekday_mean_km: float
    weekend_mean_km: float
    home_lat: float
    home_lon: float
    tariff_id: str


def sample_fleet(
    *, n_vehicles: int, seed: int, calibration: CalibrationProfile | None = None
) -> list[SampledVehicle]:
    """Return a heterogeneous fleet of `n_vehicles` simulated EVs."""
    rng = np.random.default_rng(seed)
    cal = calibration or CalibrationProfile.defaults()
    probs = np.array(cal.driver_profile_probs, dtype=float)
    probs = probs / max(probs.sum(), 1e-9)

    fleet: list[SampledVehicle] = []
    for i in range(n_vehicles):
        archetype = VEHICLE_ARCHETYPES[rng.integers(0, len(VEHICLE_ARCHETYPES))]
        driver = DRIVER_PROFILES[rng.choice(len(DRIVER_PROFILES), p=probs)]

        capacity = float(archetype.battery_capacity_kwh * rng.uniform(0.95, 1.02))
        efficiency = float(
            archetype.nominal_efficiency_kwh_per_km
            * driver.efficiency_multiplier
            * rng.uniform(0.95, 1.10)
        )
        weekday_km = float(
            np.clip(rng.normal(cal.weekday_km_mean, 18.0), 8.0, 140.0)
        )
        weekend_km = float(np.clip(rng.normal(cal.weekend_km_mean, 25.0), 0.0, 220.0))
        initial_soh = float(rng.uniform(cal.initial_soh_low, cal.initial_soh_high))

        fleet.append(
            SampledVehicle(
                vehicle_id=f"veh_{i:04d}",
                archetype=archetype,
                driver=driver,
                battery_capacity_kwh=capacity,
                nominal_efficiency_kwh_per_km=efficiency,
                max_ac_charge_kw=archetype.max_ac_charge_kw,
                max_dc_charge_kw=archetype.max_dc_charge_kw,
                initial_soh=initial_soh,
                weekday_mean_km=weekday_km,
                weekend_mean_km=weekend_km,
                home_lat=float(rng.uniform(45.0, 55.0)),
                home_lon=float(rng.uniform(2.0, 18.0)),
                tariff_id=DEFAULT_TARIFF.tariff_id,
            )
        )
    return fleet


def fleet_to_rows(fleet: list[SampledVehicle]) -> list[dict]:
    """Convert fleet to dict rows suitable for the `vehicles` table."""
    now = datetime.now(tz=UTC)
    return [
        {
            "vehicle_id": v.vehicle_id,
            "archetype_id": v.archetype.archetype_id,
            "driver_profile_id": v.driver.profile_id,
            "battery_capacity_kwh": v.battery_capacity_kwh,
            "nominal_efficiency_kwh_per_km": v.nominal_efficiency_kwh_per_km,
            "max_ac_charge_kw": v.max_ac_charge_kw,
            "max_dc_charge_kw": v.max_dc_charge_kw,
            "initial_soh": v.initial_soh,
            "home_lat": v.home_lat,
            "home_lon": v.home_lon,
            "tariff_id": v.tariff_id,
            "created_at": now,
        }
        for v in fleet
    ]
