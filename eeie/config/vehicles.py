"""OEM-agnostic generic EV archetypes used by the simulator.

Each archetype is a coarse profile of battery capacity, baseline efficiency,
and charging limits. The simulator samples vehicles by mixing archetypes
with per-vehicle noise so the resulting fleet is heterogeneous but anchored
to plausible real-world numbers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class VehicleArchetype(BaseModel):
    """Static profile of a generic EV class."""

    archetype_id: str
    label: str
    battery_capacity_kwh: float = Field(gt=0)
    nominal_efficiency_kwh_per_km: float = Field(gt=0)
    max_ac_charge_kw: float = Field(gt=0)
    max_dc_charge_kw: float = Field(gt=0)
    soc_min_recommended: float = Field(ge=0, le=1, default=0.10)
    soc_max_recommended: float = Field(ge=0, le=1, default=0.90)


VEHICLE_ARCHETYPES: list[VehicleArchetype] = [
    VehicleArchetype(
        archetype_id="city_compact",
        label="City compact (~40 kWh)",
        battery_capacity_kwh=40.0,
        nominal_efficiency_kwh_per_km=0.155,
        max_ac_charge_kw=7.4,
        max_dc_charge_kw=50.0,
    ),
    VehicleArchetype(
        archetype_id="mid_sedan",
        label="Mid sedan (~60 kWh)",
        battery_capacity_kwh=60.0,
        nominal_efficiency_kwh_per_km=0.175,
        max_ac_charge_kw=11.0,
        max_dc_charge_kw=120.0,
    ),
    VehicleArchetype(
        archetype_id="long_range_suv",
        label="Long-range SUV (~85 kWh)",
        battery_capacity_kwh=85.0,
        nominal_efficiency_kwh_per_km=0.210,
        max_ac_charge_kw=11.0,
        max_dc_charge_kw=150.0,
    ),
    VehicleArchetype(
        archetype_id="performance",
        label="Performance (~95 kWh)",
        battery_capacity_kwh=95.0,
        nominal_efficiency_kwh_per_km=0.230,
        max_ac_charge_kw=22.0,
        max_dc_charge_kw=250.0,
    ),
]


ARCHETYPES_BY_ID: dict[str, VehicleArchetype] = {a.archetype_id: a for a in VEHICLE_ARCHETYPES}


class DriverProfile(BaseModel):
    """Driver behavior class. Modulates consumption and charging style."""

    profile_id: str
    label: str
    efficiency_multiplier: float = Field(gt=0, description=">1 = wastes energy")
    aggressive_acceleration_prob: float = Field(ge=0, le=1)
    fast_charge_preference: float = Field(ge=0, le=1)
    target_soc_preference: float = Field(ge=0, le=1)


DRIVER_PROFILES: list[DriverProfile] = [
    DriverProfile(
        profile_id="efficient",
        label="Efficient",
        efficiency_multiplier=0.92,
        aggressive_acceleration_prob=0.05,
        fast_charge_preference=0.10,
        target_soc_preference=0.80,
    ),
    DriverProfile(
        profile_id="moderate",
        label="Moderate",
        efficiency_multiplier=1.00,
        aggressive_acceleration_prob=0.20,
        fast_charge_preference=0.25,
        target_soc_preference=0.85,
    ),
    DriverProfile(
        profile_id="aggressive",
        label="Aggressive",
        efficiency_multiplier=1.15,
        aggressive_acceleration_prob=0.55,
        fast_charge_preference=0.50,
        target_soc_preference=0.95,
    ),
]


PROFILES_BY_ID: dict[str, DriverProfile] = {p.profile_id: p for p in DRIVER_PROFILES}
