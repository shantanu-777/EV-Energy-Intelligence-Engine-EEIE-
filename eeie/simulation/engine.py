"""Top-level simulation engine.

Combines fleet sampling, driving patterns, weather, and tariffs to produce
hourly vehicle telemetry and charging events for an entire fleet over the
requested horizon. The SOC dynamics are deliberately simple but physically
plausible:

- Driving energy   = km * efficiency_kwh_per_km * temperature_factor
- Charging energy  = charge_power_kw * dt * one_way_efficiency
- SOH degradation  = calendar_aging + cycle_aging * stress(SOC, temp, dc_fast)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC

import numpy as np
import pandas as pd
from loguru import logger

from eeie.config.tariffs import DEFAULT_TARIFF, TariffSchedule
from eeie.simulation.driving import generate_day
from eeie.simulation.fleet import SampledVehicle, sample_fleet
from eeie.simulation.tariffs import materialize_tariff
from eeie.simulation.weather import CENTRAL_EUROPE, generate_weather

_TEMP_COLD_THRESHOLD = 5.0
_TEMP_HOT_THRESHOLD = 30.0
_CHARGING_ONE_WAY_EFF = 0.93
_DC_FAST_THRESHOLD_KW = 50.0
_CYCLE_AGING_PER_FULL_EQ_CYCLE = 4.5e-5
_CALENDAR_AGING_PER_HOUR = 1.0e-7


@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""

    n_vehicles: int = 100
    months: int = 12
    seed: int = 42
    start_ts: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2025-01-01", tz="UTC"))
    tariff: TariffSchedule = field(default_factory=lambda: DEFAULT_TARIFF)
    region_id: str = "central_eu"


@dataclass
class SimulationResult:
    """Output of `simulate_fleet`."""

    vehicles: pd.DataFrame
    telemetry: pd.DataFrame
    weather: pd.DataFrame
    tariffs: pd.DataFrame
    charging_events: pd.DataFrame


def _temperature_factor(ambient_c: np.ndarray) -> np.ndarray:
    """Consumption multiplier as a function of ambient temperature."""
    cold = np.clip(_TEMP_COLD_THRESHOLD - ambient_c, 0.0, None) * 0.020
    hot = np.clip(ambient_c - _TEMP_HOT_THRESHOLD, 0.0, None) * 0.006
    return 1.0 + cold + hot


def _simulate_vehicle(
    vehicle: SampledVehicle,
    *,
    weather: pd.DataFrame,
    tariff_rates: np.ndarray,
    tariff_tiers: np.ndarray,
    start_ts: pd.Timestamp,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a single vehicle. Returns (telemetry, charging_events)."""
    periods = len(weather)
    ts = weather["ts"].to_numpy()
    ambient = weather["temperature_c"].to_numpy()
    temp_factor = _temperature_factor(ambient)

    # Pre-generate per-day driving profiles.
    days = periods // 24
    km_h = np.zeros(periods, dtype=np.float32)
    speed_h = np.zeros(periods, dtype=np.float32)
    aggr_h = np.zeros(periods, dtype=np.int32)
    day_index = pd.date_range(start=start_ts, periods=days, freq="D", tz="UTC")
    for d in range(days):
        is_weekend = day_index[d].dayofweek >= 5
        is_holiday = day_index[d].dayofyear in {1, 359, 360}
        km, sp, ag = generate_day(vehicle, rng, is_weekend=is_weekend, is_holiday=is_holiday)
        slot = slice(d * 24, (d + 1) * 24)
        km_h[slot] = km
        speed_h[slot] = sp
        aggr_h[slot] = ag

    # Allocate state vectors.
    soc = np.zeros(periods, dtype=np.float32)
    energy_consumed = np.zeros(periods, dtype=np.float32)
    energy_charged = np.zeros(periods, dtype=np.float32)
    battery_temp = np.zeros(periods, dtype=np.float32)
    is_charging = np.zeros(periods, dtype=bool)
    is_driving = np.zeros(periods, dtype=bool)
    soh = np.zeros(periods, dtype=np.float32)
    odometer = np.zeros(periods, dtype=np.float32)

    soc_state = float(rng.uniform(0.45, 0.85))
    soh_state = float(vehicle.initial_soh)
    odo_state = float(rng.uniform(2_000, 60_000))
    batt_temp_state = float(ambient[0] + 3.0)
    capacity = vehicle.battery_capacity_kwh

    plug_in_threshold = float(np.clip(rng.normal(0.30, 0.06), 0.10, 0.45))
    target_soc = vehicle.driver.target_soc_preference
    fast_charge_pref = vehicle.driver.fast_charge_preference

    charging_events: list[dict] = []
    cur_event: dict | None = None

    for t in range(periods):
        amb_t = float(ambient[t])
        batt_temp_state += 0.30 * (amb_t - batt_temp_state)
        driving = km_h[t] > 0.5
        is_driving[t] = bool(driving)

        # Close any open event before the drive starts so end_soc reflects
        # the SOC at the moment of unplugging (not after driving).
        if cur_event is not None and driving:
            cur_event["end_ts"] = ts[t]
            cur_event["end_soc"] = soc_state
            avg_power = cur_event["energy_kwh"] / max(cur_event["hours"], 1)
            charging_events.append(
                {
                    "event_id": cur_event["event_id"],
                    "start_ts": cur_event["start_ts"],
                    "end_ts": cur_event["end_ts"],
                    "vehicle_id": cur_event["vehicle_id"],
                    "start_soc": cur_event["start_soc"],
                    "end_soc": cur_event["end_soc"],
                    "energy_kwh": cur_event["energy_kwh"],
                    "avg_power_kw": avg_power,
                    "is_dc_fast": cur_event["is_dc_fast"],
                    "cost_eur": cur_event["cost_eur"],
                }
            )
            cur_event = None

        if driving:
            consumed = float(km_h[t] * vehicle.nominal_efficiency_kwh_per_km * temp_factor[t])
            energy_consumed[t] = consumed
            soc_state -= consumed / max(capacity * soh_state, 1.0)
            soc_state = max(soc_state, 0.0)
            odo_state += float(km_h[t])
            batt_temp_state += 1.5
            # Driving cycle aging.
            depth = consumed / max(capacity * soh_state, 1.0)
            stress = 1.0 + 0.5 * max(0.0, batt_temp_state - 35.0) / 10.0
            soh_state -= _CYCLE_AGING_PER_FULL_EQ_CYCLE * depth * stress

        plug_in = (not driving) and (
            soc_state < plug_in_threshold or (soc_state < target_soc and rng.random() < 0.04)
        )

        if cur_event is None and plug_in:
            use_dc = rng.random() < fast_charge_pref
            max_power = vehicle.max_dc_charge_kw if use_dc else vehicle.max_ac_charge_kw
            cur_event = {
                "event_id": f"{vehicle.vehicle_id}_evt_{len(charging_events):05d}",
                "start_ts": ts[t],
                "vehicle_id": vehicle.vehicle_id,
                "start_soc": soc_state,
                "energy_kwh": 0.0,
                "power_kw": float(max_power),
                "is_dc_fast": bool(use_dc),
                "cost_eur": 0.0,
                "hours": 0,
            }

        if cur_event is not None and soc_state < target_soc and not driving:
            is_charging[t] = True
            # Power tapers above 80% SOC.
            taper = 1.0 if soc_state < 0.80 else max(0.25, 1.0 - (soc_state - 0.80) * 4)
            power = float(cur_event["power_kw"]) * taper
            delivered = power * _CHARGING_ONE_WAY_EFF
            energy_charged[t] = delivered
            soc_state += delivered / max(capacity * soh_state, 1.0)
            soc_state = min(soc_state, 1.0)
            cur_event["energy_kwh"] += delivered
            cur_event["cost_eur"] += delivered * float(tariff_rates[t])
            cur_event["hours"] += 1
            batt_temp_state += 2.5 if cur_event["is_dc_fast"] else 0.8
            # Charging cycle aging is harsher for DC fast.
            dc_factor = 1.7 if cur_event["is_dc_fast"] else 1.0
            soh_state -= _CYCLE_AGING_PER_FULL_EQ_CYCLE * (delivered / capacity) * dc_factor
        elif cur_event is not None and soc_state >= target_soc:
            cur_event["end_ts"] = ts[t]
            cur_event["end_soc"] = soc_state
            avg_power = cur_event["energy_kwh"] / max(cur_event["hours"], 1)
            charging_events.append(
                {
                    "event_id": cur_event["event_id"],
                    "start_ts": cur_event["start_ts"],
                    "end_ts": cur_event["end_ts"],
                    "vehicle_id": cur_event["vehicle_id"],
                    "start_soc": cur_event["start_soc"],
                    "end_soc": cur_event["end_soc"],
                    "energy_kwh": cur_event["energy_kwh"],
                    "avg_power_kw": avg_power,
                    "is_dc_fast": cur_event["is_dc_fast"],
                    "cost_eur": cur_event["cost_eur"],
                }
            )
            cur_event = None

        # Calendar aging on every step.
        soh_state -= _CALENDAR_AGING_PER_HOUR

        soc[t] = soc_state
        battery_temp[t] = batt_temp_state
        soh[t] = soh_state
        odometer[t] = odo_state

    # Close any open event at horizon end.
    if cur_event is not None:
        cur_event["end_ts"] = ts[-1]
        cur_event["end_soc"] = soc_state
        avg_power = cur_event["energy_kwh"] / max(cur_event["hours"], 1)
        charging_events.append(
            {
                "event_id": cur_event["event_id"],
                "start_ts": cur_event["start_ts"],
                "end_ts": cur_event["end_ts"],
                "vehicle_id": cur_event["vehicle_id"],
                "start_soc": cur_event["start_soc"],
                "end_soc": cur_event["end_soc"],
                "energy_kwh": cur_event["energy_kwh"],
                "avg_power_kw": avg_power,
                "is_dc_fast": cur_event["is_dc_fast"],
                "cost_eur": cur_event["cost_eur"],
            }
        )

    tel = pd.DataFrame(
        {
            "ts": ts,
            "vehicle_id": vehicle.vehicle_id,
            "soc": soc,
            "odometer_km": odometer,
            "km_driven_hour": km_h,
            "energy_consumed_kwh": energy_consumed,
            "energy_charged_kwh": energy_charged,
            "battery_temp_c": battery_temp,
            "ambient_temp_c": ambient.astype(np.float32),
            "avg_speed_kmh": speed_h,
            "aggressive_event_count": aggr_h,
            "is_charging": is_charging,
            "is_driving": is_driving,
            "soh": soh,
        }
    )

    events = (
        pd.DataFrame(charging_events)
        if charging_events
        else pd.DataFrame(
            columns=[
                "event_id",
                "start_ts",
                "end_ts",
                "vehicle_id",
                "start_soc",
                "end_soc",
                "energy_kwh",
                "avg_power_kw",
                "is_dc_fast",
                "cost_eur",
            ]
        )
    )
    return tel, events


def simulate_fleet(config: SimulationConfig) -> SimulationResult:
    """Run a full fleet simulation per `config`."""
    fleet = sample_fleet(n_vehicles=config.n_vehicles, seed=config.seed)
    logger.info(
        "Sampled fleet of {} vehicles. Horizon: {} months from {}.",
        config.n_vehicles,
        config.months,
        config.start_ts,
    )

    periods = config.months * 30 * 24
    weather = generate_weather(
        start=config.start_ts,
        periods=periods,
        climate=CENTRAL_EUROPE,
        seed=config.seed + 1,
    )
    tariffs = materialize_tariff(config.tariff, start=config.start_ts, periods=periods)
    tariff_rates = tariffs["rate_eur_per_kwh"].to_numpy()
    tariff_tiers = tariffs["tier"].to_numpy()

    telemetry_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []

    for i, vehicle in enumerate(fleet):
        veh_rng = np.random.default_rng(config.seed + 1000 + i)
        tel, evt = _simulate_vehicle(
            vehicle,
            weather=weather,
            tariff_rates=tariff_rates,
            tariff_tiers=tariff_tiers,
            start_ts=config.start_ts,
            rng=veh_rng,
        )
        telemetry_frames.append(tel)
        if not evt.empty:
            event_frames.append(evt)
        if (i + 1) % 10 == 0 or i == len(fleet) - 1:
            logger.info("Simulated {}/{} vehicles", i + 1, len(fleet))

    telemetry = pd.concat(telemetry_frames, ignore_index=True)
    charging_events = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame(
            columns=[
                "event_id",
                "start_ts",
                "end_ts",
                "vehicle_id",
                "start_soc",
                "end_soc",
                "energy_kwh",
                "avg_power_kw",
                "is_dc_fast",
                "cost_eur",
            ]
        )
    )

    vehicles_df = pd.DataFrame(
        [
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
                "created_at": pd.Timestamp.now(tz=UTC),
            }
            for v in fleet
        ]
    )

    return SimulationResult(
        vehicles=vehicles_df,
        telemetry=telemetry,
        weather=weather,
        tariffs=tariffs,
        charging_events=charging_events,
    )
