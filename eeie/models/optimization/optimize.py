"""Unified charging-optimization surface.

`optimize_charging()` is the single entry point the API uses. It dispatches
to MILP or PPO based on configuration and returns a `ChargingPlan` with the
schedule plus a structured cost / wear / peak decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from loguru import logger

from eeie.config import get_settings
from eeie.models.optimization.env import ChargingEnv, ChargingEnvConfig
from eeie.models.optimization.milp import MILPInputs, solve_charging_milp

PPO_CHECKPOINT = "optimization/ppo.zip"


@dataclass
class OptimizationInputs:
    """Concrete decision inputs for the optimizer."""

    horizon_hours: int
    initial_soc: float
    target_soc: float
    soc_min: float
    soc_max: float
    capacity_kwh: float
    max_power_kw: float
    charging_efficiency: float
    hourly_rates: list[float]
    peak_flags: list[int]
    battery_temp_c: list[float]
    lambda_wear: float = 0.5
    lambda_peak: float = 0.3


@dataclass
class ChargingPlan:
    optimizer: Literal["milp", "rl", "naive"]
    feasible: bool
    power_per_hour: list[float]
    soc_per_hour: list[float]
    total_cost_eur: float
    total_wear_penalty: float
    total_peak_penalty: float
    target_soc: float
    achieved_final_soc: float
    chosen_window_start_hour: int | None
    chosen_window_end_hour: int | None
    chosen_avg_rate_kw: float
    confidence: float
    feature_contributions: dict[str, float] = field(default_factory=dict)


def _summarize(
    power: list[float], soc: list[float], total_cost: float, wear: float, peak: float, target: float
) -> tuple[int | None, int | None, float]:
    active = [i for i, p in enumerate(power) if p > 1e-3]
    if not active:
        return None, None, 0.0
    start, end = active[0], active[-1] + 1
    avg_rate = float(np.mean([p for p in power[start:end] if p > 0]) or 0.0)
    return start, end, avg_rate


def _optimize_milp(inp: OptimizationInputs) -> ChargingPlan:
    milp = MILPInputs(
        horizon_hours=inp.horizon_hours,
        initial_soc=inp.initial_soc,
        target_soc=inp.target_soc,
        soc_min=inp.soc_min,
        soc_max=inp.soc_max,
        capacity_kwh=inp.capacity_kwh,
        max_power_kw=inp.max_power_kw,
        charging_efficiency=inp.charging_efficiency,
        hourly_rates=inp.hourly_rates,
        peak_flags=inp.peak_flags,
        battery_temp_c=inp.battery_temp_c,
        lambda_wear=inp.lambda_wear,
        lambda_peak=inp.lambda_peak,
    )
    res = solve_charging_milp(milp)
    start, end, avg_rate = _summarize(
        res.power_per_hour,
        res.soc_per_hour,
        res.total_cost,
        res.total_wear_penalty,
        res.total_peak_penalty,
        inp.target_soc,
    )
    contributions = {
        "energy_cost": float(res.total_cost),
        "battery_wear_penalty": float(res.total_wear_penalty * inp.lambda_wear),
        "peak_penalty": float(res.total_peak_penalty * inp.lambda_peak),
    }
    confidence = 0.92 if res.feasible else 0.20
    return ChargingPlan(
        optimizer="milp",
        feasible=res.feasible,
        power_per_hour=res.power_per_hour,
        soc_per_hour=res.soc_per_hour,
        total_cost_eur=float(res.total_cost),
        total_wear_penalty=float(res.total_wear_penalty),
        total_peak_penalty=float(res.total_peak_penalty),
        target_soc=inp.target_soc,
        achieved_final_soc=float(res.soc_per_hour[-1]),
        chosen_window_start_hour=start,
        chosen_window_end_hour=end,
        chosen_avg_rate_kw=avg_rate,
        confidence=confidence,
        feature_contributions=contributions,
    )


def _optimize_rl(inp: OptimizationInputs) -> ChargingPlan:
    from eeie.models.optimization.rl import load_ppo, rollout

    path = get_settings().checkpoint_dir / PPO_CHECKPOINT
    if not path.exists():
        logger.warning("PPO checkpoint not found, falling back to MILP.")
        return _optimize_milp(inp)

    env_config = ChargingEnvConfig(
        horizon_hours=inp.horizon_hours,
        capacity_kwh=inp.capacity_kwh,
        max_power_kw=inp.max_power_kw,
        charging_efficiency=inp.charging_efficiency,
        soc_min=inp.soc_min,
        soc_max=inp.soc_max,
        target_soc=inp.target_soc,
        initial_soc=inp.initial_soc,
        lambda_wear=inp.lambda_wear,
        lambda_peak=inp.lambda_peak,
    )
    env = ChargingEnv(
        config=env_config,
        hourly_rates=np.asarray(inp.hourly_rates, dtype=np.float32),
        peak_flags=np.asarray(inp.peak_flags, dtype=np.float32),
        battery_temps=np.asarray(inp.battery_temp_c, dtype=np.float32),
        seed=0,
    )
    model = load_ppo(path)
    trace = rollout(model, env=env)
    power = trace.power_trace
    soc = trace.soc_trace
    total_cost = sum(p * r for p, r in zip(power, inp.hourly_rates, strict=False))
    wear = sum(
        p * (1.0 + max(0.0, t - 35.0) * 0.05)
        for p, t in zip(power, inp.battery_temp_c, strict=False)
    )
    peak = sum(p * f for p, f in zip(power, inp.peak_flags, strict=False))
    start, end, avg_rate = _summarize(power, soc, total_cost, wear, peak, inp.target_soc)
    return ChargingPlan(
        optimizer="rl",
        feasible=trace.met_target,
        power_per_hour=power,
        soc_per_hour=soc,
        total_cost_eur=float(total_cost),
        total_wear_penalty=float(wear),
        total_peak_penalty=float(peak),
        target_soc=inp.target_soc,
        achieved_final_soc=trace.final_soc,
        chosen_window_start_hour=start,
        chosen_window_end_hour=end,
        chosen_avg_rate_kw=avg_rate,
        confidence=0.75 if trace.met_target else 0.40,
        feature_contributions={
            "energy_cost": float(total_cost),
            "battery_wear_penalty": float(wear * inp.lambda_wear),
            "peak_penalty": float(peak * inp.lambda_peak),
        },
    )


def naive_plan(inp: OptimizationInputs) -> ChargingPlan:
    """Charge immediately at full power until the target is hit. Baseline."""
    energy_needed = max(0.0, inp.target_soc - inp.initial_soc) * inp.capacity_kwh
    hours_needed = int(np.ceil(energy_needed / (inp.max_power_kw * inp.charging_efficiency)))
    power = [0.0] * inp.horizon_hours
    for t in range(min(hours_needed, inp.horizon_hours)):
        power[t] = inp.max_power_kw
    soc = [inp.initial_soc]
    for t in range(inp.horizon_hours):
        soc.append(
            min(inp.soc_max, soc[-1] + power[t] * inp.charging_efficiency / inp.capacity_kwh)
        )
    total_cost = sum(p * r for p, r in zip(power, inp.hourly_rates, strict=False))
    wear = sum(
        p * (1.0 + max(0.0, t - 35.0) * 0.05)
        for p, t in zip(power, inp.battery_temp_c, strict=False)
    )
    peak = sum(p * f for p, f in zip(power, inp.peak_flags, strict=False))
    start, end, avg_rate = _summarize(power, soc, total_cost, wear, peak, inp.target_soc)
    return ChargingPlan(
        optimizer="naive",
        feasible=soc[-1] >= inp.target_soc,
        power_per_hour=power,
        soc_per_hour=soc,
        total_cost_eur=float(total_cost),
        total_wear_penalty=float(wear),
        total_peak_penalty=float(peak),
        target_soc=inp.target_soc,
        achieved_final_soc=float(soc[-1]),
        chosen_window_start_hour=start,
        chosen_window_end_hour=end,
        chosen_avg_rate_kw=avg_rate,
        confidence=1.0,
    )


def optimize_charging(
    inp: OptimizationInputs,
    *,
    optimizer: Literal["milp", "rl", "naive"] | None = None,
) -> ChargingPlan:
    """Dispatch to the configured optimizer."""
    optimizer = optimizer or get_settings().optimizer  # type: ignore[assignment]
    if optimizer == "milp":
        return _optimize_milp(inp)
    if optimizer == "rl":
        return _optimize_rl(inp)
    if optimizer == "naive":
        return naive_plan(inp)
    raise ValueError(f"Unknown optimizer '{optimizer}'")
