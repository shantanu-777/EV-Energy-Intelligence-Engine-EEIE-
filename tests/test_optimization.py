"""Optimization engine: MILP, env, and naive plan smoke tests."""

from __future__ import annotations

import numpy as np

from eeie.models.optimization import OptimizationInputs, optimize_charging
from eeie.models.optimization.env import ChargingEnv, ChargingEnvConfig
from eeie.models.optimization.milp import MILPInputs, solve_charging_milp
from eeie.models.optimization.optimize import naive_plan


def _inputs() -> OptimizationInputs:
    rates = [
        0.12 if h < 7 or h > 21 else (0.30 if 7 <= h < 10 or 17 <= h < 21 else 0.20)
        for h in range(24)
    ]
    peaks = [1 if 7 <= h < 10 or 17 <= h < 21 else 0 for h in range(24)]
    temps = [25.0] * 24
    return OptimizationInputs(
        horizon_hours=24,
        initial_soc=0.30,
        target_soc=0.80,
        soc_min=0.10,
        soc_max=0.95,
        capacity_kwh=60.0,
        max_power_kw=11.0,
        charging_efficiency=0.93,
        hourly_rates=rates,
        peak_flags=peaks,
        battery_temp_c=temps,
    )


def test_milp_feasible_and_target_met():
    inp = _inputs()
    milp_in = MILPInputs(
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
    )
    res = solve_charging_milp(milp_in)
    assert res.feasible
    assert res.soc_per_hour[-1] >= inp.target_soc - 1e-6


def test_milp_prefers_off_peak():
    inp = _inputs()
    plan = optimize_charging(inp, optimizer="milp")
    naive = naive_plan(inp)
    assert plan.total_cost_eur <= naive.total_cost_eur + 1e-6


def test_charging_env_runs():
    env = ChargingEnv(config=ChargingEnvConfig(), seed=0)
    obs, _ = env.reset()
    assert obs.shape == (9,)
    total_reward = 0.0
    for _ in range(env.config.horizon_hours):
        obs, reward, term, _, _ = env.step(np.array([0.5], dtype=np.float32))
        total_reward += reward
        if term:
            break
    assert np.isfinite(total_reward)
