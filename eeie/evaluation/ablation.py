"""Charging optimization ablation: naive vs rule-based vs EEIE (MILP/RL).

Constructs a small set of representative scenarios and reports per-strategy
cost, peak-hour energy, and battery-wear penalty. The harness is dataset-
agnostic — it accepts a list of `OptimizationInputs` so callers can drive
it from the DB or from synthetic scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from loguru import logger

from eeie.evaluation.metrics import (
    cost_reduction_pct,
    degradation_reduction_pct,
    peak_avoidance_pct,
)
from eeie.models.optimization import OptimizationInputs, optimize_charging
from eeie.models.optimization.optimize import naive_plan


def rule_based_plan(inp: OptimizationInputs):
    """Charge only during off-peak hours (peak_flag == 0)."""
    horizon = inp.horizon_hours
    capacity = inp.capacity_kwh
    needed = max(0.0, inp.target_soc - inp.initial_soc) * capacity
    off_peak_hours = [t for t in range(horizon) if inp.peak_flags[t] == 0]
    power = [0.0] * horizon
    rate = inp.max_power_kw
    delivered = 0.0
    for t in off_peak_hours:
        if delivered >= needed:
            break
        p = min(rate, (needed - delivered) / inp.charging_efficiency)
        power[t] = p
        delivered += p * inp.charging_efficiency
    soc = [inp.initial_soc]
    for t in range(horizon):
        soc.append(min(inp.soc_max, soc[-1] + power[t] * inp.charging_efficiency / capacity))
    total_cost = sum(p * r for p, r in zip(power, inp.hourly_rates, strict=False))
    wear = sum(
        p * (1.0 + max(0.0, t - 35.0) * 0.05)
        for p, t in zip(power, inp.battery_temp_c, strict=False)
    )
    peak = sum(p * f for p, f in zip(power, inp.peak_flags, strict=False))
    return {
        "strategy": "rule_based",
        "feasible": soc[-1] >= inp.target_soc,
        "total_cost_eur": float(total_cost),
        "total_wear": float(wear),
        "total_peak_kwh": float(peak),
        "final_soc": float(soc[-1]),
        "power_per_hour": power,
        "soc_per_hour": soc,
    }


def _summary(plan) -> dict[str, float | bool | str | list[float]]:
    return {
        "strategy": plan.optimizer,
        "feasible": bool(plan.feasible),
        "total_cost_eur": float(plan.total_cost_eur),
        "total_wear": float(plan.total_wear_penalty),
        "total_peak_kwh": float(plan.total_peak_penalty),
        "final_soc": float(plan.achieved_final_soc),
        "power_per_hour": list(plan.power_per_hour),
        "soc_per_hour": list(plan.soc_per_hour),
    }


@dataclass
class AblationReport:
    n_scenarios: int
    by_strategy: dict[str, dict[str, float]] = field(default_factory=dict)
    cost_reduction_pct_eeie_vs_naive: float = 0.0
    peak_reduction_pct_eeie_vs_naive: float = 0.0
    wear_reduction_pct_eeie_vs_naive: float = 0.0


def run_charging_ablation(
    scenarios: list[OptimizationInputs],
    *,
    eeie_optimizer: Literal["milp", "rl"] = "milp",
) -> AblationReport:
    """Run all three strategies on every scenario and aggregate the results."""
    rows: dict[str, list[dict]] = {"naive": [], "rule_based": [], "eeie": []}

    for inp in scenarios:
        naive = naive_plan(inp)
        rule = rule_based_plan(inp)
        eeie = optimize_charging(inp, optimizer=eeie_optimizer)
        rows["naive"].append(_summary(naive))
        rows["rule_based"].append(rule)
        rows["eeie"].append(_summary(eeie))

    def _mean(seq: list[dict], key: str) -> float:
        # Only average over feasible scenarios with finite values, so a
        # single infeasibility doesn't poison the aggregate with Infinity/NaN.
        vals = [
            float(s[key])
            for s in seq
            if s.get("feasible") and np.isfinite(float(s[key]))
        ]
        return float(np.mean(vals)) if vals else 0.0

    by_strategy: dict[str, dict[str, float]] = {}
    for name, scenarios_results in rows.items():
        by_strategy[name] = {
            "mean_cost_eur": _mean(scenarios_results, "total_cost_eur"),
            "mean_wear": _mean(scenarios_results, "total_wear"),
            "mean_peak_kwh": _mean(scenarios_results, "total_peak_kwh"),
            "feasible_pct": float(
                np.mean([1.0 if s["feasible"] else 0.0 for s in scenarios_results]) * 100.0
            ),
        }

    cost_red = cost_reduction_pct(
        by_strategy["naive"]["mean_cost_eur"], by_strategy["eeie"]["mean_cost_eur"]
    )
    peak_red = peak_avoidance_pct(
        by_strategy["naive"]["mean_peak_kwh"], by_strategy["eeie"]["mean_peak_kwh"]
    )
    wear_red = degradation_reduction_pct(
        by_strategy["naive"]["mean_wear"], by_strategy["eeie"]["mean_wear"]
    )

    report = AblationReport(
        n_scenarios=len(scenarios),
        by_strategy=by_strategy,
        cost_reduction_pct_eeie_vs_naive=cost_red,
        peak_reduction_pct_eeie_vs_naive=peak_red,
        wear_reduction_pct_eeie_vs_naive=wear_red,
    )
    logger.info(
        "Ablation done. cost_red={:.1f}% peak_red={:.1f}% wear_red={:.1f}%",
        cost_red,
        peak_red,
        wear_red,
    )
    return report
