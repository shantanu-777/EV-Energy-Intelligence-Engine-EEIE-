"""MILP charging schedule optimizer (OR-Tools CP-SAT).

We discretize charging over a planning horizon (default 24 hourly slots),
deciding charge power in [0, P_max] per slot to minimize

    total cost + lambda_w * battery wear + lambda_p * peak penalty

subject to:
- SOC[t] = SOC[t-1] + (P[t] * dt * eta) / capacity
- SOC_min <= SOC[t] <= SOC_max
- SOC[T] >= target SOC
- P[t] in [0, P_max]

Battery wear is approximated as a penalty on energy charged when temperature
is high and an additional penalty on DC-fast (large) charging events.
Peak penalty discourages charging during peak-tier hours.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from ortools.linear_solver import pywraplp


@dataclass
class MILPInputs:
    """All inputs the MILP needs in concrete numbers."""

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
    dt_h: float = 1.0


@dataclass
class MILPResult:
    feasible: bool
    objective: float
    total_cost: float
    total_wear_penalty: float
    total_peak_penalty: float
    power_per_hour: list[float]
    soc_per_hour: list[float]


def solve_charging_milp(inp: MILPInputs) -> MILPResult:
    """Solve the charging MILP. Returns an `MILPResult`."""
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("OR-Tools CBC solver not available.")

    T = inp.horizon_hours
    p_max = inp.max_power_kw

    # If the vehicle starts already above soc_max (e.g. fully charged), lift
    # the upper bound so the start state stays feasible; otherwise the
    # constraints SOC[0] == initial_soc and SOC[0] <= soc_max contradict.
    effective_soc_max = max(inp.soc_max, inp.initial_soc)
    effective_soc_min = min(inp.soc_min, inp.initial_soc)

    power = [solver.NumVar(0.0, p_max, f"P_{t}") for t in range(T)]
    soc = [
        solver.NumVar(effective_soc_min, effective_soc_max, f"SOC_{t}")
        for t in range(T + 1)
    ]

    solver.Add(soc[0] == inp.initial_soc)
    for t in range(T):
        solver.Add(
            soc[t + 1]
            == soc[t] + (power[t] * inp.dt_h * inp.charging_efficiency) / inp.capacity_kwh
        )
    solver.Add(soc[T] >= inp.target_soc)

    cost_terms = []
    wear_terms = []
    peak_terms = []
    for t in range(T):
        cost_terms.append(power[t] * inp.dt_h * inp.hourly_rates[t])
        wear_coef = 1.0 + max(0.0, inp.battery_temp_c[t] - 35.0) * 0.05
        wear_terms.append(power[t] * inp.dt_h * wear_coef)
        peak_terms.append(power[t] * inp.dt_h * inp.peak_flags[t])

    solver.Minimize(
        solver.Sum(cost_terms)
        + inp.lambda_wear * solver.Sum(wear_terms)
        + inp.lambda_peak * solver.Sum(peak_terms)
    )

    status = solver.Solve()
    feasible = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    if not feasible:
        logger.warning(
            "MILP infeasible. initial_soc={:.2f} target_soc={:.2f}",
            inp.initial_soc,
            inp.target_soc,
        )
        return MILPResult(
            feasible=False,
            objective=float("inf"),
            total_cost=float("inf"),
            total_wear_penalty=float("inf"),
            total_peak_penalty=float("inf"),
            power_per_hour=[0.0] * T,
            soc_per_hour=[inp.initial_soc] * (T + 1),
        )

    p_sched = [float(power[t].solution_value()) for t in range(T)]
    soc_sched = [float(soc[t].solution_value()) for t in range(T + 1)]
    total_cost = sum(p_sched[t] * inp.dt_h * inp.hourly_rates[t] for t in range(T))
    wear_penalty = sum(
        p_sched[t] * inp.dt_h * (1.0 + max(0.0, inp.battery_temp_c[t] - 35.0) * 0.05)
        for t in range(T)
    )
    peak_penalty = sum(p_sched[t] * inp.dt_h * inp.peak_flags[t] for t in range(T))

    return MILPResult(
        feasible=True,
        objective=float(solver.Objective().Value()),
        total_cost=float(total_cost),
        total_wear_penalty=float(wear_penalty),
        total_peak_penalty=float(peak_penalty),
        power_per_hour=p_sched,
        soc_per_hour=soc_sched,
    )
