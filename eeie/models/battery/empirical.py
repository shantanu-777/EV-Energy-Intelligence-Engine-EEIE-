"""Empirical battery aging model.

Two parallel mechanisms:

- Calendar aging: square-root-of-time decay, Arrhenius-scaled by mean SOC and temp.
- Cycle aging: linear in equivalent full cycles, accelerated by DC-fast share
  and operation outside the 20-80% SOC band.

These are coarse, literature-inspired approximations. The ML correction
layer in `correction.py` learns the residual against the simulated truth.

References (qualitative):
  - Wang et al., 'Cycle-life model for graphite-LiFePO4 cells' (2011)
  - Schmalstieg et al., 'Holistic aging model for Li-ion batteries' (2014)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmpiricalAgingParams:
    """Coarse aging coefficients tuned to plausible automotive ranges."""

    calendar_k: float = 0.02  # SOH lost per sqrt(year) at reference conditions
    ref_temp_c: float = 25.0
    temp_sensitivity: float = 0.05  # SOH loss per 10C above reference
    soc_stress_coef: float = 0.6  # SOH loss penalty for high mean SOC
    cycle_k: float = 4.5e-5  # SOH lost per equivalent full cycle
    dc_fast_penalty: float = 1.7
    deep_cycle_penalty: float = 1.4


DEFAULT_PARAMS = EmpiricalAgingParams()


def calendar_aging_per_year(
    mean_soc: float,
    mean_temp_c: float,
    *,
    params: EmpiricalAgingParams = DEFAULT_PARAMS,
) -> float:
    """SOH loss per year from calendar aging."""
    temp_factor = 1.0 + params.temp_sensitivity * max(0.0, (mean_temp_c - params.ref_temp_c) / 10.0)
    soc_factor = 1.0 + params.soc_stress_coef * max(0.0, mean_soc - 0.50) ** 2 * 4.0
    return params.calendar_k * temp_factor * soc_factor


def cycle_aging_per_eq_cycle(
    *,
    dc_fast_fraction: float,
    deep_cycle_fraction: float,
    params: EmpiricalAgingParams = DEFAULT_PARAMS,
) -> float:
    """SOH loss per equivalent full cycle from cycling aging."""
    return params.cycle_k * (
        1.0
        + (params.dc_fast_penalty - 1.0) * np.clip(dc_fast_fraction, 0.0, 1.0)
        + (params.deep_cycle_penalty - 1.0) * np.clip(deep_cycle_fraction, 0.0, 1.0)
    )


def predict_soh_trajectory_empirical(
    *,
    initial_soh: float,
    years: float,
    annual_eq_cycles: float,
    mean_soc: float,
    mean_temp_c: float,
    dc_fast_fraction: float,
    deep_cycle_fraction: float,
    n_points: int = 24,
    params: EmpiricalAgingParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """Project SOH vs time. Returns columns: years, soh."""
    t = np.linspace(0.0, years, n_points)
    cal = calendar_aging_per_year(mean_soc, mean_temp_c, params=params) * np.sqrt(t + 1e-9)
    cyc = (
        cycle_aging_per_eq_cycle(
            dc_fast_fraction=dc_fast_fraction,
            deep_cycle_fraction=deep_cycle_fraction,
            params=params,
        )
        * annual_eq_cycles
        * t
    )
    soh = np.clip(initial_soh - cal - cyc, 0.5, 1.0)
    return pd.DataFrame({"years": t, "soh": soh})


def annual_degradation_pct(
    *,
    initial_soh: float,
    annual_eq_cycles: float,
    mean_soc: float,
    mean_temp_c: float,
    dc_fast_fraction: float,
    deep_cycle_fraction: float,
    params: EmpiricalAgingParams = DEFAULT_PARAMS,
) -> float:
    """Single-year SOH loss in percentage points."""
    cal = calendar_aging_per_year(mean_soc, mean_temp_c, params=params)
    cyc = (
        cycle_aging_per_eq_cycle(
            dc_fast_fraction=dc_fast_fraction,
            deep_cycle_fraction=deep_cycle_fraction,
            params=params,
        )
        * annual_eq_cycles
    )
    return float((cal + cyc) * 100.0)
