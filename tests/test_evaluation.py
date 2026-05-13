"""Evaluation harness smoke tests."""

from __future__ import annotations

from eeie.evaluation import run_charging_ablation
from eeie.models.optimization import OptimizationInputs


def _scenario(initial_soc: float, target_soc: float) -> OptimizationInputs:
    rates = [
        0.12 if h < 7 or h >= 22 else (0.30 if 7 <= h < 10 or 17 <= h < 21 else 0.20)
        for h in range(24)
    ]
    peaks = [1 if 7 <= h < 10 or 17 <= h < 21 else 0 for h in range(24)]
    return OptimizationInputs(
        horizon_hours=24,
        initial_soc=initial_soc,
        target_soc=target_soc,
        soc_min=0.10,
        soc_max=0.95,
        capacity_kwh=60.0,
        max_power_kw=11.0,
        charging_efficiency=0.93,
        hourly_rates=rates,
        peak_flags=peaks,
        battery_temp_c=[25.0] * 24,
    )


def test_ablation_eeie_beats_naive_on_cost():
    scenarios = [_scenario(0.25, 0.80), _scenario(0.35, 0.80), _scenario(0.50, 0.75)]
    report = run_charging_ablation(scenarios, eeie_optimizer="milp")
    assert report.n_scenarios == 3
    assert "naive" in report.by_strategy
    assert "eeie" in report.by_strategy
    assert (
        report.by_strategy["eeie"]["mean_cost_eur"]
        <= report.by_strategy["naive"]["mean_cost_eur"] + 1e-6
    )
    assert report.cost_reduction_pct_eeie_vs_naive >= 0.0
