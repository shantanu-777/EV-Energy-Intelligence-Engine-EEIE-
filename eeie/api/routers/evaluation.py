"""Expose the naive-vs-rule-based-vs-EEIE ablation harness via HTTP.

The ablation is moderately expensive (each scenario invokes the MILP solver),
so the endpoint caches results in-process keyed by `(n_scenarios, horizon,
optimizer, seed)`. Pass `refresh=true` to force a recompute.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from eeie.api.deps import db
from eeie.evaluation.ablation import run_charging_ablation
from eeie.evaluation.scenarios import build_scenarios_from_db

router = APIRouter(prefix="/evaluate", tags=["evaluation"])


_CACHE: dict[tuple[int, int, str, int], dict[str, Any]] = {}
_CACHE_LOCK = Lock()


def _run(
    session: Session,
    *,
    n_scenarios: int,
    horizon_hours: int,
    optimizer: Literal["milp", "rl"],
    seed: int,
) -> dict[str, Any]:
    scenarios = build_scenarios_from_db(
        session, n=n_scenarios, horizon_hours=horizon_hours, seed=seed
    )
    if not scenarios:
        raise HTTPException(
            status_code=404,
            detail="No scenarios available. Run the simulator first.",
        )
    report = run_charging_ablation(scenarios, eeie_optimizer=optimizer)
    return {
        "n_scenarios": report.n_scenarios,
        "horizon_hours": horizon_hours,
        "optimizer": optimizer,
        "by_strategy": report.by_strategy,
        "cost_reduction_pct_eeie_vs_naive": report.cost_reduction_pct_eeie_vs_naive,
        "peak_reduction_pct_eeie_vs_naive": report.peak_reduction_pct_eeie_vs_naive,
        "wear_reduction_pct_eeie_vs_naive": report.wear_reduction_pct_eeie_vs_naive,
    }


@router.get("/ablation")
def ablation(
    session: Session = Depends(db),
    n_scenarios: int = Query(20, ge=5, le=100),
    horizon_hours: int = Query(24, ge=6, le=72),
    optimizer: Literal["milp", "rl"] = Query("milp"),
    seed: int = Query(0, ge=0),
    refresh: bool = Query(False),
) -> dict[str, Any]:
    key = (n_scenarios, horizon_hours, optimizer, seed)
    with _CACHE_LOCK:
        if not refresh and key in _CACHE:
            return _CACHE[key]

    result = _run(
        session,
        n_scenarios=n_scenarios,
        horizon_hours=horizon_hours,
        optimizer=optimizer,
        seed=seed,
    )
    with _CACHE_LOCK:
        _CACHE[key] = result
    return result
