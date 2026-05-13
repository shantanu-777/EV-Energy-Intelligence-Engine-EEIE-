"""Run the naive-vs-rule-based-vs-EEIE ablation harness."""

from __future__ import annotations

import json

import typer
from loguru import logger

from eeie.db import session_scope
from eeie.evaluation.ablation import run_charging_ablation
from eeie.evaluation.scenarios import build_scenarios_from_db

app = typer.Typer(add_completion=False, help="Run EEIE evaluation.")


@app.command()
def main(
    n_scenarios: int = typer.Option(30, help="Number of scenarios to evaluate."),
    horizon_hours: int = typer.Option(24, help="Optimization horizon."),
    optimizer: str = typer.Option("milp", help="EEIE optimizer: milp or rl."),
    out: str = typer.Option("data/evaluation/ablation_report.json", help="Output JSON path."),
) -> None:
    with session_scope() as session:
        scenarios = build_scenarios_from_db(session, n=n_scenarios, horizon_hours=horizon_hours)
        if not scenarios:
            logger.error("No scenarios. Run the simulator first.")
            raise typer.Exit(code=1)
        report = run_charging_ablation(scenarios, eeie_optimizer=optimizer)  # type: ignore[arg-type]

    import pathlib

    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "n_scenarios": report.n_scenarios,
                "by_strategy": report.by_strategy,
                "cost_reduction_pct_eeie_vs_naive": report.cost_reduction_pct_eeie_vs_naive,
                "peak_reduction_pct_eeie_vs_naive": report.peak_reduction_pct_eeie_vs_naive,
                "wear_reduction_pct_eeie_vs_naive": report.wear_reduction_pct_eeie_vs_naive,
            },
            indent=2,
        )
    )
    logger.info("Wrote ablation report to {}", out_path)
    print(out_path.read_text())


if __name__ == "__main__":
    app()
