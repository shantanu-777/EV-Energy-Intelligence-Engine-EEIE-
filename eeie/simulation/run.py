"""CLI entrypoint for the simulator.

Usage:

    python -m eeie.simulation.run --vehicles 100 --months 12

Writes the result both to TimescaleDB hypertables (if reachable) and to
Parquet snapshots under `$EEIE_DATA_DIR/simulation/`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from eeie.config import get_settings
from eeie.db import session_scope
from eeie.db.engine import get_engine
from eeie.db.hypertables import bootstrap_hypertables
from eeie.ingestion.loader import load_simulation_result
from eeie.simulation.engine import SimulationConfig, simulate_fleet

app = typer.Typer(add_completion=False, help="Run the EEIE synthetic data simulator.")


@app.command()
def main(
    vehicles: int = typer.Option(None, help="Number of vehicles (default from settings)."),
    months: int = typer.Option(None, help="Simulation horizon in months."),
    seed: int = typer.Option(None, help="RNG seed."),
    start: str = typer.Option("2025-01-01", help="Start timestamp (ISO 8601, UTC)."),
    out_dir: Path = typer.Option(
        None,
        help="Parquet output directory. Defaults to $EEIE_DATA_DIR/simulation.",
    ),
    write_db: bool = typer.Option(True, help="Write to TimescaleDB."),
    bootstrap_hypertables_flag: bool = typer.Option(
        True, "--bootstrap-hypertables/--no-bootstrap-hypertables"
    ),
) -> None:
    """Run the simulator end-to-end."""
    settings = get_settings()
    settings.ensure_dirs()

    cfg = SimulationConfig(
        n_vehicles=vehicles if vehicles is not None else settings.sim_vehicles,
        months=months if months is not None else settings.sim_months,
        seed=seed if seed is not None else settings.sim_seed,
        start_ts=pd.Timestamp(start, tz="UTC"),
    )
    logger.info("Simulation config: {}", cfg)

    result = simulate_fleet(cfg)

    out_root = out_dir or (settings.data_dir / "simulation")
    out_root.mkdir(parents=True, exist_ok=True)
    result.vehicles.to_parquet(out_root / "vehicles.parquet", index=False)
    result.telemetry.to_parquet(out_root / "telemetry.parquet", index=False)
    result.weather.to_parquet(out_root / "weather.parquet", index=False)
    result.tariffs.to_parquet(out_root / "tariffs.parquet", index=False)
    result.charging_events.to_parquet(out_root / "charging_events.parquet", index=False)
    logger.info("Wrote Parquet snapshots to {}.", out_root)

    if write_db:
        engine = get_engine()
        if bootstrap_hypertables_flag:
            bootstrap_hypertables(engine)
        with session_scope() as session:
            load_simulation_result(session, result)
        logger.info("Wrote simulation to database.")

    logger.info(
        "Done. {} telemetry rows, {} charging events, {} weather rows.",
        len(result.telemetry),
        len(result.charging_events),
        len(result.weather),
    )


if __name__ == "__main__":
    app()
