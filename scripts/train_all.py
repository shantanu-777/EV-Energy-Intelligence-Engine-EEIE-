"""Train every model variant in one go.

Usage:

    python -m scripts.train_all
"""

from __future__ import annotations

import typer
from loguru import logger

from eeie.config import get_settings
from eeie.db import session_scope
from eeie.models.battery.train import train_battery_correction
from eeie.models.behavior.train import train_behavior
from eeie.models.demand.train import train_demand_tft, train_demand_xgb
from eeie.models.optimization.env import ChargingEnvConfig
from eeie.models.optimization.rl import PPOTrainConfig
from eeie.models.optimization.train import train_optimizer_rl
from eeie.models.range.train import train_range_lstm, train_range_xgb

app = typer.Typer(add_completion=False, help="Train every EEIE model.")


@app.command()
def main(
    skip_tft: bool = typer.Option(False, help="Skip the TFT (slow on CPU)."),
    skip_lstm: bool = typer.Option(False, help="Skip the LSTM."),
    skip_rl: bool = typer.Option(False, help="Skip the PPO RL agent."),
    rl_timesteps: int = typer.Option(20_000, help="PPO training timesteps."),
) -> None:
    get_settings().ensure_dirs()
    results: dict[str, dict[str, float]] = {}

    with session_scope() as session:
        logger.info("Training Range XGB...")
        results["range_xgb"] = train_range_xgb(session)

        if not skip_lstm:
            logger.info("Training Range LSTM...")
            results["range_lstm"] = train_range_lstm(session, epochs=4)

        logger.info("Training Demand XGB...")
        results["demand_xgb"] = train_demand_xgb(session)

        if not skip_tft:
            try:
                logger.info("Training Demand TFT...")
                results["demand_tft"] = train_demand_tft(session, max_epochs=3)
            except Exception as exc:
                logger.warning("Demand TFT failed: {}", exc)

        logger.info("Training Battery correction...")
        results["battery"] = train_battery_correction(session)

        logger.info("Training Behavior models...")
        results["behavior"] = train_behavior(session)

    if not skip_rl:
        logger.info("Training Charging RL agent (PPO)...")
        train_optimizer_rl(
            env_config=ChargingEnvConfig(),
            train_config=PPOTrainConfig(total_timesteps=rl_timesteps),
        )

    logger.info("All training complete. Metrics: {}", results)


if __name__ == "__main__":
    app()
