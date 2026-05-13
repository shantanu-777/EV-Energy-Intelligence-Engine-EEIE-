"""PPO training entrypoint for the charging RL agent."""

from __future__ import annotations

from loguru import logger

from eeie.config import get_settings
from eeie.models.optimization.env import ChargingEnvConfig
from eeie.models.optimization.optimize import PPO_CHECKPOINT
from eeie.models.optimization.rl import PPOTrainConfig, save_ppo, train_ppo


def train_optimizer_rl(
    *,
    env_config: ChargingEnvConfig | None = None,
    train_config: PPOTrainConfig | None = None,
) -> None:
    model = train_ppo(env_config=env_config, train_config=train_config)
    target = get_settings().checkpoint_dir / PPO_CHECKPOINT
    save_ppo(model, target)
    logger.info("Saved PPO charging agent to {}", target)
