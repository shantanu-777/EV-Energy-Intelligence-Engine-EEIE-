"""PPO agent for the charging optimization environment.

Phase 1 trains for a small budget; reward shaping and hyperparameter
search are deferred. The persisted policy can be loaded and rolled out
inside the unified `optimize_charging` API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from eeie.models.optimization.env import ChargingEnv, ChargingEnvConfig


@dataclass
class PPOTrainConfig:
    total_timesteps: int = 20_000
    n_envs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.95
    n_steps: int = 128
    batch_size: int = 64
    seed: int = 42


def _make_env(env_config: ChargingEnvConfig, seed: int):
    def _f():
        return ChargingEnv(config=env_config, seed=seed)

    return _f


def train_ppo(
    *,
    env_config: ChargingEnvConfig | None = None,
    train_config: PPOTrainConfig | None = None,
) -> PPO:
    env_config = env_config or ChargingEnvConfig()
    train_config = train_config or PPOTrainConfig()
    envs = DummyVecEnv(
        [_make_env(env_config, train_config.seed + i) for i in range(train_config.n_envs)]
    )
    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=train_config.learning_rate,
        gamma=train_config.gamma,
        n_steps=train_config.n_steps,
        batch_size=train_config.batch_size,
        seed=train_config.seed,
        verbose=0,
    )
    logger.info("Training PPO charging agent for {} timesteps.", train_config.total_timesteps)
    model.learn(total_timesteps=train_config.total_timesteps, progress_bar=False)
    return model


def save_ppo(model: PPO, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def load_ppo(path: Path) -> PPO:
    return PPO.load(str(path))


@dataclass
class RolloutResult:
    soc_trace: list[float]
    power_trace: list[float]
    cost_trace: list[float]
    total_reward: float
    final_soc: float
    met_target: bool


def rollout(
    model: PPO,
    *,
    env: ChargingEnv,
    deterministic: bool = True,
) -> RolloutResult:
    """Roll out one episode and return per-step trace + summary."""
    obs, _ = env.reset()
    soc_trace: list[float] = [float(env.soc)]
    power_trace: list[float] = []
    cost_trace: list[float] = []
    total_reward = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        total_reward += float(reward)
        power_trace.append(float(action[0]) * env.config.max_power_kw)
        cost_trace.append(float(info["cost"]))
        soc_trace.append(float(info["soc"]))
        if terminated or truncated:
            break
    return RolloutResult(
        soc_trace=soc_trace,
        power_trace=power_trace,
        cost_trace=cost_trace,
        total_reward=total_reward,
        final_soc=float(soc_trace[-1]),
        met_target=bool(soc_trace[-1] >= env.config.target_soc),
    )
