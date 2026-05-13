"""Gymnasium environment for the charging optimization RL agent.

State (continuous):
    [soc, hour_of_day_sin, hour_of_day_cos, current_rate, battery_temp,
     remaining_hours, target_soc, energy_needed_kwh, peak_flag]

Action (continuous in [0, 1]):
    Fraction of max charging power applied this hour.

Reward at each step:
    -cost(t)
    - lambda_wear * wear_penalty(t)
    - lambda_peak * peak_penalty(t)
    + terminal bonus if SOC >= target at end
    - terminal penalty if SOC < target at end (under-deliver)
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class ChargingEnvConfig:
    horizon_hours: int = 24
    capacity_kwh: float = 60.0
    max_power_kw: float = 11.0
    charging_efficiency: float = 0.93
    soc_min: float = 0.10
    soc_max: float = 0.95
    target_soc: float = 0.80
    initial_soc: float = 0.30
    lambda_wear: float = 0.5
    lambda_peak: float = 0.3
    terminal_bonus: float = 30.0
    terminal_miss_penalty: float = 100.0


class ChargingEnv(gym.Env):
    """A single-vehicle charging optimization environment."""

    metadata = {"render_modes": []}  # noqa: RUF012 - gymnasium expects this attr

    def __init__(
        self,
        *,
        config: ChargingEnvConfig | None = None,
        hourly_rates: np.ndarray | None = None,
        peak_flags: np.ndarray | None = None,
        battery_temps: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ChargingEnvConfig()
        T = self.config.horizon_hours
        self.rng = np.random.default_rng(seed)
        self._default_rates = hourly_rates
        self._default_peaks = peak_flags
        self._default_temps = battery_temps

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, 0.0, -30.0, 0.0, 0.0, -200.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 2.0, 80.0, float(T), 1.0, 200.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._t = 0
        self.rates = np.zeros(T, dtype=np.float32)
        self.peaks = np.zeros(T, dtype=np.float32)
        self.temps = np.full(T, 25.0, dtype=np.float32)
        self.soc = self.config.initial_soc

    def _sample_episode(self) -> None:
        T = self.config.horizon_hours
        if self._default_rates is not None:
            self.rates = self._default_rates.astype(np.float32)
        else:
            base = np.full(T, 0.20, dtype=np.float32)
            base[7:10] += 0.15
            base[17:21] += 0.18
            base[0:7] -= 0.08
            self.rates = (base + self.rng.normal(0.0, 0.02, size=T)).astype(np.float32)
        if self._default_peaks is not None:
            self.peaks = self._default_peaks.astype(np.float32)
        else:
            self.peaks = (self.rates > 0.30).astype(np.float32)
        if self._default_temps is not None:
            self.temps = self._default_temps.astype(np.float32)
        else:
            self.temps = (
                25.0
                + 8.0 * np.sin(np.linspace(0, 2 * np.pi, T))
                + self.rng.normal(0.0, 1.0, size=T)
            ).astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self.soc = float(
            options["initial_soc"]
            if options and "initial_soc" in options
            else self.config.initial_soc
        )
        self._sample_episode()
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        T = self.config.horizon_hours
        hour = self._t % 24
        target = self.config.target_soc
        energy_needed = max(0.0, target - self.soc) * self.config.capacity_kwh
        return np.array(
            [
                self.soc,
                np.sin(2 * np.pi * hour / 24.0),
                np.cos(2 * np.pi * hour / 24.0),
                float(self.rates[self._t]) if self._t < T else 0.0,
                float(self.temps[self._t]) if self._t < T else 25.0,
                float(T - self._t),
                target,
                energy_needed,
                float(self.peaks[self._t]) if self._t < T else 0.0,
            ],
            dtype=np.float32,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        cfg = self.config
        a = float(np.clip(action[0], 0.0, 1.0))
        power = a * cfg.max_power_kw
        energy_delivered = power * cfg.charging_efficiency
        new_soc = min(cfg.soc_max, self.soc + energy_delivered / cfg.capacity_kwh)
        cost = power * float(self.rates[self._t])
        wear = power * (1.0 + max(0.0, float(self.temps[self._t]) - 35.0) * 0.05)
        peak_penalty = power * float(self.peaks[self._t])
        reward = -(cost + cfg.lambda_wear * wear + cfg.lambda_peak * peak_penalty)

        self.soc = new_soc
        self._t += 1
        terminated = self._t >= cfg.horizon_hours
        if terminated:
            if self.soc >= cfg.target_soc:
                reward += cfg.terminal_bonus
            else:
                reward -= cfg.terminal_miss_penalty * (cfg.target_soc - self.soc)
        info = {"soc": self.soc, "cost": cost, "wear": wear, "peak": peak_penalty}
        return self._obs(), float(reward), terminated, False, info
