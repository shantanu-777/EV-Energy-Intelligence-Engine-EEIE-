"""Application settings, sourced from environment variables and `.env`.

A single `Settings` object is the entry point for all configuration. Every
module imports `get_settings()` rather than reading environment variables
directly, so test overrides remain trivial via `get_settings.cache_clear()`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    # --- Database ---
    database_url: str = Field(
        default="postgresql+psycopg://eeie:eeie_dev_password@localhost:5432/eeie",
        validation_alias="EEIE_DATABASE_URL",
    )

    # --- API ---
    api_host: str = Field(default="0.0.0.0", validation_alias="EEIE_API_HOST")
    api_port: int = Field(default=8000, validation_alias="EEIE_API_PORT")
    api_log_level: str = Field(default="info", validation_alias="EEIE_API_LOG_LEVEL")
    api_base_url: str = Field(default="http://localhost:8000", validation_alias="EEIE_API_BASE_URL")

    # --- Paths ---
    data_dir: Path = Field(default=Path("data"), validation_alias="EEIE_DATA_DIR")
    checkpoint_dir: Path = Field(
        default=Path("checkpoints"), validation_alias="EEIE_CHECKPOINT_DIR"
    )

    # --- Simulation defaults ---
    sim_vehicles: int = Field(default=100, validation_alias="EEIE_SIM_VEHICLES")
    sim_months: int = Field(default=12, validation_alias="EEIE_SIM_MONTHS")
    sim_seed: int = Field(default=42, validation_alias="EEIE_SIM_SEED")

    # --- Optimization ---
    optimizer: str = Field(default="milp", validation_alias="EEIE_OPTIMIZER")

    def ensure_dirs(self) -> None:
        """Create writable runtime directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Call `get_settings.cache_clear()` in tests."""
    return Settings()
