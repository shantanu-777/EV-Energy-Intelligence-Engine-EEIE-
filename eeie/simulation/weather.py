"""Synthetic weather generator.

A simple seasonal model: annual sinusoid for mean temperature, diurnal
sinusoid on top, plus white noise. Humidity and wind are weakly correlated
to the temperature pattern. Output is hourly. One region per call.

Realism is intentionally coarse — Phase 1 prioritizes deterministic,
reproducible weather over physical accuracy. Real weather APIs can be
swapped in later via the `ingestion` layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegionClimate:
    """Coarse climate descriptor for a region."""

    region_id: str
    annual_mean_c: float
    annual_amplitude_c: float
    diurnal_amplitude_c: float
    base_humidity: float
    base_wind_kmh: float
    rain_prob_per_hour: float


CENTRAL_EUROPE = RegionClimate(
    region_id="central_eu",
    annual_mean_c=11.0,
    annual_amplitude_c=11.5,
    diurnal_amplitude_c=4.0,
    base_humidity=0.70,
    base_wind_kmh=12.0,
    rain_prob_per_hour=0.06,
)


def generate_weather(
    *,
    start: pd.Timestamp,
    periods: int,
    climate: RegionClimate = CENTRAL_EUROPE,
    seed: int = 0,
) -> pd.DataFrame:
    """Return an hourly weather frame indexed from `start` for `periods` hours."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=periods, freq="h", tz="UTC")

    day_of_year = np.asarray(idx.dayofyear)
    hour = np.asarray(idx.hour)

    # Annual cycle: trough mid-January, peak mid-July.
    annual = climate.annual_mean_c - climate.annual_amplitude_c * np.cos(
        2 * np.pi * (day_of_year - 15) / 365.0
    )
    # Diurnal cycle: coldest at 5am, warmest at 3pm.
    diurnal = climate.diurnal_amplitude_c * np.sin(2 * np.pi * (hour - 9) / 24.0)
    noise = rng.normal(0.0, 1.5, size=periods)
    temperature_c = annual + diurnal + noise

    humidity = np.clip(
        climate.base_humidity
        - 0.005 * (temperature_c - climate.annual_mean_c)
        + rng.normal(0.0, 0.05, size=periods),
        0.20,
        1.00,
    )
    wind_speed_kmh = np.clip(
        climate.base_wind_kmh + rng.normal(0.0, 4.0, size=periods),
        0.0,
        80.0,
    )
    is_raining = rng.random(periods) < climate.rain_prob_per_hour
    precipitation_mm = np.where(is_raining, rng.exponential(0.8, size=periods), 0.0)

    return pd.DataFrame(
        {
            "ts": idx,
            "region_id": climate.region_id,
            "temperature_c": temperature_c.astype(np.float32),
            "humidity": humidity.astype(np.float32),
            "wind_speed_kmh": wind_speed_kmh.astype(np.float32),
            "precipitation_mm": precipitation_mm.astype(np.float32),
        }
    )
