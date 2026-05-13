"""Driving pattern generation.

For each vehicle and day we build an hour-by-hour driving profile:

- Weekdays have two trip windows (commute) anchored around 07:30 and 18:00.
- Weekends spread trips across late morning to evening.
- Daily km is drawn from the vehicle's weekday/weekend mean with noise,
  occasionally with a long-trip multiplier.
- Aggressive drivers see a higher probability of harsh-acceleration events
  within driving hours.

Output is a per-day vector of:
    km_per_hour[24], avg_speed_per_hour[24], aggressive_events_per_hour[24]
"""

from __future__ import annotations

import numpy as np

from eeie.simulation.fleet import SampledVehicle


def _trip_window(rng: np.random.Generator, center_hour: float, spread: float) -> int:
    """Sample a trip start hour from a clipped normal around `center_hour`."""
    h = int(np.clip(round(rng.normal(center_hour, spread)), 0, 23))
    return h


def generate_day(
    vehicle: SampledVehicle,
    rng: np.random.Generator,
    *,
    is_weekend: bool,
    is_holiday: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (km_per_hour, speed_per_hour, aggressive_events_per_hour)."""
    km = np.zeros(24, dtype=np.float32)
    speed = np.zeros(24, dtype=np.float32)
    aggr = np.zeros(24, dtype=np.int32)

    if is_holiday and rng.random() < 0.3:
        return km, speed, aggr

    mean_km = vehicle.weekend_mean_km if is_weekend else vehicle.weekday_mean_km
    daily_km = float(np.clip(rng.normal(mean_km, mean_km * 0.35), 0.0, 350.0))

    if rng.random() < 0.05:
        daily_km *= rng.uniform(2.0, 4.5)

    if daily_km < 1.0:
        return km, speed, aggr

    if not is_weekend:
        morning_h = _trip_window(rng, 7.5, 0.7)
        evening_h = _trip_window(rng, 18.0, 1.0)
        leg = daily_km / 2.0
        km[morning_h] += leg
        km[evening_h] += daily_km - leg
        speed[morning_h] = float(rng.uniform(28, 45))
        speed[evening_h] = float(rng.uniform(25, 42))
    else:
        n_trips = int(rng.integers(1, 4))
        starts = [_trip_window(rng, c, 1.2) for c in rng.uniform(10, 19, size=n_trips)]
        per_trip = daily_km / n_trips
        for h in starts:
            km[h] += per_trip * float(rng.uniform(0.7, 1.3))
            speed[h] = float(rng.uniform(30, 70))

    # Aggressive events: only during driving hours.
    p = vehicle.driver.aggressive_acceleration_prob
    for h in range(24):
        if km[h] > 0:
            aggr[h] = int(rng.binomial(8, p))
            if speed[h] == 0:
                speed[h] = float(rng.uniform(25, 45))

    return km, speed, aggr
