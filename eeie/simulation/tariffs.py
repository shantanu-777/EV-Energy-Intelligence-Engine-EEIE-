"""Materialize a `TariffSchedule` as an hourly time-series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eeie.config.tariffs import TariffSchedule


def materialize_tariff(
    schedule: TariffSchedule,
    *,
    start: pd.Timestamp,
    periods: int,
) -> pd.DataFrame:
    """Return an hourly tariff frame: ts, tariff_id, rate_eur_per_kwh, tier."""
    idx = pd.date_range(start=start, periods=periods, freq="h", tz="UTC")
    weekend_mask = np.asarray(idx.dayofweek >= 5)
    hours = np.asarray(idx.hour)

    rate = np.zeros(periods, dtype=np.float32)
    tier = np.empty(periods, dtype=object)
    for i in range(periods):
        r, t = schedule.rate_at(int(hours[i]), bool(weekend_mask[i]))
        rate[i] = r
        tier[i] = t

    return pd.DataFrame(
        {
            "ts": idx,
            "tariff_id": schedule.tariff_id,
            "rate_eur_per_kwh": rate,
            "tier": tier,
        }
    )
