"""Static catalog of time-of-use electricity tariffs.

The simulator and the cost analyzer both consume these. Rates are expressed
in EUR/kWh (the simulator can be retuned for other currencies; the units
are carried in `currency`). The tariff defines hour-of-day windows where
each rate applies; the simulator broadcasts the schedule across the
simulation horizon.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TariffWindow(BaseModel):
    """A contiguous window within a day with a single rate."""

    start_hour: int = Field(ge=0, le=23)
    end_hour: int = Field(ge=1, le=24, description="exclusive")
    rate_eur_per_kwh: float = Field(ge=0)
    tier: str = Field(description="off_peak | shoulder | peak")


class TariffSchedule(BaseModel):
    """A weekday/weekend schedule of windows for a single utility plan."""

    tariff_id: str
    label: str
    currency: str = "EUR"
    weekday_windows: list[TariffWindow]
    weekend_windows: list[TariffWindow]
    standing_charge_eur_per_day: float = 0.0

    def rate_at(self, hour: int, is_weekend: bool) -> tuple[float, str]:
        """Return (rate, tier) for the given hour-of-day and day-type."""
        windows = self.weekend_windows if is_weekend else self.weekday_windows
        for w in windows:
            if w.start_hour <= hour < w.end_hour:
                return w.rate_eur_per_kwh, w.tier
        raise ValueError(f"No tariff window covers hour {hour}")


DEFAULT_TARIFF = TariffSchedule(
    tariff_id="eu_tou_default",
    label="EU-style ToU default",
    weekday_windows=[
        TariffWindow(start_hour=0, end_hour=7, rate_eur_per_kwh=0.12, tier="off_peak"),
        TariffWindow(start_hour=7, end_hour=10, rate_eur_per_kwh=0.32, tier="peak"),
        TariffWindow(start_hour=10, end_hour=17, rate_eur_per_kwh=0.22, tier="shoulder"),
        TariffWindow(start_hour=17, end_hour=21, rate_eur_per_kwh=0.38, tier="peak"),
        TariffWindow(start_hour=21, end_hour=24, rate_eur_per_kwh=0.18, tier="shoulder"),
    ],
    weekend_windows=[
        TariffWindow(start_hour=0, end_hour=8, rate_eur_per_kwh=0.12, tier="off_peak"),
        TariffWindow(start_hour=8, end_hour=22, rate_eur_per_kwh=0.20, tier="shoulder"),
        TariffWindow(start_hour=22, end_hour=24, rate_eur_per_kwh=0.14, tier="off_peak"),
    ],
    standing_charge_eur_per_day=0.30,
)


FLAT_TARIFF = TariffSchedule(
    tariff_id="flat",
    label="Flat rate (no time-of-use)",
    weekday_windows=[
        TariffWindow(start_hour=0, end_hour=24, rate_eur_per_kwh=0.25, tier="shoulder"),
    ],
    weekend_windows=[
        TariffWindow(start_hour=0, end_hour=24, rate_eur_per_kwh=0.25, tier="shoulder"),
    ],
)


TARIFFS_BY_ID: dict[str, TariffSchedule] = {
    DEFAULT_TARIFF.tariff_id: DEFAULT_TARIFF,
    FLAT_TARIFF.tariff_id: FLAT_TARIFF,
}


def average_rate(schedule: TariffSchedule) -> float:
    """Return a coarse hourly-weighted average rate (used as a heuristic)."""
    total = 0.0
    hours = 0
    for w in schedule.weekday_windows:
        h = w.end_hour - w.start_hour
        total += h * w.rate_eur_per_kwh * 5
        hours += h * 5
    for w in schedule.weekend_windows:
        h = w.end_hour - w.start_hour
        total += h * w.rate_eur_per_kwh * 2
        hours += h * 2
    return total / hours if hours else 0.0
