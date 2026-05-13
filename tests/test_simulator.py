"""Schema and basic-physics checks on the simulator output."""

from __future__ import annotations


def test_simulation_shape(small_simulation):
    res = small_simulation
    assert len(res.vehicles) == 5
    assert len(res.weather) == 1 * 30 * 24
    assert len(res.tariffs) == 1 * 30 * 24
    assert not res.telemetry.empty
    assert (res.telemetry["soc"] >= 0).all() and (res.telemetry["soc"] <= 1).all()
    assert (res.telemetry["soh"] > 0.5).all() and (res.telemetry["soh"] <= 1.0).all()


def test_telemetry_columns(small_simulation):
    expected = {
        "ts",
        "vehicle_id",
        "soc",
        "odometer_km",
        "km_driven_hour",
        "energy_consumed_kwh",
        "energy_charged_kwh",
        "battery_temp_c",
        "ambient_temp_c",
        "avg_speed_kmh",
        "aggressive_event_count",
        "is_charging",
        "is_driving",
        "soh",
    }
    assert expected.issubset(set(small_simulation.telemetry.columns))


def test_tariff_rates_positive(small_simulation):
    assert (small_simulation.tariffs["rate_eur_per_kwh"] > 0).all()
    assert small_simulation.tariffs["tier"].isin({"peak", "shoulder", "off_peak"}).all()


def test_charging_events_consistent(small_simulation):
    events = small_simulation.charging_events
    if not events.empty:
        assert (events["end_soc"] >= events["start_soc"] - 1e-6).all()
        assert (events["energy_kwh"] >= 0).all()
        assert (events["cost_eur"] >= 0).all()


def test_per_vehicle_horizon(small_simulation):
    tel = small_simulation.telemetry
    counts = tel.groupby("vehicle_id").size()
    assert (counts == 30 * 24).all()
