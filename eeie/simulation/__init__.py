"""Synthetic EV fleet simulator.

Generates a complete 12-month, hourly-resolution dataset of vehicle
telemetry, weather, time-of-use tariffs, and charging events for ~100
heterogeneous EVs. Used in Phase 1 in lieu of real OEM telemetry.

CLI:

    python -m eeie.simulation.run --vehicles 100 --months 12
"""

from eeie.simulation.engine import SimulationConfig, simulate_fleet

__all__ = ["SimulationConfig", "simulate_fleet"]
