"""Shared fixtures.

Most tests run on a small in-memory simulated dataset. Heavy tests
(integration, end-to-end DB) are marked with `integration` and skipped
in the default CI run.
"""

from __future__ import annotations

import pandas as pd
import pytest

from eeie.simulation.engine import SimulationConfig, simulate_fleet


@pytest.fixture(scope="session")
def small_simulation():
    cfg = SimulationConfig(
        n_vehicles=5,
        months=1,
        seed=7,
        start_ts=pd.Timestamp("2025-01-01", tz="UTC"),
    )
    return simulate_fleet(cfg)
