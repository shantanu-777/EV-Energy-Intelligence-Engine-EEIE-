"""Feature engineering layer.

All feature builders are pure functions over pandas DataFrames. They are
versioned via the `FEATURE_VERSION` constant; bumping the version
invalidates any persisted feature cache and signals a retraining cycle.
"""

from eeie.features.builders import (
    FEATURE_VERSION,
    build_behavior_features,
    build_demand_features,
    build_hourly_features,
    build_range_features,
    load_features_from_db,
)

__all__ = [
    "FEATURE_VERSION",
    "build_behavior_features",
    "build_demand_features",
    "build_hourly_features",
    "build_range_features",
    "load_features_from_db",
]
