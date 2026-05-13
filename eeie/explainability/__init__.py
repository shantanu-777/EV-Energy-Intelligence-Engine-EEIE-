"""Explainability: SHAP wrappers, partial dependence, unified `Insight` model."""

from eeie.explainability.insight import (
    Insight,
    InsightFactor,
    battery_insight,
    behavior_insight,
    cost_insight,
    optimization_insight,
    range_insight,
)
from eeie.explainability.pdp import partial_dependence
from eeie.explainability.shap_engine import (
    shap_summary_for_torch,
    shap_summary_for_tree,
)

__all__ = [
    "Insight",
    "InsightFactor",
    "battery_insight",
    "behavior_insight",
    "cost_insight",
    "optimization_insight",
    "partial_dependence",
    "range_insight",
    "shap_summary_for_torch",
    "shap_summary_for_tree",
]
