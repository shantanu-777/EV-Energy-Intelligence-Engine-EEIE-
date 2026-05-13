"""Range / SOC depletion forecasting engine."""

from eeie.models.range.predict import RangePrediction, predict_range
from eeie.models.range.xgb import RangeXGB

__all__ = ["RangePrediction", "RangeXGB", "predict_range"]
