"""Request and response models for the EEIE API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from eeie.explainability.insight import Insight


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- /predict_range ----------


class PredictRangeRequest(_Base):
    vehicle_id: str
    lookback_hours: int = Field(default=48, ge=24, le=336)
    model: Literal["xgb", "lstm"] = "xgb"


class PredictRangeResponse(_Base):
    vehicle_id: str
    current_soc: float
    soc_drop_24h: float
    soc_drop_24h_lower: float
    soc_drop_24h_upper: float
    days_until_recharge: float
    undercharge_risk: float
    model_used: str
    insight: Insight


# ---------- /optimize_charging ----------


class OptimizeChargingRequest(_Base):
    vehicle_id: str
    horizon_hours: int = Field(default=24, ge=1, le=168)
    target_soc: float = Field(default=0.80, gt=0, le=1)
    initial_soc: float | None = None
    optimizer: Literal["milp", "rl", "naive"] | None = None
    lambda_wear: float = Field(default=0.5, ge=0)
    lambda_peak: float = Field(default=0.3, ge=0)


class OptimizeChargingResponse(_Base):
    vehicle_id: str
    optimizer: str
    feasible: bool
    power_per_hour_kw: list[float]
    soc_per_hour: list[float]
    total_cost_eur: float
    naive_cost_eur: float
    savings_vs_naive_eur: float
    savings_vs_naive_pct: float
    chosen_window_start_hour: int | None
    chosen_window_end_hour: int | None
    chosen_avg_rate_kw: float
    insight: Insight


# ---------- /battery_health ----------


class BatteryHealthRequest(_Base):
    vehicle_id: str
    horizon_years: float = Field(default=5.0, gt=0, le=15)


class BatteryHealthResponse(_Base):
    vehicle_id: str
    current_soh: float
    annual_degradation_pct: float
    projected_soh_3y: float
    projected_soh_5y: float
    trajectory: list[dict[str, float]]
    recommended_soc_band: tuple[float, float]
    recommended_dc_fast_cap_pct: float
    insight: Insight


# ---------- /cost_analysis ----------


class CostAnalysisRequest(_Base):
    vehicle_id: str
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    ice_price_eur_per_liter: float = 1.65
    ice_consumption_l_per_100km: float = 6.5


class CostAnalysisResponse(_Base):
    vehicle_id: str
    period_days: int
    total_kwh_charged: float
    total_cost_eur: float
    monthly_cost_eur: float
    annual_cost_eur: float
    peak_hour_share_pct: float
    ice_annual_cost_eur: float
    annual_savings_vs_ice_eur: float
    insight: Insight


# ---------- /behavior_analysis ----------


class BehaviorAnalysisRequest(_Base):
    vehicle_id: str


class BehaviorAnalysisResponse(_Base):
    vehicle_id: str
    cluster_id: int
    cluster_label: str
    predicted_kwh_per_100km: float
    efficient_kwh_per_100km: float
    potential_savings_kwh_per_100km: float
    potential_savings_pct: float
    insight: Insight


# ---------- Health ----------


class HealthResponse(_Base):
    status: str = "ok"
    version: str
    feature_version: str
