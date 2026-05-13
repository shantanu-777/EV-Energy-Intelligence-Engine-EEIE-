"""Structured `Insight` model returned alongside every prediction.

This is the contract every API response carries: a ranked list of the
most influential features, a financial estimate, a battery-health estimate,
and a confidence score. Predictions without this attachment are not
allowed to leave the platform.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InsightFactor(BaseModel):
    """A single contributing feature."""

    name: str
    importance: float = Field(ge=0)
    direction: str = Field(description="increases | decreases | neutral")
    plain_language: str = ""


class Insight(BaseModel):
    """Structured explanation that accompanies every prediction."""

    model_config = ConfigDict(extra="forbid")

    top_factors: list[InsightFactor] = Field(default_factory=list, max_length=10)
    estimated_financial_impact_eur: float | None = None
    estimated_battery_health_impact_pct: float | None = None
    confidence: float = Field(ge=0, le=1, default=0.5)
    rationale: str = ""


# ---------- Per-engine helpers ----------


def _factors(
    importances: dict[str, float],
    direction: str = "increases",
    top: int = 5,
) -> list[InsightFactor]:
    items = sorted(importances.items(), key=lambda x: -x[1])[:top]
    return [
        InsightFactor(
            name=k,
            importance=float(max(v, 0.0)),
            direction=direction,
        )
        for k, v in items
    ]


def range_insight(
    *,
    feature_contributions: dict[str, float],
    confidence: float,
    days_until_recharge: float,
    undercharge_risk: float,
) -> Insight:
    return Insight(
        top_factors=_factors(feature_contributions, "increases"),
        estimated_financial_impact_eur=None,
        estimated_battery_health_impact_pct=None,
        confidence=confidence,
        rationale=(
            f"Forecast suggests recharging in ~{days_until_recharge:.1f} days. "
            f"Risk of running below safe SOC: {undercharge_risk * 100:.0f}%."
        ),
    )


def optimization_insight(
    *,
    feature_contributions: dict[str, float],
    total_cost_eur: float,
    naive_cost_eur: float,
    confidence: float,
) -> Insight:
    savings = max(0.0, naive_cost_eur - total_cost_eur)
    return Insight(
        top_factors=[
            InsightFactor(name=k, importance=float(v), direction="increases")
            for k, v in feature_contributions.items()
        ][:5],
        estimated_financial_impact_eur=savings,
        estimated_battery_health_impact_pct=None,
        confidence=confidence,
        rationale=(f"Recommended schedule saves ~EUR {savings:.2f} versus charging immediately."),
    )


def battery_insight(
    *,
    contributing_factors: dict[str, float],
    annual_degradation_pct: float,
    projected_soh_5y: float,
    confidence: float,
) -> Insight:
    return Insight(
        top_factors=_factors(contributing_factors, "increases"),
        estimated_financial_impact_eur=None,
        estimated_battery_health_impact_pct=annual_degradation_pct,
        confidence=confidence,
        rationale=(
            f"At current habits, annual SOH loss ~{annual_degradation_pct:.2f}%; "
            f"projected SOH in 5 years ~{projected_soh_5y * 100:.1f}%."
        ),
    )


def behavior_insight(
    *,
    feature_contributions: dict[str, float],
    potential_savings_pct: float,
    cluster_label: str,
    confidence: float,
) -> Insight:
    return Insight(
        top_factors=_factors(feature_contributions, "increases"),
        estimated_financial_impact_eur=None,
        estimated_battery_health_impact_pct=None,
        confidence=confidence,
        rationale=(
            f"Driver clusters as '{cluster_label}'. Adopting efficient habits could "
            f"reduce consumption by ~{potential_savings_pct * 100:.1f}%."
        ),
    )


def cost_insight(
    *,
    monthly_eur: float,
    annual_eur: float,
    ice_comparison_eur: float,
    confidence: float,
) -> Insight:
    return Insight(
        top_factors=[],
        estimated_financial_impact_eur=ice_comparison_eur - annual_eur,
        estimated_battery_health_impact_pct=None,
        confidence=confidence,
        rationale=(
            f"EV electricity cost ~EUR {monthly_eur:.2f}/mo. Annual ~EUR {annual_eur:.2f}, "
            f"versus comparable ICE ~EUR {ice_comparison_eur:.2f}."
        ),
    )
