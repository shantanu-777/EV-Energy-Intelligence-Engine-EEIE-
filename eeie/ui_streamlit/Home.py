"""EEIE Executive Summary - the page you open with in a demo."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from eeie.ui_streamlit.lib.api import (
    fleet_stats,
    fleet_timeseries,
    health,
    list_vehicles,
)
from eeie.ui_streamlit.lib.theme import COLORWAY, PALETTE, hero, install_theme, pills, section

install_theme(page_title="EEIE | Executive Summary", icon=":zap:")

hero(
    "EV Energy Intelligence Engine",
    "Predict range. Optimize charging. Forecast battery health. "
    "Every answer ships with a confidence and a SHAP-grounded explanation.",
)

try:
    h = health()
    pills(
        [
            (f"API v{h.get('version', '?')}", "good"),
            (f"features {h.get('feature_version', '?')}", "accent"),
            ("simulator: 100 vehicles x 12 months", "accent"),
            ("models: XGBoost + LSTM + TFT + MILP + PPO + SHAP", "accent"),
        ]
    )
except Exception as exc:
    pills([(f"API not reachable: {exc}", "bad")])
    st.stop()

st.markdown("")

try:
    stats = fleet_stats()
except Exception as exc:
    st.error(f"Could not load fleet stats: {exc}")
    st.stop()

if stats["n_vehicles"] == 0:
    st.warning(
        "Database is empty. Run the simulator: "
        "`docker compose exec api python -m eeie.simulation.run`."
    )
    st.stop()


section("Fleet KPIs")
k = st.columns(5)
k[0].metric("Vehicles", f"{stats['n_vehicles']:,}")
k[1].metric("Annual energy", f"{stats['annual_energy_kwh']:,.0f} kWh")
k[2].metric(
    "Annual electricity cost",
    f"EUR {stats['annual_cost_eur']:,.0f}",
    delta=f"period {stats['period_days']} days",
    delta_color="off",
)
k[3].metric("Avg SOH (fleet)", f"{stats['avg_soh'] * 100:.1f}%")
k[4].metric(
    "Off-peak share",
    f"{100.0 - stats['peak_share_pct']:.0f}%",
    delta=f"{-stats['peak_share_pct']:.1f}% peak" if stats["peak_share_pct"] else None,
    delta_color="off",
)

st.markdown("")

left, right = st.columns([1, 1])
with left:
    section("Fleet mix")
    arch_df = (
        pd.DataFrame(
            {
                "archetype": list(stats["archetype_mix"].keys()),
                "count": list(stats["archetype_mix"].values()),
            }
        ).sort_values("count", ascending=False)
        if stats["archetype_mix"]
        else pd.DataFrame({"archetype": [], "count": []})
    )
    if not arch_df.empty:
        fig = px.pie(
            arch_df,
            names="archetype",
            values="count",
            hole=0.55,
            color_discrete_sequence=COLORWAY,
        )
        fig.update_traces(textposition="outside", textinfo="label+percent")
        fig.update_layout(
            title="Vehicle archetypes",
            showlegend=False,
            margin={"l": 10, "r": 10, "t": 50, "b": 10},
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

with right:
    section("Driver profiles")
    drv_df = (
        pd.DataFrame(
            {
                "driver": list(stats["driver_mix"].keys()),
                "count": list(stats["driver_mix"].values()),
            }
        ).sort_values("count", ascending=False)
        if stats["driver_mix"]
        else pd.DataFrame({"driver": [], "count": []})
    )
    if not drv_df.empty:
        fig = px.pie(
            drv_df,
            names="driver",
            values="count",
            hole=0.55,
            color_discrete_sequence=COLORWAY[1:],
        )
        fig.update_traces(textposition="outside", textinfo="label+percent")
        fig.update_layout(
            title="Driver profile distribution",
            showlegend=False,
            margin={"l": 10, "r": 10, "t": 50, "b": 10},
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)


section("Fleet energy & cost over time")
bucket = st.radio(
    "Aggregation", options=["day", "week", "month"], horizontal=True, index=1, key="bucket"
)
try:
    ts = fleet_timeseries(bucket=bucket)
except Exception as exc:
    st.warning(f"Time-series unavailable: {exc}")
    ts = []

if ts:
    ts_df = pd.DataFrame(ts)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ts_df["bucket"],
            y=ts_df["off_peak_kwh"],
            name="Off-peak kWh",
            marker_color=PALETTE["good"],
        )
    )
    fig.add_trace(
        go.Bar(
            x=ts_df["bucket"],
            y=ts_df["peak_kwh"],
            name="Peak kWh",
            marker_color=PALETTE["bad"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts_df["bucket"],
            y=ts_df["cost_eur"],
            name="Cost (EUR)",
            mode="lines+markers",
            line={"color": PALETTE["accent"], "width": 2.5},
            yaxis="y2",
        )
    )
    fig.update_layout(
        barmode="stack",
        title=f"Charging energy by {bucket} (peak vs off-peak) and cost",
        height=380,
        yaxis={"title": "Energy (kWh)"},
        yaxis2={
            "title": "Cost (EUR)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
section("What this platform does")
c1, c2, c3 = st.columns(3)
c1.markdown(
    f"""
<div class='eeie-card'>
<h4 style='margin-top:0;color:{PALETTE["accent"]}'>Predict</h4>
<p style='color:{PALETTE["muted"]}'>SOC depletion forecast over 24h.
LSTM sequence model + XGBoost baseline, ensembled with calibrated confidence.</p>
</div>
""",
    unsafe_allow_html=True,
)
c2.markdown(
    f"""
<div class='eeie-card'>
<h4 style='margin-top:0;color:{PALETTE["accent"]}'>Optimize</h4>
<p style='color:{PALETTE["muted"]}'>Hourly charging plan that minimizes cost, peak exposure, and battery wear.
MILP solver (OR-Tools) plus PPO reinforcement learning.</p>
</div>
""",
    unsafe_allow_html=True,
)
c3.markdown(
    f"""
<div class='eeie-card'>
<h4 style='margin-top:0;color:{PALETTE["accent"]}'>Explain</h4>
<p style='color:{PALETTE["muted"]}'>Every prediction returns a SHAP-grounded Insight: top factors,
financial impact, battery health impact, and confidence.</p>
</div>
""",
    unsafe_allow_html=True,
)


with st.expander("Fleet directory (first 50 vehicles)"):
    try:
        vehicles = list_vehicles()
        st.dataframe(pd.DataFrame(vehicles).head(50), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Could not load vehicles: {exc}")
