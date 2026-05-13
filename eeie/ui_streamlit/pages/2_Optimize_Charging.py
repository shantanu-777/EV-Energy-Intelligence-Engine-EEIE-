"""Charging schedule optimization page."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eeie.ui_streamlit.lib.api import list_vehicles, post, render_insight
from eeie.ui_streamlit.lib.theme import PALETTE, hero, install_theme, section

install_theme(page_title="EEIE | Optimize Charging", icon=":zap:")

hero(
    "Optimize Charging",
    "Minimize cost + battery wear + peak-tariff exposure under SOC and rate constraints. "
    "MILP solver (OR-Tools) and PPO reinforcement learning, switchable at request time.",
)

vehicles = list_vehicles()
if not vehicles:
    st.error("No vehicles. Run the simulator first.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
vehicle_id = c1.selectbox("Vehicle", options=[v["vehicle_id"] for v in vehicles], index=0)
optimizer = c2.selectbox("Optimizer", ["milp", "rl", "naive"])
horizon = c3.slider("Horizon (h)", 6, 48, 24, step=6)
target_soc = c4.slider("Target SOC", 0.4, 0.95, 0.80, step=0.05)

c5, c6 = st.columns(2)
lambda_wear = c5.slider("Wear penalty (lambda)", 0.0, 2.0, 0.5, step=0.1)
lambda_peak = c6.slider("Peak penalty (lambda)", 0.0, 2.0, 0.3, step=0.1)

if st.button("Optimize", type="primary", use_container_width=True):
    body = {
        "vehicle_id": vehicle_id,
        "optimizer": optimizer,
        "horizon_hours": horizon,
        "target_soc": target_soc,
        "lambda_wear": lambda_wear,
        "lambda_peak": lambda_peak,
    }
    try:
        resp = post("/optimize_charging", body)
    except Exception as exc:
        st.error(f"API call failed: {exc}")
        st.stop()

    section("Results")
    cols = st.columns(4)
    cols[0].metric("Plan cost", f"EUR {resp['total_cost_eur']:.2f}")
    cols[1].metric("Naive cost", f"EUR {resp['naive_cost_eur']:.2f}")
    cols[2].metric(
        "Savings vs naive",
        f"EUR {resp['savings_vs_naive_eur']:.2f}",
        delta=f"{resp['savings_vs_naive_pct'] * 100:.1f}%",
    )
    cols[3].metric("Achieved final SOC", f"{resp['soc_per_hour'][-1] * 100:.1f}%")

    section("Plan")
    plan_df = pd.DataFrame(
        {
            "hour": list(range(len(resp["power_per_hour_kw"]))),
            "EEIE plan (kW)": resp["power_per_hour_kw"],
        }
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plan_df["hour"],
            y=plan_df["EEIE plan (kW)"],
            name="Charging power",
            marker_color=PALETTE["accent"],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(resp["soc_per_hour"]))),
            y=[s * 100 for s in resp["soc_per_hour"]],
            mode="lines+markers",
            name="SOC (%)",
            line={"color": PALETTE["good"], "width": 2.5},
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=380,
        xaxis={"title": "hour"},
        yaxis={"title": "Power (kW)"},
        yaxis2={
            "title": "SOC (%)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "range": [0, 100],
        },
        legend={"orientation": "h", "y": 1.12, "x": 0},
        title="Charging plan & SOC trajectory",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    render_insight(resp["insight"])
