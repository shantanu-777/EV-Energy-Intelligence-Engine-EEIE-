"""Battery health & SOH projection page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from eeie.ui_streamlit.lib.api import list_vehicles, post, render_insight
from eeie.ui_streamlit.lib.theme import PALETTE, hero, install_theme, section

install_theme(page_title="EEIE | Battery Health", icon=":battery:")

hero(
    "Battery Health",
    "Empirical calendar + cycle aging model fused with an XGBoost residual correction layer. "
    "Returns SOH trajectory plus actionable usage recommendations.",
)

vehicles = list_vehicles()
if not vehicles:
    st.error("No vehicles. Run the simulator first.")
    st.stop()

c1, c2 = st.columns([2, 1])
vehicle_id = c1.selectbox("Vehicle", options=[v["vehicle_id"] for v in vehicles], index=0)
horizon = c2.slider("Projection horizon (years)", 1.0, 10.0, 5.0, step=0.5)

if st.button("Estimate", type="primary", use_container_width=True):
    try:
        resp = post("/battery_health", {"vehicle_id": vehicle_id, "horizon_years": horizon})
    except Exception as exc:
        st.error(f"API call failed: {exc}")
        st.stop()

    section("Headline")
    cols = st.columns(4)
    cols[0].metric("Current SOH", f"{resp['current_soh'] * 100:.1f}%")
    cols[1].metric("Annual degradation", f"{resp['annual_degradation_pct']:.2f}%")
    cols[2].metric("SOH in 3y", f"{resp['projected_soh_3y'] * 100:.1f}%")
    cols[3].metric("SOH in 5y", f"{resp['projected_soh_5y'] * 100:.1f}%")

    section("Projected SOH trajectory")
    traj = pd.DataFrame(resp["trajectory"])
    fig = px.area(traj, x="years", y="soh", title=None)
    fig.update_traces(
        line={"color": PALETTE["accent_2"], "width": 2.5},
        fillcolor="rgba(91,141,239,0.18)",
    )
    fig.update_layout(height=340, yaxis={"tickformat": ".0%"})
    st.plotly_chart(fig, use_container_width=True)

    band = resp["recommended_soc_band"]
    st.info(
        f"Recommended SOC band: **{band[0] * 100:.0f}%-{band[1] * 100:.0f}%**. "
        f"DC fast charging cap: **{resp['recommended_dc_fast_cap_pct'] * 100:.0f}%** of total energy."
    )

    st.markdown("---")
    render_insight(resp["insight"])
