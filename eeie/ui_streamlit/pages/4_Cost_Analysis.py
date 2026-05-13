"""EV electricity cost vs ICE comparison page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from eeie.ui_streamlit.lib.api import list_vehicles, post, render_insight
from eeie.ui_streamlit.lib.theme import PALETTE, hero, install_theme, section

install_theme(page_title="EEIE | Cost Analysis", icon=":euro:")

hero(
    "Cost Analysis",
    "Isolate EV electricity cost over the observation window, project to annual run-rate, "
    "and compare against an ICE baseline using vehicle-specific efficiency.",
)

vehicles = list_vehicles()
if not vehicles:
    st.error("No vehicles. Run the simulator first.")
    st.stop()

c1, c2, c3 = st.columns([2, 1, 1])
vehicle_id = c1.selectbox("Vehicle", options=[v["vehicle_id"] for v in vehicles], index=0)
ice_price = c2.number_input("Petrol price (EUR/L)", value=1.65, step=0.05)
ice_cons = c3.number_input("ICE consumption (L/100km)", value=6.5, step=0.1)

if st.button("Analyze", type="primary", use_container_width=True):
    body = {
        "vehicle_id": vehicle_id,
        "ice_price_eur_per_liter": ice_price,
        "ice_consumption_l_per_100km": ice_cons,
    }
    try:
        resp = post("/cost_analysis", body)
    except Exception as exc:
        st.error(f"API call failed: {exc}")
        st.stop()

    section("Headline")
    cols = st.columns(4)
    cols[0].metric("Total kWh charged", f"{resp['total_kwh_charged']:,.0f}")
    cols[1].metric("Monthly cost", f"EUR {resp['monthly_cost_eur']:,.2f}")
    cols[2].metric("Annual cost", f"EUR {resp['annual_cost_eur']:,.2f}")
    cols[3].metric(
        "Annual savings vs ICE",
        f"EUR {resp['annual_savings_vs_ice_eur']:,.2f}",
        delta=f"vs ICE EUR {resp['ice_annual_cost_eur']:,.0f}",
        delta_color="off",
    )

    section("EV vs ICE annual cost")
    df = pd.DataFrame(
        {
            "powertrain": ["EV (EEIE)", "ICE"],
            "annual_eur": [resp["annual_cost_eur"], resp["ice_annual_cost_eur"]],
        }
    )
    fig = px.bar(
        df,
        x="annual_eur",
        y="powertrain",
        orientation="h",
        color="powertrain",
        color_discrete_map={"EV (EEIE)": PALETTE["good"], "ICE": PALETTE["bad"]},
        text="annual_eur",
    )
    fig.update_traces(texttemplate="EUR %{x:,.0f}", textposition="outside")
    fig.update_layout(
        height=260, showlegend=False, xaxis={"title": "EUR / year"}, yaxis={"title": ""}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"Peak-hour share of charged energy: **{resp['peak_hour_share_pct']:.1f}%**")

    st.markdown("---")
    render_insight(resp["insight"])
