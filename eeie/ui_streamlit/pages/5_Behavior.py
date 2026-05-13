"""Driver behavior cluster + consumption forecast page."""

from __future__ import annotations

import streamlit as st

from eeie.ui_streamlit.lib.api import list_vehicles, post, render_insight
from eeie.ui_streamlit.lib.theme import PALETTE, hero, install_theme, pills, section

install_theme(page_title="EEIE | Behavior", icon=":car:")

hero(
    "Driver Behavior",
    "K-means driver clustering plus XGBoost consumption regression. "
    "Quantifies the gap between this driver and the efficient cohort.",
)

vehicles = list_vehicles()
if not vehicles:
    st.error("No vehicles. Run the simulator first.")
    st.stop()

vehicle_id = st.selectbox("Vehicle", options=[v["vehicle_id"] for v in vehicles], index=0)

if st.button("Analyze", type="primary", use_container_width=True):
    try:
        resp = post("/behavior_analysis", {"vehicle_id": vehicle_id})
    except Exception as exc:
        st.error(f"API call failed: {exc}")
        st.stop()

    section("Headline")
    pills(
        [
            (f"cluster: {resp['cluster_label']}", "accent"),
            (f"id: {resp['cluster_id']}", ""),
        ]
    )
    st.markdown("")

    cols = st.columns(4)
    cols[0].metric("Predicted kWh / 100km", f"{resp['predicted_kwh_per_100km']:.1f}")
    cols[1].metric("Efficient kWh / 100km", f"{resp['efficient_kwh_per_100km']:.1f}")
    cols[2].metric(
        "Savings potential",
        f"{resp['potential_savings_pct'] * 100:.1f}%",
        delta=f"-{resp['potential_savings_kwh_per_100km']:.2f} kWh/100km",
    )
    cols[3].metric(
        "Gap to efficient",
        f"{(resp['predicted_kwh_per_100km'] - resp['efficient_kwh_per_100km']):.2f} kWh/100km",
        delta_color="off",
    )

    if resp["potential_savings_pct"] > 0.05:
        st.markdown(
            f"<div class='eeie-card'>"
            f"<b style='color:{PALETTE['warn']}'>Coaching opportunity</b><br>"
            f"This driver is consuming {resp['potential_savings_pct'] * 100:.0f}% more energy than the "
            f"efficient cohort with comparable trip profiles. Smoother throttle and lower peak speeds "
            f"could close most of the gap."
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    render_insight(resp["insight"])
