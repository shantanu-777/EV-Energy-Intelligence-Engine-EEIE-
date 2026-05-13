"""Range / SOC depletion forecast page."""

from __future__ import annotations

import streamlit as st

from eeie.ui_streamlit.lib.api import list_vehicles, post, render_insight
from eeie.ui_streamlit.lib.theme import hero, install_theme, section

install_theme(page_title="EEIE | Predict Range", icon=":battery:")

hero(
    "Predict Range",
    "24-hour SOC depletion forecast, days-to-recharge, and undercharge risk - "
    "XGBoost baseline ensembled with an LSTM sequence model.",
)

vehicles = list_vehicles()
if not vehicles:
    st.error("No vehicles. Run the simulator first.")
    st.stop()

controls = st.columns([2, 1, 2])
vehicle_id = controls[0].selectbox("Vehicle", options=[v["vehicle_id"] for v in vehicles], index=0)
model = controls[1].radio("Model", ["xgb", "lstm"], horizontal=True)
lookback = controls[2].slider("Lookback hours", min_value=24, max_value=240, value=72, step=24)

if st.button("Predict", type="primary", use_container_width=True):
    body = {"vehicle_id": vehicle_id, "model": model, "lookback_hours": lookback}
    try:
        resp = post("/predict_range", body)
    except Exception as exc:
        st.error(f"API call failed: {exc}")
        st.stop()

    section("Forecast")
    cols = st.columns(4)
    cols[0].metric("Current SOC", f"{resp['current_soc'] * 100:.1f}%")
    cols[1].metric("24h SOC drop", f"{resp['soc_drop_24h'] * 100:.1f}%")
    cols[2].metric("Days until recharge", f"{resp['days_until_recharge']:.2f}")
    cols[3].metric("Undercharge risk", f"{resp['undercharge_risk'] * 100:.0f}%")

    st.markdown("---")
    render_insight(resp["insight"])
