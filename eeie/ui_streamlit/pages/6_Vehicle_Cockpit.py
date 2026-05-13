"""Vehicle Cockpit - all five EEIE engines firing on one screen, for one vehicle.

This is the demo page. Pick a car, hit Run, and the platform answers every
question it knows how to answer: range, optimal charging, battery health,
annual cost, and driver behavior - in one synchronized view.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from eeie.ui_streamlit.lib.api import list_vehicles, post
from eeie.ui_streamlit.lib.theme import PALETTE, hero, install_theme, pills, section

install_theme(page_title="EEIE | Vehicle Cockpit", icon=":battery:")

hero(
    "Vehicle Cockpit",
    "One vehicle. All five engines. SHAP-grounded reasoning for every number on the screen.",
)


vehicles = list_vehicles()
if not vehicles:
    st.error("No vehicles in the database. Run the simulator first.")
    st.stop()

top = st.columns([2, 1, 1])
vehicle_id = top[0].selectbox(
    "Vehicle",
    options=[v["vehicle_id"] for v in vehicles],
    index=0,
    format_func=lambda vid: f"{vid}",
)
horizon = top[1].slider("Charging horizon (h)", 6, 48, 24, step=6)
target_soc = top[2].slider("Target SOC", 0.50, 0.95, 0.80, step=0.05)

selected = next((v for v in vehicles if v["vehicle_id"] == vehicle_id), None)
if selected:
    pills(
        [
            (f"archetype: {selected['archetype_id']}", "accent"),
            (f"driver: {selected['driver_profile_id']}", "accent"),
            (f"battery: {selected['battery_capacity_kwh']:.0f} kWh", "accent"),
            (f"tariff: {selected['tariff_id']}", "accent"),
        ]
    )

run = st.button("Run all five engines", type="primary", use_container_width=True)
if not run:
    st.info(
        "Press the button above to call every EEIE engine for this vehicle in parallel. "
        "Typical end-to-end latency is well under one second."
    )
    st.stop()


def _gauge(
    value: float, title: str, *, vmin: float = 0.0, vmax: float = 1.0, suffix: str = "%"
) -> go.Figure:
    val = value * 100.0 if vmax <= 1.0 else value
    rng_max = vmax * 100.0 if vmax <= 1.0 else vmax
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix": suffix, "font": {"color": PALETTE["text"]}},
            title={"text": title, "font": {"color": PALETTE["muted"], "size": 14}},
            gauge={
                "axis": {"range": [vmin, rng_max], "tickcolor": PALETTE["muted"]},
                "bar": {"color": PALETTE["accent"]},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [vmin, 0.4 * rng_max], "color": "#1A1024"},
                    {"range": [0.4 * rng_max, 0.7 * rng_max], "color": "#162033"},
                    {"range": [0.7 * rng_max, rng_max], "color": "#0E2A2E"},
                ],
            },
        )
    )
    fig.update_layout(
        height=220,
        margin={"l": 20, "r": 20, "t": 40, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


with st.spinner("Calling EEIE engines..."):
    errors: list[str] = []

    def _safe(path: str, body: dict) -> dict | None:
        try:
            return post(path, body)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            return None

    rng_resp = _safe(
        "/predict_range",
        {"vehicle_id": vehicle_id, "model": "xgb", "lookback_hours": 72},
    )
    bat_resp = _safe(
        "/battery_health",
        {"vehicle_id": vehicle_id, "horizon_years": 5.0},
    )
    opt_resp = _safe(
        "/optimize_charging",
        {
            "vehicle_id": vehicle_id,
            "optimizer": "milp",
            "horizon_hours": horizon,
            "target_soc": target_soc,
            "lambda_wear": 0.5,
            "lambda_peak": 0.3,
        },
    )
    cost_resp = _safe(
        "/cost_analysis",
        {
            "vehicle_id": vehicle_id,
            "ice_price_eur_per_liter": 1.65,
            "ice_consumption_l_per_100km": 6.5,
        },
    )
    beh_resp = _safe("/behavior_analysis", {"vehicle_id": vehicle_id})

for err in errors:
    st.warning(err)


section("Live state")
g1, g2, g3, g4 = st.columns(4)
if rng_resp is not None:
    with g1:
        st.plotly_chart(_gauge(rng_resp["current_soc"], "Current SOC"), use_container_width=True)
if bat_resp is not None:
    with g2:
        st.plotly_chart(_gauge(bat_resp["current_soh"], "Battery SOH"), use_container_width=True)
if rng_resp is not None:
    with g3:
        st.plotly_chart(
            _gauge(
                rng_resp["undercharge_risk"],
                "Undercharge risk (24h)",
            ),
            use_container_width=True,
        )
if beh_resp is not None:
    with g4:
        st.markdown(
            f"<div class='eeie-card' style='height:220px;display:flex;flex-direction:column;justify-content:center'>"
            f"<div style='color:{PALETTE['muted']};font-size:0.78rem;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:6px'>Driver cluster</div>"
            f"<div style='font-size:1.6rem;font-weight:700;color:{PALETTE['accent']}'>{beh_resp['cluster_label']}</div>"
            f"<div style='color:{PALETTE['text']};margin-top:8px;font-size:0.95rem'>"
            f"{beh_resp['predicted_kwh_per_100km']:.1f} kWh / 100km</div>"
            f"<div style='color:{PALETTE['muted']};font-size:0.85rem'>"
            f"vs efficient {beh_resp['efficient_kwh_per_100km']:.1f} - "
            f"savings potential {beh_resp['potential_savings_pct'] * 100:.0f}%</div>"
            "</div>",
            unsafe_allow_html=True,
        )


section("Range forecast (next 24h)")
if rng_resp is not None:
    rc = st.columns(4)
    rc[0].metric("24h SOC drop", f"{rng_resp['soc_drop_24h'] * 100:.1f}%")
    rc[1].metric(
        "Range (lo / hi)",
        f"{rng_resp['soc_drop_24h_lower'] * 100:.1f}% / {rng_resp['soc_drop_24h_upper'] * 100:.1f}%",
    )
    rc[2].metric("Days until recharge", f"{rng_resp['days_until_recharge']:.2f}")
    rc[3].metric("Model", rng_resp["model_used"].upper())


section("Optimal charging plan")
if opt_resp is not None:
    oc = st.columns(4)
    savings_eur = opt_resp["savings_vs_naive_eur"]
    oc[0].metric("Plan cost", f"EUR {opt_resp['total_cost_eur']:.2f}")
    oc[1].metric("Naive cost", f"EUR {opt_resp['naive_cost_eur']:.2f}")
    oc[2].metric(
        "Savings vs naive",
        f"EUR {savings_eur:.2f}",
        delta=f"{opt_resp['savings_vs_naive_pct'] * 100:.1f}%",
    )
    oc[3].metric("Achieved final SOC", f"{opt_resp['soc_per_hour'][-1] * 100:.1f}%")

    plan_df = pd.DataFrame(
        {
            "hour": list(range(len(opt_resp["power_per_hour_kw"]))),
            "EEIE plan (kW)": opt_resp["power_per_hour_kw"],
        }
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plan_df["hour"],
            y=plan_df["EEIE plan (kW)"],
            name="EEIE plan",
            marker_color=PALETTE["accent"],
        )
    )
    soc_x = list(range(len(opt_resp["soc_per_hour"])))
    fig.add_trace(
        go.Scatter(
            x=soc_x,
            y=[s * 100 for s in opt_resp["soc_per_hour"]],
            name="SOC (%)",
            mode="lines+markers",
            line={"color": PALETTE["good"], "width": 2.5},
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Charging plan: kW per hour and SOC trajectory",
        xaxis={"title": "hour"},
        yaxis={"title": "Power (kW)"},
        yaxis2={
            "title": "SOC (%)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "range": [0, 100],
        },
        height=360,
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    st.plotly_chart(fig, use_container_width=True)


section("Battery health trajectory (5 yr)")
if bat_resp is not None:
    bc = st.columns(4)
    bc[0].metric("Current SOH", f"{bat_resp['current_soh'] * 100:.1f}%")
    bc[1].metric("Annual degradation", f"{bat_resp['annual_degradation_pct']:.2f}%")
    bc[2].metric("SOH in 3y", f"{bat_resp['projected_soh_3y'] * 100:.1f}%")
    bc[3].metric("SOH in 5y", f"{bat_resp['projected_soh_5y'] * 100:.1f}%")

    traj = pd.DataFrame(bat_resp["trajectory"])
    fig = px.area(traj, x="years", y="soh", title="SOH over time")
    fig.update_traces(
        line={"color": PALETTE["accent_2"], "width": 2.5},
        fillcolor="rgba(91,141,239,0.15)",
    )
    fig.update_layout(height=300, yaxis={"tickformat": ".0%"})
    st.plotly_chart(fig, use_container_width=True)

    band = bat_resp["recommended_soc_band"]
    st.info(
        f"Recommended SOC band: **{band[0] * 100:.0f}% - {band[1] * 100:.0f}%**. "
        f"DC fast charging cap: **{bat_resp['recommended_dc_fast_cap_pct'] * 100:.0f}%** of total energy."
    )


section("Annual cost vs ICE")
if cost_resp is not None:
    cc = st.columns(4)
    cc[0].metric("Annual EV cost", f"EUR {cost_resp['annual_cost_eur']:,.0f}")
    cc[1].metric("Annual ICE cost", f"EUR {cost_resp['ice_annual_cost_eur']:,.0f}")
    cc[2].metric(
        "Annual savings vs ICE",
        f"EUR {cost_resp['annual_savings_vs_ice_eur']:,.0f}",
    )
    cc[3].metric("Peak-hour share", f"{cost_resp['peak_hour_share_pct']:.1f}%")
    df = pd.DataFrame(
        {
            "powertrain": ["EV (EEIE)", "ICE"],
            "annual_eur": [cost_resp["annual_cost_eur"], cost_resp["ice_annual_cost_eur"]],
        }
    )
    fig = px.bar(
        df,
        x="annual_eur",
        y="powertrain",
        orientation="h",
        color="powertrain",
        color_discrete_map={"EV (EEIE)": PALETTE["good"], "ICE": PALETTE["bad"]},
    )
    fig.update_layout(
        title="EV vs ICE annual energy cost",
        height=240,
        showlegend=False,
        xaxis={"title": "EUR / year"},
        yaxis={"title": ""},
    )
    st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
section("Explainability (top factors per engine)")
exp_cols = st.columns(3)


def _factor_block(col, title: str, insight: dict | None) -> None:
    with col:
        st.markdown(f"**{title}**")
        if not insight:
            st.write("- (no data)")
            return
        factors = insight.get("top_factors") or []
        if not factors:
            st.write("- (no factors)")
            return
        for f in factors[:5]:
            direction = (f.get("direction") or "").lower()
            arrow = "▲" if direction.startswith("up") or direction == "positive" else "▼"
            color = PALETTE["good"] if arrow == "▲" else PALETTE["bad"]
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"border-bottom:1px solid {PALETTE['border']};padding:4px 0'>"
                f"<span><code>{f['name']}</code></span>"
                f"<span style='color:{color};font-weight:600'>{arrow} {float(f.get('importance', 0)):.3f}</span>"
                "</div>",
                unsafe_allow_html=True,
            )


_factor_block(exp_cols[0], "Range", rng_resp.get("insight") if rng_resp else None)
_factor_block(exp_cols[1], "Optimization", opt_resp.get("insight") if opt_resp else None)
_factor_block(exp_cols[2], "Behavior", beh_resp.get("insight") if beh_resp else None)
