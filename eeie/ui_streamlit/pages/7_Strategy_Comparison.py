"""Naive vs Rule-based vs EEIE charging strategy comparison.

Calls the cached `/evaluate/ablation` endpoint, then renders the three
strategies side-by-side. The headline metrics (cost reduction, peak reduction,
wear reduction) are computed server-side against the naive baseline.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from eeie.ui_streamlit.lib.api import run_ablation
from eeie.ui_streamlit.lib.theme import PALETTE, hero, install_theme, section

install_theme(page_title="EEIE | Strategy Comparison", icon=":bar_chart:")

hero(
    "Strategy Comparison",
    "Three charging strategies, one ablation. Naive (charge now), rule-based (off-peak only), "
    "and EEIE (MILP-optimized) - run across N synthetic scenarios.",
)


controls = st.columns([1, 1, 1, 1, 2])
n_scenarios = controls[0].slider("Scenarios", 5, 60, 20, step=5)
horizon = controls[1].slider("Horizon (h)", 12, 48, 24, step=6)
optimizer = controls[2].radio("EEIE optimizer", ["milp", "rl"], horizontal=True)
seed = controls[3].number_input("Seed", min_value=0, max_value=10_000, value=0, step=1)
refresh = controls[4].toggle(
    "Force recompute", value=False, help="Bypass the API's in-memory cache."
)

run_btn = st.button("Run ablation", type="primary", use_container_width=True)
if not run_btn:
    st.info(
        "The ablation runs every strategy on every scenario. Results are cached server-side "
        "by `(scenarios, horizon, optimizer, seed)`, so subsequent identical runs are instant."
    )
    st.stop()


with st.spinner("Running ablation..."):
    try:
        report = run_ablation(
            n_scenarios=n_scenarios,
            horizon_hours=horizon,
            optimizer=optimizer,
            seed=int(seed),
            refresh=bool(refresh),
        )
    except Exception as exc:
        st.error(f"Ablation failed: {exc}")
        st.stop()


by = report["by_strategy"]
naive = by["naive"]
rule = by["rule_based"]
eeie = by["eeie"]


section("Headline results")
h = st.columns(3)
h[0].metric(
    "Cost reduction (EEIE vs naive)",
    f"{report['cost_reduction_pct_eeie_vs_naive']:.1f}%",
    delta="lower is better",
    delta_color="off",
)
h[1].metric(
    "Peak-hour energy reduction",
    f"{report['peak_reduction_pct_eeie_vs_naive']:.1f}%",
    delta="lower is better",
    delta_color="off",
)
h[2].metric(
    "Battery wear reduction",
    f"{report['wear_reduction_pct_eeie_vs_naive']:.1f}%",
    delta="lower is better",
    delta_color="off",
)

st.markdown("")

section("Per-strategy means")
table = pd.DataFrame(
    [
        {
            "strategy": "naive",
            "Cost (EUR)": naive["mean_cost_eur"],
            "Wear (proxy)": naive["mean_wear"],
            "Peak energy (kWh)": naive["mean_peak_kwh"],
            "Feasible %": naive["feasible_pct"],
        },
        {
            "strategy": "rule_based",
            "Cost (EUR)": rule["mean_cost_eur"],
            "Wear (proxy)": rule["mean_wear"],
            "Peak energy (kWh)": rule["mean_peak_kwh"],
            "Feasible %": rule["feasible_pct"],
        },
        {
            "strategy": "eeie",
            "Cost (EUR)": eeie["mean_cost_eur"],
            "Wear (proxy)": eeie["mean_wear"],
            "Peak energy (kWh)": eeie["mean_peak_kwh"],
            "Feasible %": eeie["feasible_pct"],
        },
    ]
).set_index("strategy")
st.dataframe(table.round(2), use_container_width=True)


color_map = {"naive": PALETTE["bad"], "rule_based": PALETTE["warn"], "eeie": PALETTE["accent"]}


def _bar(metric_key: str, label: str) -> go.Figure:
    strategies = ["naive", "rule_based", "eeie"]
    values = [by[s][metric_key] for s in strategies]
    fig = go.Figure(
        go.Bar(
            x=strategies,
            y=values,
            marker_color=[color_map[s] for s in strategies],
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=label,
        height=320,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        showlegend=False,
        yaxis={"title": label},
    )
    return fig


section("Side-by-side")
g = st.columns(3)
g[0].plotly_chart(_bar("mean_cost_eur", "Mean cost (EUR)"), use_container_width=True)
g[1].plotly_chart(_bar("mean_wear", "Mean wear (penalty units)"), use_container_width=True)
g[2].plotly_chart(_bar("mean_peak_kwh", "Mean peak-hour energy (kWh)"), use_container_width=True)


st.markdown("")
section("Read-out")
st.markdown(
    f"""
- **EEIE cut electricity cost by {report["cost_reduction_pct_eeie_vs_naive"]:.1f}%** vs the naive
  strategy across {report["n_scenarios"]} scenarios.
- It moved **{report["peak_reduction_pct_eeie_vs_naive"]:.1f}%** of energy out of peak hours, easing
  grid stress and avoiding peak tariff charges.
- Battery wear (a proxy for accelerated SOH degradation) dropped by
  **{report["wear_reduction_pct_eeie_vs_naive"]:.1f}%**.
- The rule-based baseline is included to show that *"charge off-peak"* is a weak heuristic - it
  helps with cost but does nothing intelligent about battery wear or feasibility under tight deadlines.
"""
)
