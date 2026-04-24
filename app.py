"""
Competitive Readout Simulator — Streamlit app.

UI for clinical trial readout timeline prediction via Monte Carlo simulation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from simulator import (
    TrialAssumptions,
    run_monte_carlo,
    summarize_results,
    required_events,
    months_to_date,
)

st.set_page_config(
    page_title="Competitive Readout Simulator",
    page_icon="📊",
    layout="wide",
)

st.title("Competitive Readout Simulator")
st.caption(
    "Monte Carlo simulation of event-driven clinical trial timelines. "
    "Adjust assumptions in the sidebar to see how predicted readout dates shift."
)

# ---------- Sidebar: assumptions ----------
with st.sidebar:
    st.header("Trial assumptions")

    st.subheader("Enrollment")
    enrollment_start = st.date_input(
        "Trial start date",
        value=datetime(2025, 1, 1),
        help="Date of first patient in.",
    )
    total_sample_size = st.number_input(
        "Total sample size (N)",
        min_value=50, max_value=5000, value=600, step=50,
    )
    monthly_accrual_rate = st.number_input(
        "Monthly accrual rate (steady state)",
        min_value=1.0, max_value=200.0, value=25.0, step=1.0,
        help="Patients enrolled per month once all sites are active.",
    )
    accrual_rampup_months = st.number_input(
        "Accrual ramp-up period (months)",
        min_value=0.0, max_value=24.0, value=6.0, step=0.5,
        help="Time to reach steady-state accrual. Models site activation.",
    )
    randomization_ratio = st.selectbox(
        "Randomization ratio (treatment:control)",
        options=[1.0, 2.0, 3.0],
        index=0,
        format_func=lambda x: f"{int(x)}:1",
    )

    st.subheader("Survival assumptions")
    control_median_months = st.number_input(
        "Control arm median survival (months)",
        min_value=1.0, max_value=120.0, value=12.0, step=0.5,
        help="Median PFS or OS in the control arm. Use real-world data or published benchmarks.",
    )
    target_hazard_ratio = st.slider(
        "Target hazard ratio",
        min_value=0.30, max_value=0.95, value=0.70, step=0.05,
        help="Assumed treatment effect. HR < 1 favors treatment. Smaller = bigger effect but harder to be right about.",
    )
    dropout_rate_annual = st.slider(
        "Annual dropout rate",
        min_value=0.0, max_value=0.30, value=0.05, step=0.01,
        format="%.2f",
    )

    st.subheader("Analysis plan")
    alpha_two_sided = st.selectbox(
        "Two-sided alpha",
        options=[0.01, 0.025, 0.05],
        index=2,
    )
    power = st.selectbox(
        "Power",
        options=[0.80, 0.85, 0.90],
        index=0,
    )
    ia1_frac = st.slider("IA1 information fraction", 0.3, 0.9, 0.5, 0.05)
    ia2_frac = st.slider("IA2 information fraction", 0.5, 0.95, 0.8, 0.05)
    information_fractions = sorted(set([ia1_frac, ia2_frac, 1.0]))

    st.subheader("Simulation")
    n_simulations = st.select_slider(
        "Number of simulations",
        options=[100, 500, 1000, 2000, 5000],
        value=1000,
        help="More simulations = tighter estimates but slower.",
    )

# ---------- Build assumptions object ----------
assumptions = TrialAssumptions(
    enrollment_start=datetime.combine(enrollment_start, datetime.min.time()),
    total_sample_size=total_sample_size,
    monthly_accrual_rate=monthly_accrual_rate,
    accrual_rampup_months=accrual_rampup_months,
    randomization_ratio=randomization_ratio,
    control_median_months=control_median_months,
    target_hazard_ratio=target_hazard_ratio,
    dropout_rate_annual=dropout_rate_annual,
    alpha_two_sided=alpha_two_sided,
    power=power,
    information_fractions=information_fractions,
)

# ---------- Top-line metrics ----------
col1, col2, col3 = st.columns(3)
total_events = required_events(target_hazard_ratio, alpha_two_sided, power, randomization_ratio)
col1.metric("Events required (final analysis)", f"{total_events}")
col2.metric("Sample size", f"{total_sample_size}")
col3.metric("Simulations", f"{n_simulations:,}")

# ---------- Run simulation ----------
with st.spinner("Running Monte Carlo simulation..."):
    df = run_monte_carlo(assumptions, n_simulations=n_simulations)
    summary = summarize_results(df, assumptions)

# ---------- Results table ----------
st.subheader("Predicted readout dates")

display_summary = summary.copy()
for col in ["P10 date (optimistic)", "P50 date (median)", "P90 date (conservative)"]:
    display_summary[col] = display_summary[col].apply(
        lambda d: d.strftime("%b %Y") if d is not None else "—"
    )
st.dataframe(display_summary, use_container_width=True, hide_index=True)

st.caption(
    "P10 = 10% of simulations hit this milestone earlier than this date. "
    "P50 = median prediction. P90 = conservative estimate (only 10% of sims took longer)."
)

# ---------- Distribution plot ----------
st.subheader("Distribution of readout dates")

fig = go.Figure()
colors = ["#636EFA", "#EF553B", "#00CC96"]
for i, frac in enumerate(assumptions.information_fractions):
    if frac not in df.columns:
        continue
    # Convert months to calendar dates for plotting
    dates = df[frac].dropna().apply(
        lambda m: months_to_date(m, assumptions.enrollment_start)
    )
    fig.add_trace(go.Histogram(
        x=dates,
        name=f"{int(frac*100)}% IF",
        opacity=0.65,
        marker_color=colors[i % len(colors)],
        nbinsx=40,
    ))

fig.update_layout(
    barmode="overlay",
    xaxis_title="Predicted readout date",
    yaxis_title="Number of simulations",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Sensitivity analysis ----------
with st.expander("Sensitivity analysis: how does HR affect readout timing?"):
    st.caption(
        "Holding all else constant, see how the P50 final readout date shifts "
        "as the true hazard ratio changes. Useful for stress-testing assumptions."
    )
    hr_range = np.arange(0.55, 0.91, 0.05)
    sensitivity_rows = []
    for hr in hr_range:
        temp_assumptions = TrialAssumptions(
            **{**assumptions.__dict__, "target_hazard_ratio": round(hr, 2)}
        )
        temp_df = run_monte_carlo(temp_assumptions, n_simulations=200)
        final_times = temp_df[1.0].dropna()
        if len(final_times) > 0:
            p50_months = final_times.quantile(0.50)
            sensitivity_rows.append({
                "Hazard ratio": round(hr, 2),
                "P50 final readout": months_to_date(
                    p50_months, assumptions.enrollment_start
                ).strftime("%b %Y"),
                "P50 months from start": round(p50_months, 1),
                "Events needed": required_events(round(hr, 2), alpha_two_sided, power, randomization_ratio),
            })
    st.dataframe(pd.DataFrame(sensitivity_rows), use_container_width=True, hide_index=True)

# ---------- Raw data download ----------
with st.expander("Download raw simulation output"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv,
        file_name="simulation_results.csv",
        mime="text/csv",
    )
    st.dataframe(df.head(20), use_container_width=True)

st.divider()
st.caption(
    "Built for competitive intelligence and strategy teams. "
    "This tool simulates statistical milestones assuming proportional hazards and exponential survival; "
    "results are estimates, not guarantees."
)
