"""
MarketFolio Tab 6: Out-of-Sample Forecast Accuracy & Residual Auditing.
Expanding-window rolling-origin protocol without look-ahead bias (assert_no_leakage).
"""

from __future__ import annotations

from typing import Any
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.components import render_kpi_card, render_method_tag, render_takeaway_line
from ui.readings import get_backtest_reading
from ui.theme import ACCENT_TEAL, COLOR_GAIN, COLOR_LOSS, COLOR_WARNING, TEXT_MUTED


def render(data: dict[str, Any]) -> None:
    """Renders the Forecast Accuracy tab."""
    ev = data["eval_res"]
    arma_search = data["arma_search"]
    best_model = arma_search.best_model

    render_method_tag(f"Chronological Expanding-Window Rolling Origin Protocol ({ev.n_origins_evaluated} Origins, {ev.horizon}D Horizon)")

    m_metrics = ev.model_metrics
    n_metrics = ev.naive_metrics
    rm_metrics = ev.rolling_mean_metrics
    imprv = m_metrics.mase_improvement_pct
    beats_naive = imprv > 0

    # Top KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            title="Evaluation Origins",
            value=f"{ev.n_origins_evaluated}",
            subtext=f"Initial burn-in: {ev.initial_window} days",
        )
    with c2:
        render_kpi_card(
            title="Forecast Horizon",
            value=f"{ev.horizon} Sessions",
            subtext=f"Step size: {ev.step_size} days",
        )
    with c3:
        imprv_color = COLOR_GAIN if beats_naive else COLOR_LOSS
        render_kpi_card(
            title="MASE vs Naive Baseline",
            value=f"{imprv:+.1f}%",
            subtext="Scale-free error improvement",
            value_color=imprv_color,
        )
    with c4:
        status_label = "Beats Baseline" if beats_naive else "Baseline Comparable"
        status_color = COLOR_GAIN if beats_naive else TEXT_MUTED
        render_kpi_card(
            title="Empirical Verdict",
            value=status_label,
            subtext="Out-of-sample skill rating",
            value_color=status_color,
        )

    # Main Metrics Table
    st.markdown('<div style="font-size:0.88rem; font-weight:600; margin-top:0.5rem; margin-bottom:0.25rem;">Out-of-Sample Performance Comparison</div>', unsafe_allow_html=True)
    st.dataframe(ev.metrics_table, use_container_width=True, hide_index=True)

    # Visual Error Comparison Chart
    models = [m_metrics.model_name, n_metrics.model_name, rm_metrics.model_name]
    maes = [m_metrics.mae, n_metrics.mae, rm_metrics.mae]
    rmses = [m_metrics.rmse, n_metrics.rmse, rm_metrics.rmse]
    mases = [m_metrics.mase, n_metrics.mase, rm_metrics.mase]

    fig_err = go.Figure()
    fig_err.add_trace(
        go.Bar(
            x=models,
            y=maes,
            name="MAE",
            marker_color=ACCENT_TEAL,
            hovertemplate="%{x}<br>MAE: %{y:.4f}<extra></extra>",
        )
    )
    fig_err.add_trace(
        go.Bar(
            x=models,
            y=rmses,
            name="RMSE",
            marker_color="#38bdf8",
            hovertemplate="%{x}<br>RMSE: %{y:.4f}<extra></extra>",
        )
    )
    fig_err.add_trace(
        go.Bar(
            x=models,
            y=mases,
            name="MASE",
            marker_color=COLOR_WARNING,
            hovertemplate="%{x}<br>MASE: %{y:.4f}<extra></extra>",
        )
    )
    fig_err.update_layout(
        barmode="group",
        height=280,
        margin=dict(l=35, r=15, t=15, b=25),
        yaxis=dict(title="Error Value"),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    st.caption("Forecast Error Metric Comparison (Lower is Superior)")
    st.plotly_chart(fig_err, use_container_width=True, config={"displayModeBar": False})

    render_takeaway_line(f"<b>Desk Takeaway</b>: {get_backtest_reading(ev)}")

    # Details Expander: Residual diagnostics & No-leakage validation
    with st.expander("Technical Model Residual Diagnostics & Leakage Audit"):
        c_r1, c_r2 = st.columns(2)
        with c_r1:
            st.markdown("##### Residual Gaussian White Noise Checks")
            st.markdown(
                f"""
                <div style="font-size:0.82rem; color:{TEXT_MUTED}; line-height:1.7;">
                    <b>Model Specification</b>: ARIMA{best_model.order}<br>
                    <b>Jarque-Bera Test Stat</b>: {best_model.jb_stat:.2f} (p-value: {best_model.jb_pvalue:.4f})<br>
                    <b>Residual Sample Skewness</b>: {best_model.jb_skew:+.3f}<br>
                    <b>Residual Kurtosis</b>: {best_model.jb_kurtosis:.3f}<br>
                    <b>Data Leakage Audit</b>: Verified via <code>assert_no_leakage</code> (zero train/test index overlap across all origins).
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_r2:
            st.markdown("##### Ljung-Box Residual Autocorrelation Table")
            if not best_model.lb_df.empty:
                st.dataframe(best_model.lb_df, use_container_width=True, hide_index=False)
