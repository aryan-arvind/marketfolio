"""
MarketFolio Tab 2: Price & Trend Analysis.
STL Decomposition, Trend vs Seasonal strength metrics, and Calendar Tendencies.
"""

from __future__ import annotations

from typing import Any
import pandas as pd
import streamlit as st

from ui.charts import (
    build_calendar_bars,
    build_lag1_scatter_chart,
    build_stl_decomposition_chart,
)
from ui.components import render_kpi_card, render_method_tag, render_takeaway_line
from ui.readings import get_calendar_reading, get_trend_reading
from ui.theme import ACCENT_TEAL, COLOR_GAIN, COLOR_LOSS, TEXT_MUTED


def render(data: dict[str, Any]) -> None:
    """Renders the Price & Trend tab."""
    render_method_tag("Seasonal-Trend Decomposition (STL, Period=21) & Calendar Hypothesis Testing")

    stl_res = data["stl_res"]
    cal_res = data["cal_res"]
    clean_res = data["clean_res"]

    # Top Metric Bar: Trend Strength (F_T), Seasonal Strength (F_S), Calendar Verdict
    c1, c2, c3 = st.columns([1, 1, 1.5])
    with c1:
        f_t = stl_res.trend_strength_f_t
        render_kpi_card(
            title="Trend Strength (F_T)",
            value=f"{f_t:.3f}",
            subtext="Variance explained by trend (0 to 1)",
            value_color=ACCENT_TEAL if f_t >= 0.5 else TEXT_MUTED,
        )
    with c2:
        f_s = stl_res.seasonal_strength_f_s
        render_kpi_card(
            title="Seasonal Strength (F_S)",
            value=f"{f_s:.3f}",
            subtext="21-session monthly cyclicality",
            value_color=ACCENT_TEAL if f_s >= 0.3 else TEXT_MUTED,
        )
    with c3:
        sig = cal_res.is_day_of_week_significant
        render_kpi_card(
            title="Calendar Pattern Significance",
            value="Significant Anomaly" if sig else "Statistically Uniform",
            subtext=f"Joint OLS F p={cal_res.ols_f_pvalue:.4f}",
            value_color=COLOR_GAIN if sig else TEXT_MUTED,
        )

    # STL Component Breakdown Chart
    st.markdown('<div style="font-size:0.88rem; font-weight:600; margin-top:0.5rem; margin-bottom:0.25rem;">STL Component Decomposition</div>', unsafe_allow_html=True)
    fig_stl = build_stl_decomposition_chart(stl_res, data["display_name"])
    st.plotly_chart(fig_stl, use_container_width=True, config={"displayModeBar": False})

    # Calendar Return Bars: Weekday and Monthly side-by-side
    st.markdown('<div style="font-size:0.88rem; font-weight:600; margin-top:0.75rem; margin-bottom:0.25rem;">Calendar Return Distributions</div>', unsafe_allow_html=True)
    col_dow, col_month = st.columns(2)
    fig_dow, fig_month = build_calendar_bars(cal_res)

    with col_dow:
        st.caption("Day-of-Week Mean Daily Returns (% ± 95% CI)")
        st.plotly_chart(fig_dow, use_container_width=True, config={"displayModeBar": False})

    with col_month:
        st.caption("Month-of-Year Mean Daily Returns (% ± 95% CI)")
        st.plotly_chart(fig_month, use_container_width=True, config={"displayModeBar": False})

    render_takeaway_line(
        f"<b>Desk Takeaway</b>: {get_trend_reading(stl_res)} {get_calendar_reading(cal_res)}"
    )

    # Details Expander: Lag-1 scatter with trimmed axes, Anomalies, and Statistical Tables
    with st.expander("Statistical Details & Lag-1 Serial Structure"):
        st.markdown("##### Lag-1 Return Persistence")
        fig_lag, lag_note = build_lag1_scatter_chart(clean_res.log_return_series)
        st.caption(lag_note)
        st.plotly_chart(fig_lag, use_container_width=True, config={"displayModeBar": False})

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("##### Weekday Sample Statistics")
            dow_disp = cal_res.dow_stats.copy()
            dow_disp["mean_pct"] = (dow_disp["mean"] * 100).map("{:+.3f}%".format)
            dow_disp["std_pct"] = (dow_disp["std"] * 100).map("{:.3f}%".format)
            st.dataframe(
                dow_disp[["day_name", "count", "mean_pct", "std_pct"]],
                use_container_width=True,
                hide_index=True,
            )
        with col_t2:
            st.markdown("##### Calendar Hypothesis Tests")
            st.markdown(
                f"""
                <div style="font-size:0.82rem; color:{TEXT_MUTED}; line-height:1.7;">
                    <b>Joint OLS F-Statistic</b>: {cal_res.ols_f_stat:.3f} (p-value: {cal_res.ols_f_pvalue:.4f})<br>
                    <b>Kruskal-Wallis Statistic</b>: {cal_res.kruskal_stat:.3f} (p-value: {cal_res.kruskal_pvalue:.4f})<br>
                    <b>Decision Threshold</b>: alpha = 0.05<br>
                    <b>Econometric Conclusion</b>: {cal_res.summary_verdict}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if not clean_res.flagged_anomalies.empty:
            st.markdown("---")
            st.markdown(
                f'<span class="mf-badge mf-badge-stale">Data Quality Alert &bull; {len(clean_res.flagged_anomalies)} Outlier Sessions (|r| > 15%)</span>',
                unsafe_allow_html=True,
            )
            st.dataframe(clean_res.flagged_anomalies, use_container_width=True, hide_index=True)
