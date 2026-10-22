"""
MarketFolio Tab 3: Returns & Predictability.
Return distribution metrics, EMH weak-form martingale tests, and AICc model grid search.
"""

from __future__ import annotations

from typing import Any
import pandas as pd
import streamlit as st

from ui.charts import (
    build_acf_pacf_tidy_chart,
    build_cumulative_returns_chart,
    build_returns_distribution_chart,
)
from ui.components import render_kpi_card, render_method_tag, render_takeaway_line
from ui.readings import get_predictability_reading
from ui.state import calculate_derived_stats
from ui.theme import ACCENT_TEAL, COLOR_GAIN, COLOR_LOSS, COLOR_WARNING, TEXT_MUTED


def render(data: dict[str, Any]) -> None:
    """Renders the Returns & Predictability tab."""
    render_method_tag("Conditional Mean ARMA Selection (AICc Criterion) & Weak-Form Efficiency")

    clean_res = data["clean_res"]
    returns = clean_res.log_return_series
    prices = clean_res.price_series
    arma_search = data["arma_search"]
    stat_suite = data["stat_suite"]
    lb_returns = data["lb_returns"]
    lb_sq_returns = data["lb_sq_returns"]

    stats_dict = calculate_derived_stats(prices, returns)

    # Top KPI Row: Skewness, Kurtosis, Extreme Sessions, Predictability Finding
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            title="Sample Skewness",
            value=f"{stats_dict['skewness']:+.2f}",
            subtext="Asymmetry (0 = symmetric)",
            value_color=COLOR_LOSS if stats_dict['skewness'] < -0.5 else TEXT_MUTED,
        )
    with c2:
        render_kpi_card(
            title="Excess Kurtosis",
            value=f"{stats_dict['kurtosis']:+.2f}",
            subtext="Fat tails (Fisher excess > 0)",
            value_color=COLOR_WARNING if stats_dict['kurtosis'] > 1.0 else TEXT_MUTED,
        )
    with c3:
        render_kpi_card(
            title="Extreme Sessions (|r| > 3s)",
            value=f"{stats_dict['n_extreme_sessions']:,}",
            subtext=f"Out of {len(returns):,} total days",
        )
    with c4:
        delta_aicc = arma_search.best_model.aicc - arma_search.constant_model.aicc
        is_pred = arma_search.is_predictable
        pred_label = "Linear Signal Detected" if is_pred else "Martingale / Random Walk"
        pred_color = COLOR_GAIN if is_pred else TEXT_MUTED
        render_kpi_card(
            title=f"Best Model ARIMA{arma_search.best_order}",
            value=pred_label,
            subtext=f"Delta AICc vs drift: {delta_aicc:+.2f}",
            value_color=pred_color,
        )

    # 2 Main Visualizations: Left distribution histogram, Right cumulative returns
    col_hist, col_cum = st.columns(2)
    with col_hist:
        st.markdown('<div style="font-size:0.88rem; font-weight:600; margin-bottom:0.25rem;">Daily Return Distribution & Normal Fit</div>', unsafe_allow_html=True)
        fig_dist = build_returns_distribution_chart(returns)
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

    with col_cum:
        st.markdown('<div style="font-size:0.88rem; font-weight:600; margin-bottom:0.25rem;">Cumulative Sample Trajectory</div>', unsafe_allow_html=True)
        fig_cum = build_cumulative_returns_chart(stats_dict["cum_returns"], data["display_name"])
        st.plotly_chart(fig_cum, use_container_width=True, config={"displayModeBar": False})

    render_takeaway_line(
        f"<b>Desk Takeaway</b>: {get_predictability_reading(arma_search)} "
        f"Return distribution displays {stats_dict['kurtosis']:+.1f} excess kurtosis with "
        f"{stats_dict['n_extreme_sessions']} outlier session(s)."
    )

    # Details Expander: ACF/PACF, Ljung-Box tables, Unit Root synthesis, Grid Search Table
    with st.expander("Diagnostic Tests: Autocorrelation, Stationarity & ARMA Grid Search"):
        st.markdown("##### Autocorrelation Structure (ACF & PACF)")
        fig_acf = build_acf_pacf_tidy_chart(returns, nlags=20)
        st.plotly_chart(fig_acf, use_container_width=True, config={"displayModeBar": False})

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("##### Ljung-Box Test (Directional vs ARCH Clustering)")
            lb_combined = pd.DataFrame({
                "Lag": [10, 20],
                "Returns Stat": [lb_returns.loc[10, "lb_stat"], lb_returns.loc[20, "lb_stat"]],
                "Returns p-val": [lb_returns.loc[10, "lb_pvalue"], lb_returns.loc[20, "lb_pvalue"]],
                "Squared Returns Stat": [lb_sq_returns.loc[10, "lb_stat"], lb_sq_returns.loc[20, "lb_stat"]],
                "Squared Returns p-val": [lb_sq_returns.loc[10, "lb_pvalue"], lb_sq_returns.loc[20, "lb_pvalue"]],
            })
            st.dataframe(
                lb_combined.style.format({
                    "Returns Stat": "{:.2f}",
                    "Returns p-val": "{:.4f}",
                    "Squared Returns Stat": "{:.2f}",
                    "Squared Returns p-val": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with col_t2:
            st.markdown("##### Unit Root Testing (ADF / KPSS)")
            ur_p = stat_suite["log_price"]
            ur_r = stat_suite["log_returns"]
            ur_df = pd.DataFrame([
                {
                    "Series": "Log Price (P_t)",
                    "ADF Stat (p-val)": f"{ur_p.adf_stat:.2f} ({ur_p.adf_pvalue:.3f})",
                    "KPSS Stat (p-val)": f"{ur_p.kpss_stat:.2f} ({ur_p.kpss_pvalue:.3f})",
                    "Decision": ur_p.verdict,
                },
                {
                    "Series": "Log Return (r_t)",
                    "ADF Stat (p-val)": f"{ur_r.adf_stat:.2f} ({ur_r.adf_pvalue:.3f})",
                    "KPSS Stat (p-val)": f"{ur_r.kpss_stat:.2f} ({ur_r.kpss_pvalue:.3f})",
                    "Decision": ur_r.verdict,
                },
            ])
            st.dataframe(ur_df, use_container_width=True, hide_index=True)

        st.markdown("##### Full ARMA(p, 0, q) Model Grid Search")
        st.dataframe(arma_search.all_models_df, use_container_width=True, hide_index=True)
