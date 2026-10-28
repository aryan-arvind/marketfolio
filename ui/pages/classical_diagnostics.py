"""
MarketFolio Tab: Classical Econometrics & Baseline Diagnostics.
Consolidates conventional time-series analysis into a dedicated reference desk:
- Joint Stationarity (ADF & KPSS) & Autocorrelation (ACF/PACF, Ljung-Box)
- Seasonality & Calendar Regressions (Day-of-Week OLS, Kruskal-Wallis, STL)
- Benchmark Factor Regression (Cochrane-Orcutt SARIMAX Beta vs HAC OLS)
- Global Macro VAR Transmission (5-Variable Granger Causality & IRF)
- Rolling-Origin Out-of-Sample Backtesting (MAE, RMSE, MASE Scorecard)
"""

from __future__ import annotations

from typing import Any
import pandas as pd
import streamlit as st

from ui.components import render_kpi_card, render_method_tag, render_takeaway_line
from ui.theme import ACCENT_CYAN, ACCENT_TEAL, COLOR_GAIN, COLOR_LOSS, TEXT_MUTED


def render(data: dict[str, Any]) -> None:
    """Renders the Consolidated Classical Diagnostics tab."""
    render_method_tag("Classical Financial Econometrics · Reference & Benchmark Desk")

    stat_suite = data.get("stat_suite", {})
    cal_res = data.get("cal_res")
    stl_res = data.get("stl_res")
    mkt_res = data.get("market_res")
    var_res = data.get("var_res")
    eval_res = data.get("eval_res")
    arma_search = data.get("arma_search")

    # 1. Classical Stationarity & Autocorrelation Section
    st.markdown('<div class="mf-card-title">1. Stationarity & Autocorrelation Structure</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    price_verdict = stat_suite.get("log_price").verdict if "log_price" in stat_suite else "Non-Stationary"
    ret_verdict = stat_suite.get("log_returns").verdict if "log_returns" in stat_suite else "Stationary"

    with c1:
        render_kpi_card("Price Level Verdict", price_verdict, "ADF & KPSS Joint Synthesis")
    with c2:
        render_kpi_card("Log Returns Verdict", ret_verdict, "I(0) Stationary Process", value_color=COLOR_GAIN)
    with c3:
        best_order_str = str(arma_search.best_order) if arma_search else "(0,0,0)"
        render_kpi_card("Best ARMA Order", f"ARMA{best_order_str}", "AICc Grid Search Minimizer")
    with c4:
        delta_aicc = arma_search.best_model.aicc - arma_search.constant_model.aicc if arma_search else 0.0
        render_kpi_card("Δ AICc vs Constant", f"{delta_aicc:+.2f}", "Weak-Form Efficiency Benchmark")

    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

    # 2. Benchmark Regression & Macro VAR
    st.markdown('<div class="mf-card-title">2. Factor Sensitivity & Macro VAR Transmission</div>', unsafe_allow_html=True)
    c_beta, c_var = st.columns(2)

    with c_beta:
        if mkt_res:
            st.markdown(f"**Single-Index SARIMAX Model vs {mkt_res.benchmark_name}**")
            b_df = pd.DataFrame([
                {"Metric": "Market Beta (β)", "Value": f"{mkt_res.arma_beta:.3f} (SE: {mkt_res.arma_beta_se:.3f})"},
                {"Metric": "95% Beta Confidence Interval", "Value": f"[{mkt_res.arma_beta_ci[0]:.3f}, {mkt_res.arma_beta_ci[1]:.3f}]"},
                {"Metric": "Annualized Alpha (α)", "Value": f"{mkt_res.arma_alpha_annualized * 100.0:+.2f}% (p={mkt_res.arma_alpha_pvalue:.3f})"},
                {"Metric": "OLS Durbin-Watson", "Value": f"{mkt_res.ols_dw_stat:.2f} (Residual Autocorrelation Check)"},
                {"Metric": "Aligned Sample Size", "Value": f"{mkt_res.n_aligned_obs} trading days"},
            ])
            st.dataframe(b_df, use_container_width=True, hide_index=True)
        else:
            st.info("Benchmark regression not available.")

    with c_var:
        if var_res:
            st.markdown("**5-Variable Macro VAR System (S&P 500, Brent, USD/INR, VIX, NIFTY)**")
            causal_vars = [g.causing_variable for g in var_res.granger_tests if g.is_causal_05]
            var_df = pd.DataFrame([
                {"Property": "Selected Lag Order (p)", "Value": f"Lag {var_res.selected_lag}"},
                {"Property": "Granger Causal Drivers (p < 0.05)", "Value": ", ".join(causal_vars) if causal_vars else "None at 5% level"},
                {"Property": "Out-of-Sample 1-Step Skill", "Value": var_res.out_of_sample_verdict},
                {"Property": "Cholesky Ordering", "Value": " -> ".join(var_res.cholesky_order)},
            ])
            st.dataframe(var_df, use_container_width=True, hide_index=True)
        else:
            st.info("Macro VAR system not available.")

    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

    # 3. Calendar Regressions & STL Seasonality
    st.markdown('<div class="mf-card-title">3. Calendar Anomalies & STL Decomposition</div>', unsafe_allow_html=True)
    if cal_res and stl_res:
        c_cal1, c_cal2, c_cal3 = st.columns(3)
        with c_cal1:
            render_kpi_card("Day-of-Week OLS F-Stat", f"{cal_res.ols_f_stat:.2f}", f"p-value: {cal_res.ols_f_pvalue:.4f}")
        with c_cal2:
            render_kpi_card("Kruskal-Wallis Stat", f"{cal_res.kruskal_stat:.2f}", f"p-value: {cal_res.kruskal_pvalue:.4f}")
        with c_cal3:
            render_kpi_card("Seasonal Strength (Fs)", f"{stl_res.seasonal_strength_f_s:.3f}", "Monthly STL decomposition")

        render_takeaway_line(
            takeaway=f"Calendar Verdict: {cal_res.summary_verdict}",
            detail="Calendar anomaly dummy regressions evaluate whether historical return distributions significantly diverge across days of the week.",
        )

    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

    # 4. Out-of-Sample Rolling-Origin Backtest Scorecard
    if eval_res:
        st.markdown('<div class="mf-card-title">4. Rolling-Origin Out-of-Sample Forecast Evaluation</div>', unsafe_allow_html=True)
        st.dataframe(eval_res.metrics_table, use_container_width=True, hide_index=True)
        render_takeaway_line(
            takeaway=eval_res.summary_verdict,
            detail=f"Backtested across {eval_res.n_origins_evaluated} expanding-window origins without lookahead bias.",
        )
