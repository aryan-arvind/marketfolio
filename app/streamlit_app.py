"""
RiskLens: Interactive ATSA Time Series Risk and Macro-Linkage Dashboard.
Built with Streamlit and cached quantitative analysis routines.
Run via: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Ensure src is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from risklens.arma_models import grid_search_arma, plot_arma_residual_diagnostics
from risklens.cleaning import align_macro_series, clean_primary_asset
from risklens.config import DEFAULT_BENCHMARK_SYMBOL, DEFAULT_FORECAST_HORIZON
from risklens.data_sources import (
    fetch_all_macro_series,
    fetch_mutual_fund_nav,
    fetch_yfinance_ticker,
    resolve_mutual_fund,
)
from risklens.diagnostics import (
    compute_ljung_box,
    plot_acf_pacf_diagnostics,
    run_stationarity_suite,
)
from risklens.evaluation import run_rolling_origin_evaluation
from risklens.narrator import generate_narrative
from risklens.regression_arma import fit_regression_arma, plot_market_regression
from risklens.seasonality import (
    analyze_calendar_effects,
    plot_seasonality_and_subseries,
    plot_stl_components,
    run_stl_decomposition,
)
from risklens.var_analysis import plot_var_diagnostics_and_irf, run_var_analysis
from risklens.volatility import forecast_volatility, plot_volatility_forecast

st.set_page_config(
    page_title="RiskLens | ATSA Quantitative Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .freshness-card {
        background-color: #f8fafc;
        border-left: 5px solid #2563eb;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }
    .metric-badge {
        font-weight: 600;
        color: #1e293b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=21600, show_spinner=False)
def execute_analysis(
    asset_mode: str,
    asset_input: str,
    benchmark_symbol: str,
    horizon: int,
    force_refresh: bool,
) -> dict[str, Any]:
    """
    Cached analysis runner with 6-hour TTL (21600 seconds).
    Pulls live data, cleans, and runs all statistical modeling.
    """
    warnings_list: list[str] = []

    if asset_mode == "Indian Mutual Fund":
        scheme_code, scheme_name = resolve_mutual_fund(asset_input)
        raw_price, meta = fetch_mutual_fund_nav(scheme_code, scheme_name=scheme_name, force_refresh=force_refresh)
        display_name = scheme_name
        symbol_code = str(scheme_code)
        if "dividend" in scheme_name.lower() or "idcw" in scheme_name.lower():
            warnings_list.append(
                f"Selected fund '{scheme_name}' appears to be a Dividend/IDCW plan. "
                "Dividend payouts distort NAV returns. Growth plans are advised."
            )
    else:
        raw_price, meta = fetch_yfinance_ticker(asset_input, force_refresh=force_refresh, is_primary=True)
        display_name = asset_input
        symbol_code = asset_input

    if meta.get("is_stale"):
        warnings_list.append(f"Network interruption encountered: Stale cache as of {meta.get('last_date')} used.")

    # Cleaning
    clean_res = clean_primary_asset(raw_price, asset_name=display_name)
    prices = clean_res.price_series
    returns = clean_res.log_return_series

    # Diagnostics
    stat_suite = run_stationarity_suite(prices, returns, display_name)
    lb_returns = compute_ljung_box(returns, lags=[10, 20])
    lb_sq_returns = compute_ljung_box(returns ** 2, lags=[10, 20])

    # Seasonality
    cal_res = analyze_calendar_effects(returns, display_name)
    stl_res = run_stl_decomposition(prices, display_name, period=21)

    # ARMA
    arma_search = grid_search_arma(returns, display_name, d=0, max_p=3, max_q=3)

    # Volatility
    vol_res = forecast_volatility(returns, display_name, horizon=horizon)

    # Benchmark Regression
    bench_clean = None
    market_res = None
    try:
        bench_raw, _ = fetch_yfinance_ticker(benchmark_symbol, force_refresh=force_refresh, is_primary=False)
        bench_clean = clean_primary_asset(bench_raw, asset_name=benchmark_symbol, min_observations=250)
        market_res = fit_regression_arma(returns, bench_clean.log_return_series, asset_name=display_name, benchmark_name=benchmark_symbol)
    except Exception as exc:
        warnings_list.append(f"Market regression against {benchmark_symbol} skipped: {exc}")

    # VAR
    macro_dict, _ = fetch_all_macro_series(force_refresh=force_refresh)
    aligned_macro, macro_valid, macro_reason = align_macro_series(macro_dict)
    var_res = None
    if macro_valid:
        v_res, v_ok, _ = run_var_analysis(aligned_macro)
        if v_ok:
            var_res = v_res

    # Evaluation
    eval_res = run_rolling_origin_evaluation(
        series=vol_res.log_vol_series,
        target_name=f"{display_name} Volatility",
        model_order=vol_res.best_order,
        horizon=horizon,
    )

    # Narrative
    assembled_results = {
        "asset_metadata": {
            "name": display_name,
            "symbol_or_code": symbol_code,
            "source": meta.get("source", "Public Data"),
            "last_available_date": meta.get("last_date", str(prices.index.max().date())),
            "first_available_date": meta.get("first_date", str(prices.index.min().date())),
            "fetch_timestamp": meta.get("fetch_timestamp", ""),
            "n_observations": len(returns),
            "is_stale": meta.get("is_stale", False),
        },
        "warnings": warnings_list,
        "stationarity": {
            "log_price_verdict": stat_suite["log_price"].verdict,
            "log_returns_verdict": stat_suite["log_returns"].verdict,
            "returns_adf_pvalue": stat_suite["log_returns"].adf_pvalue,
            "returns_kpss_pvalue": stat_suite["log_returns"].kpss_pvalue,
        },
        "seasonality": {
            "day_of_week_significant": cal_res.is_day_of_week_significant,
            "ols_f_pvalue": cal_res.ols_f_pvalue,
            "trend_strength_f_t": stl_res.trend_strength_f_t,
            "seasonal_strength_f_s": stl_res.seasonal_strength_f_s,
        },
        "arma_predictability": {
            "best_order": f"({arma_search.best_model.order[0]},{arma_search.best_model.order[1]},{arma_search.best_model.order[2]})",
            "delta_aicc_vs_constant": arma_search.best_model.aicc - arma_search.constant_model.aicc,
            "is_predictable": arma_search.is_predictable,
        },
        "volatility_forecast": {
            "last_observed_vol_pct": vol_res.last_observed_vol_pct,
            "point_forecast_pct": vol_res.point_forecast_pct,
            "expected_risk_range_lower_pct": vol_res.expected_risk_range_pct[0],
            "expected_risk_range_upper_pct": vol_res.expected_risk_range_pct[1],
            "bootstrap_fallback_used": vol_res.boot_failed_or_fallback,
        },
        "market_model": (
            {
                "benchmark_name": market_res.benchmark_name,
                "beta": market_res.arma_beta,
                "beta_se": market_res.arma_beta_se,
                "alpha_annualized_pct": market_res.arma_alpha_annualized * 100.0,
                "alpha_pvalue": market_res.arma_alpha_pvalue,
                "r_squared": market_res.ols_r_squared,
            }
            if market_res
            else {}
        ),
        "var_macro": (
            {
                "available": True,
                "selected_lag": var_res.selected_lag,
                "granger_causal_factors": [g.causing_variable for g in var_res.granger_tests if g.is_causal_05],
                "var_beats_baseline": var_res.var_beats_baseline,
            }
            if var_res
            else {"available": False}
        ),
    }

    narrative = generate_narrative(assembled_results, use_llm=False)

    return {
        "display_name": display_name,
        "symbol_code": symbol_code,
        "meta": meta,
        "warnings": warnings_list,
        "clean_res": clean_res,
        "stat_suite": stat_suite,
        "lb_returns": lb_returns,
        "lb_sq_returns": lb_sq_returns,
        "cal_res": cal_res,
        "stl_res": stl_res,
        "arma_search": arma_search,
        "vol_res": vol_res,
        "market_res": market_res,
        "bench_returns": bench_clean.log_return_series if bench_clean else None,
        "var_res": var_res,
        "eval_res": eval_res,
        "narrative": narrative,
        "assembled_results": assembled_results,
    }


# ==============================
# UI Sidebar Configuration
# ==============================
st.sidebar.title("🛡️ RiskLens Settings")
asset_mode = st.sidebar.radio("Asset Selection Mode", ["Indian Mutual Fund", "Stock / Index (Yahoo Finance)"])

if asset_mode == "Indian Mutual Fund":
    asset_input = st.sidebar.text_input(
        "AMFI Scheme Query",
        value="Nifty 50 Index Fund Direct Growth",
        help="Search query to resolve fund from official AMFI database (e.g. 'Parag Parikh Flexi Cap Direct Growth')",
    )
else:
    asset_input = st.sidebar.text_input(
        "Yahoo Finance Ticker",
        value="RELIANCE.NS",
        help="NSE equity or index symbol with .NS suffix (e.g. 'RELIANCE.NS', 'HDFCBANK.NS', '^NSEI')",
    )

benchmark_symbol = st.sidebar.text_input(
    "Market Benchmark Symbol",
    value=DEFAULT_BENCHMARK_SYMBOL,
    help="Default benchmark is NIFTY 50 (^NSEI)",
)

horizon = st.sidebar.slider("Volatility Forecast Horizon (Trading Days)", min_value=5, max_value=60, value=21)
force_refresh = st.sidebar.checkbox("Bypass Local 6-Hour Cache (--refresh)", value=False)
run_btn = st.sidebar.button("🚀 Run ATSA Analysis", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Applied Time Series Analysis (ATSA)**  
    University Project Specification:
    - Real public data only
    - Units 1, 2, 3, 4, 5 & 8
    - Residual bootstrap ($B=2000$)
    - Vector Autoregression & Cholesky IRF
    """
)

# Header
st.markdown('<div class="main-header">RiskLens: ATSA Risk & Macro-Linkage Monitor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Production-grade econometric diagnostics, ARMA conditional mean identification, '
    'realized volatility forecasting, and global macro vector autoregression for Indian financial assets.</div>',
    unsafe_allow_html=True,
)

if run_btn or "analysis_data" in st.session_state:
    if run_btn:
        with st.spinner("Executing rigorous ATSA econometric pipeline on live real data..."):
            try:
                data = execute_analysis(asset_mode, asset_input, benchmark_symbol, horizon, force_refresh)
                st.session_state["analysis_data"] = data
            except Exception as e:
                st.error(f"Analysis halted with error: {e}")
                st.stop()

    data = st.session_state["analysis_data"]

    # Data Freshness Stamp
    meta = data["meta"]
    st.markdown(
        f"""
        <div class="freshness-card">
            <b>Target Asset:</b> <span class="metric-badge">{data['display_name']} ({data['symbol_code']})</span> &nbsp;|&nbsp; 
            <b>Data Source:</b> {meta.get('source', 'Public API')} &nbsp;|&nbsp; 
            <b>Latest Available Date:</b> <span class="metric-badge">{meta.get('last_date')}</span> &nbsp;|&nbsp; 
            <b>Observations:</b> {len(data['clean_res'].log_return_series)} sessions &nbsp;|&nbsp; 
            <b>Fetch Timestamp:</b> {meta.get('fetch_timestamp')} &nbsp;|&nbsp;
            <b>Status:</b> {'⚠️ STALE CACHE' if meta.get('is_stale') else '🟢 LIVE FRESH (TTL < 6h)'}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data["warnings"]:
        for w in data["warnings"]:
            st.warning(w)

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        [
            "📋 Overview",
            "🔬 Stationarity & ACF",
            "📅 Seasonality",
            "📐 ARMA & EMH",
            "⚡ Volatility Forecast",
            "🎯 Benchmark Linkage",
            "🌐 Macro VAR",
            "📊 Out-of-Sample",
            "📑 Full Report",
        ]
    )

    # 1. Overview Tab
    with tab1:
        st.subheader("Executive ATSA Summary")
        st.markdown(data["narrative"])

        col1, col2, col3, col4 = st.columns(4)
        vol_info = data["vol_res"]
        col1.metric("Last Realized Vol", f"{vol_info.last_observed_vol_pct:.2f}% p.a.")
        col2.metric("30-Day Risk Forecast", f"{vol_info.point_forecast_pct:.2f}% p.a.")
        col3.metric("90% Risk Range", f"[{vol_info.expected_risk_range_pct[0]:.1f}% - {vol_info.expected_risk_range_pct[1]:.1f}%]")
        mkt_info = data["market_res"]
        if mkt_info:
            col4.metric(f"Beta vs {mkt_info.benchmark_name}", f"{mkt_info.arma_beta:.3f}")

        if not data["clean_res"].flagged_anomalies.empty:
            st.markdown("#### Flagged Outlier Return Dates (|r_t| > 15%)")
            st.dataframe(data["clean_res"].flagged_anomalies, use_container_width=True)

    # 2. Stationarity Tab
    with tab2:
        st.subheader("Unit Root & Stationarity Diagnostics (ATSA Units 1 & 5)")
        s_price = data["stat_suite"]["log_price"]
        s_ret = data["stat_suite"]["log_returns"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Log Price / NAV")
            st.write(f"- **ADF Statistic:** {s_price.adf_stat:.4f} (p-value: {s_price.adf_pvalue:.4f})")
            st.write(f"- **KPSS Statistic:** {s_price.kpss_stat:.4f} (p-value: {s_price.kpss_pvalue:.4f})")
            st.info(f"**Verdict:** {s_price.verdict}")
        with col2:
            st.markdown("##### Log Returns r_t")
            st.write(f"- **ADF Statistic:** {s_ret.adf_stat:.4f} (p-value: {s_ret.adf_pvalue:.4f})")
            st.write(f"- **KPSS Statistic:** {s_ret.kpss_stat:.4f} (p-value: {s_ret.kpss_pvalue:.4f})")
            st.success(f"**Verdict:** {s_ret.verdict}")

        st.markdown("---")
        st.subheader("Autocorrelation & Volatility Clustering")
        fig_acf = plot_acf_pacf_diagnostics(data["clean_res"].log_return_series, data["display_name"])
        st.pyplot(fig_acf)
        plt.close(fig_acf)

    # 3. Seasonality Tab
    with tab3:
        st.subheader("Seasonality, Calendar Anomalies & STL Decomposition (ATSA Unit 1)")
        cal = data["cal_res"]
        stl = data["stl_res"]

        st.write(f"- **Day-of-Week OLS Joint F-test:** Statistic={cal.ols_f_stat:.3f}, p-value={cal.ols_f_pvalue:.4f}")
        st.write(f"- **Kruskal-Wallis Non-Parametric Test:** Statistic={cal.kruskal_stat:.3f}, p-value={cal.kruskal_pvalue:.4f}")
        st.write(f"- **STL Trend Strength (F_T):** {stl.trend_strength_f_t:.4f} | **Seasonal Strength (F_S):** {stl.seasonal_strength_f_s:.4f}")
        st.info(f"**Calendar Verdict:** {cal.summary_verdict}")

        fig_seas = plot_seasonality_and_subseries(data["clean_res"].log_return_series, cal, data["display_name"])
        st.pyplot(fig_seas)
        plt.close(fig_seas)

        fig_stl = plot_stl_components(stl, data["display_name"])
        st.pyplot(fig_stl)
        plt.close(fig_stl)

    # 4. ARMA & EMH Tab
    with tab4:
        st.subheader("Conditional Mean ARMA Modeling & EMH Predictability (ATSA Units 3, 4 & 5)")
        arma = data["arma_search"]
        best = arma.best_model
        st.write(f"- **Best Model Selected by AICc:** ARIMA{arma.best_order}")
        st.write(f"- **AICc:** {best.aicc:.2f} | **BIC:** {best.bic:.2f} | **Delta AICc vs Constant Mean:** {best.aicc - arma.constant_model.aicc:.2f}")
        st.write(f"- **Residual Diagnostics:** Ljung-Box (Lag 10) p={best.lb_df.iloc[0]['lb_pvalue']:.4f}, Jarque-Bera p={best.jb_pvalue:.4f}")

        if arma.is_predictable:
            st.warning(f"**Verdict:** {arma.predictability_verdict}")
        else:
            st.success(f"**Verdict:** {arma.predictability_verdict}")

        fig_resid = plot_arma_residual_diagnostics(best, data["display_name"])
        st.pyplot(fig_resid)
        plt.close(fig_resid)

        st.markdown("#### Candidate Model Grid Search Comparison")
        st.dataframe(arma.all_models_df, use_container_width=True)

    # 5. Volatility Forecast Tab
    with tab5:
        st.subheader("Realized Volatility Modeling & Residual Bootstrap (ATSA Units 4, 5 & 8)")
        vol = data["vol_res"]
        st.write(
            f"Target: $v_t = \\ln(\\sqrt{{252}} \\cdot \\text{{std}}_{{21}}(r_t))$. "
            f"Fitted ARIMA{vol.best_order} on realized volatility."
        )
        st.write(f"- **Point Forecast (Median):** {vol.point_forecast_pct:.2f}% p.a.")
        st.write(f"- **90% Expected Risk Range:** [{vol.expected_risk_range_pct[0]:.2f}% - {vol.expected_risk_range_pct[1]:.2f}%] p.a.")
        st.write(f"- **Baselines:** Naive={vol.naive_forecast_pct:.2f}%, 21-Day Rolling Mean={vol.rolling_mean_forecast_pct:.2f}%")

        fig_vol = plot_volatility_forecast(vol, data["display_name"])
        st.pyplot(fig_vol)
        plt.close(fig_vol)

    # 6. Benchmark Linkage Tab
    with tab6:
        st.subheader(f"Systematic Benchmark Linkage: Regression with ARMA Errors (ATSA Units 2 & 5)")
        mkt = data["market_res"]
        if mkt:
            col1, col2, col3 = st.columns(3)
            col1.metric("Beta", f"{mkt.arma_beta:.3f}", f"SE: {mkt.arma_beta_se:.3f}")
            col2.metric("Annualized Alpha", f"{mkt.arma_alpha_annualized*100:+.2f}%", f"p-val: {mkt.arma_alpha_pvalue:.4f}")
            col3.metric("R-Squared (OLS)", f"{mkt.ols_r_squared:.3f}", f"DW: {mkt.ols_dw_stat:.2f}")

            st.info(mkt.interpretation)

            fig_mkt = plot_market_regression(mkt, data["clean_res"].log_return_series, data["bench_returns"])
            st.pyplot(fig_mkt)
            plt.close(fig_mkt)
        else:
            st.warning("Market benchmark regression unavailable.")

    # 7. Macro VAR Tab
    with tab7:
        st.subheader("Global Macro Linkages: Vector Autoregression (ATSA Unit 8)")
        var_res = data["var_res"]
        if var_res:
            st.write(f"- **System Dimension:** 5 variables across {var_res.aligned_obs} common trading sessions.")
            st.write(f"- **Selected Lag Order:** p={var_res.selected_lag} | **Stability:** {'Stable (Roots < 1)' if var_res.is_stable else 'Unstable'}")
            st.write(f"- **Cholesky Ordering:** `{var_res.cholesky_order}`")

            st.markdown("##### Granger Causality Tests on NIFTY 50")
            for g in var_res.granger_tests:
                st.write(f"- {g.conclusion}")

            fig_var = plot_var_diagnostics_and_irf(var_res)
            st.pyplot(fig_var)
            plt.close(fig_var)

            st.markdown("##### NIFTY 50 Forecast Error Variance Decomposition (FEVD %)")
            st.dataframe(var_res.fevd_nifty_horizon10, use_container_width=True)
        else:
            st.warning("Macro VAR analysis was skipped due to insufficient aligned observations or system instability.")

    # 8. Out-of-Sample Evaluation Tab
    with tab8:
        st.subheader("Expanding-Window Rolling-Origin Out-of-Sample Evaluation (ATSA Units 4 & 5)")
        ev = data["eval_res"]
        st.write(f"- **Origins Evaluated:** {ev.n_origins_evaluated} expanding-window origins without look-ahead bias.")
        st.write(f"- **Horizon:** {ev.horizon} days | **Initial Window:** {ev.initial_window} days | **Step:** {ev.step_size} days")
        st.info(ev.summary_verdict)

        st.markdown("##### Metric Comparison Table")
        st.dataframe(ev.metrics_table, use_container_width=True)

    # 9. Full Report Tab
    with tab9:
        st.subheader("Complete ATSA Markdown Report")
        from risklens.report import build_markdown_report
        report_md = build_markdown_report(data["assembled_results"], data["narrative"])
        st.download_button("📥 Download Report (Markdown)", report_md, file_name=f"risklens_report_{data['symbol_code']}.md")
        st.markdown(report_md)
else:
    st.info("👈 Please select an Indian Mutual Fund or Equity Symbol in the sidebar and click **'Run ATSA Analysis'** to begin.")
