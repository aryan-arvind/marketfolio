"""
RiskLens: Quantitative Risk & Macro-Linkage Terminal.
Interactive institutional dashboard for Indian equities and mutual funds.
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

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
from risklens.report import build_markdown_report
from risklens.seasonality import (
    analyze_calendar_effects,
    plot_seasonality_and_subseries,
    plot_stl_components,
    run_stl_decomposition,
)
from risklens.var_analysis import plot_var_diagnostics_and_irf, run_var_analysis
from risklens.volatility import forecast_volatility, plot_volatility_forecast

st.set_page_config(
    page_title="RiskLens | Quantitative Risk & Macro Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Institutional Dark/Slate Theme (No neon, no playful styling)
st.markdown(
    """
    <style>
    /* Global App Container */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], header {
        background-color: #0b0f19 !important;
        color: #f1f5f9;
    }
    [data-testid="stSidebar"] {
        background-color: #0e1424 !important;
        border-right: 1px solid #1e293b;
    }
    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #151d2f;
        border: 1px solid #23314a;
        border-radius: 4px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
    }
    .metric-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.25rem;
    }
    .metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    .provenance-card {
        background-color: #121827;
        border-left: 3px solid #3b82f6;
        border-top: 1px solid #1e293b;
        border-right: 1px solid #1e293b;
        border-bottom: 1px solid #1e293b;
        padding: 0.65rem 1rem;
        border-radius: 3px;
        margin-bottom: 1.2rem;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #23314a;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-size: 0.88rem;
        color: #94a3b8;
        border-radius: 4px 4px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        color: #f8fafc !important;
        border-bottom: 2px solid #3b82f6 !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=21600, show_spinner=False)
def execute_analysis_cached(
    asset_mode: str,
    asset_input: str,
    benchmark_symbol: str,
    horizon: int,
    force_refresh: bool,
    max_origins: int = 12,
) -> dict[str, Any]:
    """
    Cached quantitative analysis runner with 6-hour TTL.
    Executes data retrieval, cleaning, statistical modeling, and backtesting.
    """
    warnings_list: list[str] = []

    if asset_mode == "Indian Mutual Fund":
        scheme_code, scheme_name = resolve_mutual_fund(asset_input)
        raw_price, meta = fetch_mutual_fund_nav(scheme_code, scheme_name=scheme_name, force_refresh=force_refresh)
        display_name = scheme_name
        symbol_code = str(scheme_code)
        if "dividend" in scheme_name.lower() or "idcw" in scheme_name.lower():
            warnings_list.append(
                f"Selected fund '{scheme_name}' is a Dividend/IDCW plan. "
                "Dividend payouts distort NAV returns; Growth plans provide clean return series."
            )
    else:
        raw_price, meta = fetch_yfinance_ticker(asset_input, force_refresh=force_refresh, is_primary=True)
        display_name = asset_input
        symbol_code = asset_input

    if meta.get("is_stale"):
        warnings_list.append(f"Network interruption: Using cached data as of {meta.get('last_date')}.")

    # 1. Clean primary series
    clean_res = clean_primary_asset(raw_price, asset_name=display_name)
    prices = clean_res.price_series
    returns = clean_res.log_return_series

    # 2. Stationarity & Autocorrelation
    stat_suite = run_stationarity_suite(prices, returns, display_name)
    lb_returns = compute_ljung_box(returns, lags=[10, 20])
    lb_sq_returns = compute_ljung_box(returns ** 2, lags=[10, 20])

    # 3. Calendar Tendencies & Decomposition
    cal_res = analyze_calendar_effects(returns, display_name)
    stl_res = run_stl_decomposition(prices, display_name, period=21)

    # 4. Conditional Mean ARMA
    arma_search = grid_search_arma(returns, display_name, d=0, max_p=3, max_q=3)

    # 5. Volatility Modeling & Bootstrap Forecast
    vol_res = forecast_volatility(returns, display_name, horizon=horizon)

    # 6. Benchmark Sensitivity Regression
    bench_clean = None
    market_res = None
    try:
        bench_raw, _ = fetch_yfinance_ticker(benchmark_symbol, force_refresh=force_refresh, is_primary=False)
        bench_clean = clean_primary_asset(bench_raw, asset_name=benchmark_symbol, min_observations=250)
        market_res = fit_regression_arma(returns, bench_clean.log_return_series, asset_name=display_name, benchmark_name=benchmark_symbol)
    except Exception as exc:
        warnings_list.append(f"Benchmark regression vs {benchmark_symbol} skipped: {exc}")

    # 7. Global Macro VAR System
    macro_dict, _ = fetch_all_macro_series(force_refresh=force_refresh)
    aligned_macro, macro_valid, macro_reason = align_macro_series(macro_dict)
    var_res = None
    if macro_valid:
        v_res, v_ok, _ = run_var_analysis(aligned_macro)
        if v_ok:
            var_res = v_res

    # 8. Out-of-Sample Rolling-Origin Backtesting
    eval_res = run_rolling_origin_evaluation(
        series=vol_res.log_vol_series,
        target_name=f"{display_name} Volatility",
        model_order=vol_res.best_order,
        horizon=horizon,
        max_origins=max_origins,
    )

    # 9. Assembled Results Structure
    first_lb_p = (
        float(arma_search.best_model.lb_df.iloc[0]["lb_pvalue"])
        if len(arma_search.best_model.lb_df) > 0
        else 0.0
    )
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
            "price_adf_stat": stat_suite["log_price"].adf_stat,
            "price_adf_pvalue": stat_suite["log_price"].adf_pvalue,
            "price_kpss_stat": stat_suite["log_price"].kpss_stat,
            "price_kpss_pvalue": stat_suite["log_price"].kpss_pvalue,
            "returns_adf_stat": stat_suite["log_returns"].adf_stat,
            "returns_kpss_stat": stat_suite["log_returns"].kpss_stat,
            "lb_returns_pvalue_lag10": float(lb_returns.loc[10, "lb_pvalue"]),
            "lb_sq_returns_pvalue_lag10": float(lb_sq_returns.loc[10, "lb_pvalue"]),
        },
        "seasonality": {
            "day_of_week_significant": cal_res.is_dow_significant,
            "ols_f_stat": cal_res.ols_f_stat,
            "ols_f_pvalue": cal_res.ols_f_pvalue,
            "kruskal_stat": cal_res.kruskal_stat,
            "kruskal_pvalue": cal_res.kruskal_pvalue,
            "trend_strength_f_t": stl_res.trend_strength_f_t,
            "seasonal_strength_f_s": stl_res.seasonal_strength_f_s,
            "summary_verdict": cal_res.summary_verdict,
        },
        "arma_predictability": {
            "best_order": str(arma_search.best_order),
            "best_aicc": arma_search.best_model.aicc,
            "best_bic": arma_search.best_model.bic,
            "constant_aicc": arma_search.constant_model.aicc,
            "delta_aicc_vs_constant": arma_search.best_model.aicc - arma_search.constant_model.aicc,
            "is_predictable": arma_search.is_predictable,
            "residual_lb_pvalue": first_lb_p,
            "residual_jb_pvalue": arma_search.best_model.jb_pvalue,
            "verdict": arma_search.predictability_verdict,
        },
        "volatility_forecast": {
            "last_observed_vol_pct": vol_res.last_observed_vol_pct,
            "point_forecast_pct": vol_res.point_forecast_pct,
            "expected_risk_range_lower_pct": vol_res.expected_risk_range_pct[0],
            "expected_risk_range_upper_pct": vol_res.expected_risk_range_pct[1],
            "naive_forecast_pct": vol_res.naive_forecast_pct,
            "rolling_mean_forecast_pct": vol_res.rolling_mean_forecast_pct,
            "bootstrap_fallback_used": vol_res.boot_failed_or_fallback,
        },
        "market_model": (
            {
                "benchmark_name": market_res.benchmark_name,
                "n_aligned_obs": market_res.n_aligned_obs,
                "start_date": market_res.start_date,
                "end_date": market_res.end_date,
                "beta": market_res.arma_beta,
                "beta_se": market_res.arma_beta_se,
                "beta_ci_lower": market_res.arma_beta_ci[0],
                "beta_ci_upper": market_res.arma_beta_ci[1],
                "alpha_annualized_pct": market_res.arma_alpha_annualized * 100.0,
                "alpha_pvalue": market_res.arma_alpha_pvalue,
                "r_squared": market_res.ols_r_squared,
                "durbin_watson": market_res.ols_dw_stat,
                "ols_lb_pvalue": market_res.ols_residual_lb_pvalue,
                "interpretation": market_res.interpretation,
            }
            if market_res
            else {}
        ),
        "var_macro": (
            {
                "available": True,
                "aligned_obs": var_res.aligned_obs,
                "selected_lag": var_res.selected_lag,
                "cholesky_order": var_res.cholesky_order,
                "granger_causal_factors": [g.causing_variable for g in var_res.granger_tests if g.is_causal_05],
                "granger_details": [
                    {
                        "var": g.causing_variable,
                        "f_stat": g.f_statistic,
                        "p_val": g.p_value,
                        "conclusion": g.conclusion,
                    }
                    for g in var_res.granger_tests
                ],
                "var_beats_baseline": var_res.var_beats_baseline,
                "oos_verdict": var_res.out_of_sample_verdict,
            }
            if var_res
            else {"available": False, "reason": macro_reason}
        ),
        "evaluation": {
            "n_origins": eval_res.n_origins_evaluated,
            "horizon": eval_res.horizon,
            "model_mase_improvement_pct": eval_res.model_mase_improvement_pct,
            "table_rows": eval_res.metrics_table.to_dict(orient="records"),
            "verdict": eval_res.summary_verdict,
        },
        "flagged_anomalies": clean_res.flagged_anomalies.to_dict(orient="records"),
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
st.sidebar.markdown("### RiskLens Parameters")
asset_mode = st.sidebar.radio(
    "Asset Selection",
    ["Indian Stock / Index", "Indian Mutual Fund"],
    index=0,
)

if asset_mode == "Indian Stock / Index":
    asset_input = st.sidebar.text_input(
        "Yahoo Finance Symbol",
        value="RELIANCE.NS",
        help="NSE equity or index symbol with .NS suffix (e.g. RELIANCE.NS, TCS.NS, HDFCBANK.NS, ^NSEI)",
    )
else:
    asset_input = st.sidebar.text_input(
        "AMFI Fund Query",
        value="Nifty 50 Index Fund Direct Growth",
        help="Search keyword to resolve mutual fund from AMFI database (e.g. Parag Parikh Flexi Cap Direct Growth)",
    )

benchmark_symbol = st.sidebar.text_input(
    "Market Benchmark",
    value=DEFAULT_BENCHMARK_SYMBOL,
    help="Default benchmark is NIFTY 50 (^NSEI)",
)

horizon = st.sidebar.slider(
    "Volatility Forecast Horizon (Days)",
    min_value=5,
    max_value=60,
    value=DEFAULT_FORECAST_HORIZON,
)

max_origins = st.sidebar.select_slider(
    "Backtest Evaluation Depth",
    options=[6, 12, 18, 24, 36],
    value=12,
    help="Number of expanding-window rolling origins. 12 origins runs in ~10 seconds; 36 origins performs deep historical backtesting.",
)

force_refresh = st.sidebar.checkbox("Bypass Local 6h Cache", value=False)
run_btn = st.sidebar.button("Run Quantitative Analysis", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="font-size:0.78rem; color:#64748b; line-height:1.4;">
    <b>Quantitative Architecture</b><br>
    - Stationarity & Order of Integration<br>
    - Return Autocorrelation & Volatility Clustering<br>
    - Day/Month Calendar Tendencies & STL<br>
    - Realized Volatility Bootstrap Cone (B=2000)<br>
    - Single-Index Market Beta & Alpha<br>
    - 5-Variable Macro Vector Autoregression<br>
    - Expanding-Window Out-of-Sample Backtesting
    </div>
    """,
    unsafe_allow_html=True,
)

# Main Terminal Header
st.markdown('<div class="main-title">RiskLens: Quantitative Risk & Macro Monitor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Real-time risk assessment, volatility forecasting, and macro linkage analysis for Indian financial markets.</div>',
    unsafe_allow_html=True,
)

needs_execution = run_btn or ("analysis_data" not in st.session_state)

if needs_execution:
    with st.status(f"Calibrating quantitative models for {asset_input}...", expanded=True) as status_box:
        status_box.write("Ingesting historical price series and verifying integrity...")
        try:
            data = execute_analysis_cached(
                asset_mode=asset_mode,
                asset_input=asset_input,
                benchmark_symbol=benchmark_symbol,
                horizon=horizon,
                force_refresh=force_refresh,
                max_origins=max_origins,
            )
            st.session_state["analysis_data"] = data
            status_box.update(label=f"Analysis calibrated for {asset_input}", state="complete", expanded=False)
        except Exception as e:
            status_box.update(label=f"Analysis failed: {e}", state="error", expanded=True)
            st.error(f"Error during execution: {e}")
            st.stop()

if "analysis_data" in st.session_state:
    data = st.session_state["analysis_data"]
    meta = data["meta"]

    # Data Provenance Banner
    st.markdown(
        f"""
        <div class="provenance-card">
            <b>Target Asset:</b> {data['display_name']} ({data['symbol_code']}) &nbsp;&nbsp;|&nbsp;&nbsp;
            <b>Source:</b> {meta.get('source', 'Public API')} &nbsp;&nbsp;|&nbsp;&nbsp;
            <b>Last Date:</b> {meta.get('last_date')} &nbsp;&nbsp;|&nbsp;&nbsp;
            <b>Cleaned Observations:</b> {len(data['clean_res'].log_return_series)} &nbsp;&nbsp;|&nbsp;&nbsp;
            <b>Status:</b> {'Stale Cache (Offline Fallback)' if meta.get('is_stale') else 'Live Verified'}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data["warnings"]:
        for w in data["warnings"]:
            st.warning(w)

    # 6 Institutional Financial Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Overview & Risk Summary",
            "Return Dynamics & Seasonality",
            "Volatility & Risk Forecast",
            "Market & Macro Drivers",
            "Model Diagnostics & Backtest",
            "Audit Report & Export",
        ]
    )

    # ==========================================
    # TAB 1: Executive Overview & Key Metrics
    # ==========================================
    with tab1:
        st.subheader("Key Quantitative Metrics")
        vol_info = data["vol_res"]
        mkt_info = data["market_res"]
        stat_info = data["stat_suite"]["log_returns"]
        latest_price = float(data["clean_res"].price_series.iloc[-1])

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Latest Close</div>
                    <div class="metric-val">{latest_price:,.2f}</div>
                    <div class="metric-sub">Date: {meta.get('last_date')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Realized Vol (21d)</div>
                    <div class="metric-val">{vol_info.last_observed_vol_pct:.1f}%</div>
                    <div class="metric-sub">Annualized</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">30d Vol Forecast</div>
                    <div class="metric-val">{vol_info.point_forecast_pct:.1f}%</div>
                    <div class="metric-sub">Median Expected</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">30d Risk Envelope</div>
                    <div class="metric-val">[{vol_info.expected_risk_range_pct[0]:.0f}% - {vol_info.expected_risk_range_pct[1]:.0f}%]</div>
                    <div class="metric-sub">90% Confidence Band</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c5:
            beta_str = f"{mkt_info.arma_beta:.2f}" if mkt_info else "N/A"
            alpha_str = f"{mkt_info.arma_alpha_annualized*100:+.2f}%" if mkt_info else "N/A"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Market Beta / Alpha</div>
                    <div class="metric-val">{beta_str}</div>
                    <div class="metric-sub">Alpha: {alpha_str} p.a.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.subheader("Quantitative Risk & Market Takeaways")
        st.markdown(data["narrative"])

        if not data["clean_res"].flagged_anomalies.empty:
            st.markdown("#### Extreme Return Events (|Daily Return| > 15%)")
            st.dataframe(data["clean_res"].flagged_anomalies, use_container_width=True)

    # ==========================================
    # TAB 2: Return Dynamics & Seasonality
    # ==========================================
    with tab2:
        st.subheader("Autocorrelation & Volatility Clustering")
        st.markdown(
            "Autocorrelation of raw log returns evaluates directional serial dependency, "
            "while squared log returns capture volatility clustering (ARCH effects)."
        )
        fig_acf = plot_acf_pacf_diagnostics(data["clean_res"].log_return_series, data["display_name"])
        st.pyplot(fig_acf)
        plt.close(fig_acf)

        st.markdown("---")
        st.subheader("Calendar Return Patterns & Lag Structure")
        st.markdown(
            f"Evaluates systematic return variation across days of the week and months of the year. "
            f"**Joint OLS F-test p-value: {data['cal_res'].ols_f_pvalue:.4f}** | "
            f"**Kruskal-Wallis p-value: {data['cal_res'].kruskal_pvalue:.4f}**"
        )
        fig_seas = plot_seasonality_and_subseries(data["clean_res"].log_return_series, data["cal_res"], data["display_name"])
        st.pyplot(fig_seas)
        plt.close(fig_seas)

        st.markdown("---")
        st.subheader("Price Decomposition: Trend vs Seasonality")
        st.markdown(
            f"Robust STL decomposition separating macro trend from seasonal cycles. "
            f"**Trend Strength (F_T): {data['stl_res'].trend_strength_f_t:.3f}** | "
            f"**Seasonal Strength (F_S): {data['stl_res'].seasonal_strength_f_s:.3f}**"
        )
        fig_stl = plot_stl_components(data["stl_res"], data["display_name"])
        st.pyplot(fig_stl)
        plt.close(fig_stl)

    # ==========================================
    # TAB 3: Volatility & Risk Forecast
    # ==========================================
    with tab3:
        st.subheader("30-Day Realized Volatility Forecast & Risk Cone")
        st.markdown(
            "Forecasts natural log realized volatility using an autoregressive integrated moving average model, "
            "supplemented with 2,000 recursive residual bootstrap simulations to construct empirical 90% confidence bands."
        )
        fig_vol = plot_volatility_forecast(data["vol_res"], data["display_name"])
        st.pyplot(fig_vol)
        plt.close(fig_vol)

        st.markdown("#### Volatility Forecast Summary Table")
        vol_df = pd.DataFrame(
            [
                {
                    "Metric": "Current Realized Volatility (21-Day Rolling)",
                    "Annualized Value": f"{data['vol_res'].last_observed_vol_pct:.2f}% p.a.",
                    "Context": "Historical benchmark from most recent session",
                },
                {
                    "Metric": "ARIMA Model Point Forecast (Median)",
                    "Annualized Value": f"{data['vol_res'].point_forecast_pct:.2f}% p.a.",
                    "Context": f"Optimal ARIMA{data['vol_res'].best_order} specification",
                },
                {
                    "Metric": "Expected 30-Day Risk Range (90% Interval)",
                    "Annualized Value": f"[{data['vol_res'].expected_risk_range_pct[0]:.2f}% - {data['vol_res'].expected_risk_range_pct[1]:.2f}%] p.a.",
                    "Context": "5th to 95th percentile bounds from 2,000 bootstrap paths",
                },
                {
                    "Metric": "Naive Baseline Forecast",
                    "Annualized Value": f"{data['vol_res'].naive_forecast_pct:.2f}% p.a.",
                    "Context": "Assumes zero volatility change (random walk heuristic)",
                },
                {
                    "Metric": "21-Day Rolling Mean Baseline",
                    "Annualized Value": f"{data['vol_res'].rolling_mean_forecast_pct:.2f}% p.a.",
                    "Context": "Historical moving average baseline",
                },
            ]
        )
        st.table(vol_df)

    # ==========================================
    # TAB 4: Market & Macro Drivers
    # ==========================================
    with tab4:
        st.subheader("Systematic Market Sensitivity (Beta & Alpha)")
        mkt = data["market_res"]
        if mkt:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Market Beta", f"{mkt.arma_beta:.3f}", f"SE: {mkt.arma_beta_se:.3f}")
            col2.metric("Annualized Alpha", f"{mkt.arma_alpha_annualized*100:+.2f}%", f"p={mkt.arma_alpha_pvalue:.4f}")
            col3.metric("R-Squared", f"{mkt.ols_r_squared:.3f}")
            col4.metric("Durbin-Watson Stat", f"{mkt.ols_dw_stat:.2f}")

            st.markdown(f"**Empirical Finding**: {mkt.interpretation}")

            fig_mkt = plot_market_regression(mkt, data["clean_res"].log_return_series, data["bench_returns"])
            st.pyplot(fig_mkt)
            plt.close(fig_mkt)
        else:
            st.warning("Benchmark market regression is unavailable.")

        st.markdown("---")
        st.subheader("Global Macro Transmission: Vector Autoregression")
        var_res = data["var_res"]
        if var_res:
            st.markdown(
                f"Estimates a 5-variable macro system [S&P 500, Brent Crude, USD/INR, India VIX, NIFTY 50] "
                f"across **{var_res.aligned_obs}** overlapping trading dates. "
                f"Optimal lag order: **p={var_res.selected_lag}** | "
                f"System Stability: **{'Stable (all roots < 1)' if var_res.is_stable else 'Unstable'}**"
            )

            st.markdown("##### Granger Causality Lead-Lag Tests on NIFTY 50")
            granger_table = pd.DataFrame(
                [
                    {
                        "Macro Factor": g.causing_variable,
                        "F-Statistic": f"{g.f_statistic:.2f}",
                        "p-value": f"{g.p_value:.4f}",
                        "Leads NIFTY 50 (5% sig)": "Yes" if g.is_causal_05 else "No",
                        "Transmission Verdict": g.conclusion,
                    }
                    for g in var_res.granger_tests
                ]
            )
            st.table(granger_table)

            fig_var = plot_var_diagnostics_and_irf(var_res)
            st.pyplot(fig_var)
            plt.close(fig_var)

            st.markdown("##### Forecast Error Variance Decomposition (FEVD) of NIFTY 50 (%)")
            st.dataframe(var_res.fevd_nifty_horizon10, use_container_width=True)
        else:
            st.warning("Global macro VAR analysis was skipped due to insufficient overlapping observations.")

    # ==========================================
    # TAB 5: Model Diagnostics & Backtest
    # ==========================================
    with tab5:
        st.subheader("Model Residual Diagnostics")
        best_arma = data["arma_search"].best_model
        st.markdown(
            f"Assesses whether residual errors of the selected model ARIMA{data['arma_search'].best_order} "
            f"behave as Gaussian white noise. "
            f"**Ljung-Box p-value (Lag 10): {best_arma.lb_df.iloc[0]['lb_pvalue']:.4f}** | "
            f"**Jarque-Bera Normality p-value: {best_arma.jb_pvalue:.4f}**"
        )
        fig_resid = plot_arma_residual_diagnostics(best_arma, data["display_name"])
        st.pyplot(fig_resid)
        plt.close(fig_resid)

        st.markdown("---")
        st.subheader("Out-of-Sample Expanding-Window Backtest")
        ev = data["eval_res"]
        st.markdown(
            f"Strictly chronological expanding-window rolling-origin evaluation without look-ahead bias across "
            f"**{ev.n_origins_evaluated} evaluation origins** (Horizon = {ev.horizon} days)."
        )
        st.markdown(f"**Backtest Conclusion**: {ev.summary_verdict}")

        st.markdown("##### Accuracy Comparison vs Competing Baselines")
        st.table(ev.metrics_table)

    # ==========================================
    # TAB 6: Full Audit Report & Data Export
    # ==========================================
    with tab6:
        st.subheader("Audit Report & Data Export")
        report_md = build_markdown_report(data["assembled_results"], data["narrative"])

        col_d1, col_d2 = st.columns([1, 4])
        with col_d1:
            st.download_button(
                "Download Markdown Report",
                report_md,
                file_name=f"risklens_report_{data['symbol_code']}.md",
                mime="text/markdown",
            )

        st.markdown("---")
        st.markdown(report_md)
else:
    st.info("Select an Indian Stock symbol or Mutual Fund query in the sidebar and click **'Run Quantitative Analysis'** to begin.")
