"""
MarketFolio Tab 8: Model Library & Data Quality.
Institutional econometric model documentation, assumptions, and pipeline architecture.
Professional research-desk orientation: no unit numbers, no syllabus marks.
"""

from __future__ import annotations

from typing import Any
import pandas as pd
import streamlit as st

from ui.components import render_method_tag
from ui.theme import ACCENT_TEAL, BORDER_COLOR, TEXT_MUTED


def render(data: dict[str, Any] | None = None) -> None:
    """Renders the Methodology & Model Library tab."""
    render_method_tag("Econometric Model Specifications, Pipeline Protocol & Data Governance")

    st.markdown('<div style="font-size:0.95rem; font-weight:700; margin-bottom:0.5rem;">Pipeline Processing Flow</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background-color: #111827; border: 1px solid #23314a; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1.5rem;">
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 0.5rem; font-size: 0.8rem; font-weight: 600;">
                <span style="color: {ACCENT_TEAL};">1. INGEST</span>
                <span style="color: {TEXT_MUTED};">&rarr;</span>
                <span style="color: {ACCENT_TEAL};">2. CLEAN & ALIGN</span>
                <span style="color: {TEXT_MUTED};">&rarr;</span>
                <span style="color: {ACCENT_TEAL};">3. EXPLORE</span>
                <span style="color: {TEXT_MUTED};">&rarr;</span>
                <span style="color: {ACCENT_TEAL};">4. DECOMPOSE</span>
                <span style="color: {TEXT_MUTED};">&rarr;</span>
                <span style="color: {ACCENT_TEAL};">5. FORECAST</span>
                <span style="color: {TEXT_MUTED};">&rarr;</span>
                <span style="color: {ACCENT_TEAL};">6. EVALUATE</span>
                <span style="color: {TEXT_MUTED};">&rarr;</span>
                <span style="color: {ACCENT_TEAL};">7. INTERPRET</span>
            </div>
            <div style="font-size: 0.76rem; color: #64748b; margin-top: 0.6rem;">
                Official AMFI NAV API & NSE Yahoo Finance &rarr; Complete Business Day Reindex &rarr; Joint Stationarity Checks &rarr; STL & Calendar Regressions &rarr; ARIMA Mean & Volatility Models &rarr; Expanding-Window Backtest &rarr; Institutional Research Note
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div style="font-size:0.95rem; font-weight:700; margin-bottom:0.5rem;">Econometric Technique Reference</div>', unsafe_allow_html=True)

    tech_df = pd.DataFrame([
        {
            "Technique": "Stationarity Testing (ADF & KPSS)",
            "Where It Appears": "Returns & Predictability · Diagnostics",
            "Why It Matters to a Trader": "Distinguishes persistent mean-reverting series from stochastic unit-root drifts to prevent spurious regression errors.",
        },
        {
            "Technique": "Seasonal-Trend Decomposition (STL)",
            "Where It Appears": "Price & Trend",
            "Why It Matters to a Trader": "Isolates long-term secular drift from recurring monthly liquidity cycles and idiosyncratic daily noise.",
        },
        {
            "Technique": "Calendar-Effect Hypothesis Testing",
            "Where It Appears": "Price & Trend",
            "Why It Matters to a Trader": "Evaluates whether day-of-week (e.g. Friday effect) or month-of-year return biases represent tradable edges.",
        },
        {
            "Technique": "Regression with ARMA Errors (SARIMAX)",
            "Where It Appears": "Market & Macro",
            "Why It Matters to a Trader": "Delivers unbiased market beta and alpha by explicitly modeling autocorrelated disturbances instead of assuming i.i.d. noise.",
        },
        {
            "Technique": "ARMA Order Selection via AICc",
            "Where It Appears": "Returns & Predictability",
            "Why It Matters to a Trader": "Tests Fama weak-form market efficiency; penalizes overparameterization to prevent in-sample curve fitting.",
        },
        {
            "Technique": "Residual Diagnostics (Ljung-Box & Jarque-Bera)",
            "Where It Appears": "Forecast Accuracy · Details",
            "Why It Matters to a Trader": "Validates whether model errors are Gaussian white noise; catches lingering linear dependency and fat tails.",
        },
        {
            "Technique": "Vector Autoregression (VAR, Granger & FEVD)",
            "Where It Appears": "Market & Macro",
            "Why It Matters to a Trader": "Maps macro lead-lag transmission (Crude, Currency, VIX, US Equities) to understand global risk contagion.",
        },
        {
            "Technique": "Realized Volatility Modeling",
            "Where It Appears": "Volatility & Risk",
            "Why It Matters to a Trader": "Models log annualized volatility dynamics to establish expected trading ranges for options and portfolio sizing.",
        },
        {
            "Technique": "Residual Bootstrap Simulation (B=2,000)",
            "Where It Appears": "Volatility & Risk",
            "Why It Matters to a Trader": "Generates empirical 90% risk envelopes without imposing unrealistic Gaussian normality assumptions on volatility innovations.",
        },
        {
            "Technique": "Expanding-Window Rolling-Origin Backtest",
            "Where It Appears": "Forecast Accuracy",
            "Why It Matters to a Trader": "Rigorously verifies out-of-sample forecast skill across chronological market regimes with zero look-ahead leakage.",
        },
    ])

    st.dataframe(tech_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    col_clean, col_assump = st.columns(2)

    with col_clean:
        st.markdown('<div style="font-size:0.88rem; font-weight:600; margin-bottom:0.4rem;">Data Cleaning & Harmonization Protocol</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size:0.82rem; color:{TEXT_MUTED}; line-height:1.6;">
                1. <b>Business Day Grid</b>: Reindexes series to complete business days (<code>pd.bdate_range</code>).<br>
                2. <b>Gap Imputation</b>: Forward-fills exchange holidays and minor gaps of &le; 3 consecutive business days.<br>
                3. <b>Extended Gap Rejection</b>: Truncates data if disruptions exceed 3 days to preserve temporal integrity.<br>
                4. <b>Log Returns</b>: Calculates continuous compounded returns $r_t = \\ln(P_t / P_{{t-1}})$.<br>
                5. <b>Anomaly Quarantine</b>: Screens and flags extreme single-session movements $|r_t| > 15%$ (e.g. unadjusted corporate actions, splits).
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_assump:
        st.markdown('<div style="font-size:0.88rem; font-weight:600; margin-bottom:0.4rem;">Key Modeling Assumptions</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size:0.82rem; color:{TEXT_MUTED}; line-height:1.6;">
                1. <b>Trading Year</b>: 252 business days annualization factor $(\\sqrt{{252}})$.<br>
                2. <b>Stationarity Pre-requisite</b>: All ARMA and VAR engines require confirmed integrated difference stationarity.<br>
                3. <b>Macro Causality</b>: Cholesky orthogonalization recursive ordering: US Equities &rarr; Commodities &rarr; FX &rarr; Volatility &rarr; Domestic Index.<br>
                4. <b>Information Criterion</b>: Hurvich-Tsai corrected AICc is prioritized for small-sample stability over raw AIC.<br>
                5. <b>No Look-Ahead</b>: All out-of-sample evaluations strictly enforce $t_{{train}} < t_{{test}}$ via <code>assert_no_leakage</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )
