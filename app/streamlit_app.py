"""
MarketFolio: Quantitative Risk & Macro Terminal.
Interactive institutional analytics desk for Indian equities and mutual funds.
Run via: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st

# Ensure root and src are on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import importlib
from risklens.config import DEFAULT_BENCHMARK_SYMBOL, DEFAULT_FORECAST_HORIZON
from ui.components import render_header_strip, render_notice_card
import ui.charts
import ui.components
import ui.readings
import ui.theme
from ui.pages import (
    classical_diagnostics,
    forecast_accuracy,
    gcn_page,
    market_macro,
    methodology,
    overview,
    price_trend,
    report,
    returns_predictability,
    transformer_page,
    trello_page,
    vae_page,
    volatility_risk,
)

# Ensure hot-reloads always pull the latest ui module state in running servers
importlib.reload(ui.theme)
importlib.reload(ui.charts)
importlib.reload(ui.components)
importlib.reload(ui.readings)
importlib.reload(overview)
importlib.reload(vae_page)
importlib.reload(transformer_page)
importlib.reload(gcn_page)
importlib.reload(volatility_risk)
importlib.reload(classical_diagnostics)
importlib.reload(trello_page)
importlib.reload(report)
importlib.reload(methodology)

from ui.pipeline import execute_analysis_pipeline
from ui.state import clear_analysis_data, get_analysis_data, set_analysis_data
from ui.theme import GLOBAL_CSS, apply_theme

# Page Configuration
st.set_page_config(
    page_title="MarketFolio | Quantitative Risk & Macro Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Institutional Dark Theme & Plotly Template
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
apply_theme()


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown(
    '<div style="font-size:1.1rem; font-weight:700; color:#f8fafc; margin-bottom:1rem; letter-spacing:-0.02em;">MarketFolio</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="mf-card-title">Instrument</div>', unsafe_allow_html=True)
asset_mode = st.sidebar.radio(
    "Asset Mode",
    ["Indian Stock / Index", "Indian Mutual Fund"],
    index=0,
    label_visibility="collapsed",
)

if asset_mode == "Indian Stock / Index":
    asset_input = st.sidebar.text_input(
        "Yahoo Finance Ticker",
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
    "Benchmark Symbol",
    value=DEFAULT_BENCHMARK_SYMBOL,
    help="Default benchmark is NIFTY 50 (^NSEI)",
)

st.sidebar.markdown('<div class="mf-card-title" style="margin-top:1rem;">Settings</div>', unsafe_allow_html=True)
horizon = st.sidebar.slider(
    "Forecast Horizon (Days)",
    min_value=5,
    max_value=60,
    value=DEFAULT_FORECAST_HORIZON,
    help="Realized volatility ARIMA forecast horizon",
)

max_origins = st.sidebar.select_slider(
    "Backtest Evaluation Depth",
    options=[6, 12, 18, 24, 36],
    value=12,
    help="Number of expanding-window rolling origins for out-of-sample backtesting",
)

force_refresh = st.sidebar.checkbox("Bypass Local Cache", value=False)
run_btn = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

# If input target changed in sidebar without running, invalidate stale cache
current_data = get_analysis_data()
if current_data and st.session_state.get("analyzed_target") != asset_input and not run_btn:
    clear_analysis_data()
    current_data = None

# Pipeline execution trigger
if run_btn:
    with st.status(f"Calibrating quantitative models for {asset_input}...", expanded=True) as status_box:
        try:
            data = execute_analysis_pipeline(
                asset_mode=asset_mode,
                asset_input=asset_input,
                benchmark_symbol=benchmark_symbol,
                horizon=horizon,
                force_refresh=force_refresh,
                max_origins=max_origins,
                status_box=status_box,
            )
            set_analysis_data(data, asset_input)
            status_box.update(label=f"Analysis calibrated for {asset_input}", state="complete", expanded=False)
            current_data = data
        except Exception as exc:
            status_box.update(label=f"Analysis failed: {exc}", state="error", expanded=True)
            render_notice_card(
                title="Analysis Execution Error",
                message=f"Failed to complete quantitative pipeline for '{asset_input}': {exc}. Check ticker symbol or internet connection.",
                level="error",
            )
            st.stop()


# ==========================================
# MAIN APPLICATION VIEW
# ==========================================
if current_data:
    # 1. Persistent Stocktwits-style Header Strip
    render_header_strip(current_data)

    # 2. System Warnings (if any)
    if current_data.get("warnings"):
        for w in current_data["warnings"]:
            render_notice_card(title="Pipeline Advisory", message=w, level="warning")

    # 3. Evolved Financial & Machine Learning Desk Tabs
    tabs = st.tabs([
        "Overview",
        "Deep Anomaly (VAE)",
        "Volatility Regimes (Transformer)",
        "Inter-Market Topology (GCN)",
        "Realized Volatility Cone",
        "Classical Econometrics",
        "Research Tracker (Trello)",
        "Research Report",
        "Methodology",
    ])

    with tabs[0]:
        try:
            overview.render(current_data)
        except Exception as exc:
            render_notice_card("Overview Display Notice", f"Unable to display Overview: {exc}", level="error")

    with tabs[1]:
        try:
            vae_page.render(current_data)
        except Exception as exc:
            render_notice_card("VAE Anomaly Display Notice", f"Unable to display VAE Anomaly Engine: {exc}", level="error")

    with tabs[2]:
        try:
            transformer_page.render(current_data)
        except Exception as exc:
            render_notice_card("Transformer Display Notice", f"Unable to display Volatility Transformer: {exc}", level="error")

    with tabs[3]:
        try:
            gcn_page.render(current_data)
        except Exception as exc:
            render_notice_card("GCN Display Notice", f"Unable to display Inter-Market GCN: {exc}", level="error")

    with tabs[4]:
        try:
            volatility_risk.render(current_data)
        except Exception as exc:
            render_notice_card("Volatility Cone Display Notice", f"Unable to display Volatility Cone: {exc}", level="error")

    with tabs[5]:
        try:
            classical_diagnostics.render(current_data)
        except Exception as exc:
            render_notice_card("Classical Diagnostics Notice", f"Unable to display Classical Diagnostics: {exc}", level="error")

    with tabs[6]:
        try:
            trello_page.render(current_data)
        except Exception as exc:
            render_notice_card("Trello Tracker Notice", f"Unable to display Trello Tracker: {exc}", level="error")

    with tabs[7]:
        try:
            report.render(current_data)
        except Exception as exc:
            render_notice_card("Report Display Notice", f"Unable to display Report: {exc}", level="error")

    with tabs[8]:
        try:
            methodology.render(current_data)
        except Exception as exc:
            render_notice_card("Methodology Display Notice", f"Unable to display Methodology: {exc}", level="error")

else:
    # Sleek Standby Welcome Container
    st.markdown(
        f"""
        <div class="mf-standby">
            <div style="font-size: 1.5rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em; margin-bottom: 0.5rem;">
                MarketFolio Terminal Standby
            </div>
            <div style="font-size: 0.88rem; color: #94a3b8; max-width: 580px; margin: 0 auto 1.5rem auto; line-height: 1.6;">
                Institutional risk analytics, realized volatility cones, and macro transmission monitor for Indian equities and mutual funds.
            </div>
            <div style="display: inline-flex; gap: 1.5rem; font-size: 0.82rem; color: #64748b; background-color: #111827; padding: 0.75rem 1.25rem; border-radius: 6px; border: 1px solid #1e293b; margin-bottom: 1.5rem;">
                <div><span style="color:#94a3b8;">Asset:</span> <b style="color:#f8fafc;">{asset_input}</b></div>
                <div><span style="color:#94a3b8;">Benchmark:</span> <b style="color:#f8fafc;">{benchmark_symbol}</b></div>
                <div><span style="color:#94a3b8;">Horizon:</span> <b style="color:#f8fafc;">{horizon} Days</b></div>
                <div><span style="color:#94a3b8;">Backtest:</span> <b style="color:#f8fafc;">{max_origins} Origins</b></div>
            </div>
            <div>
                <span style="font-size: 0.84rem; color: #94a3b8;">Click <b>Run Analysis</b> in the sidebar to calibrate models.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
