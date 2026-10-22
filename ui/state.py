"""
MarketFolio State & Derived Series Helpers.
Manages session_state caching and derived series calculations (drawdowns, 52w stats, etc).
Strictly adheres to plain transforms without introducing new inferential models.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st


def get_analysis_data() -> dict[str, Any] | None:
    """Retrieves the active analysis result dictionary from session_state."""
    return st.session_state.get("analysis_data")


def set_analysis_data(data: dict[str, Any], target_name: str) -> None:
    """Stores the analysis pipeline outputs into session_state."""
    st.session_state["analysis_data"] = data
    st.session_state["analyzed_target"] = target_name


def clear_analysis_data() -> None:
    """Clears cached analysis data."""
    if "analysis_data" in st.session_state:
        del st.session_state["analysis_data"]
    if "analyzed_target" in st.session_state:
        del st.session_state["analyzed_target"]


def calculate_derived_stats(price_series: pd.Series, log_return_series: pd.Series) -> dict[str, Any]:
    """
    Computes deterministic summary transforms:
    - 1D price delta and percentage return
    - 52-week (252-session) high and low
    - Cumulative return series and drawdown series
    - Skewness and excess kurtosis
    - Extreme session count (|r_t| > 3 sigma)
    """
    latest_price = float(price_series.iloc[-1])
    prev_price = float(price_series.iloc[-2]) if len(price_series) > 1 else latest_price
    price_change_1d = latest_price - prev_price
    pct_change_1d = ((latest_price / prev_price) - 1.0) * 100.0 if prev_price > 0 else 0.0

    # 52-week window (last 252 sessions or full series if shorter)
    window_52w = price_series.iloc[-252:] if len(price_series) >= 252 else price_series
    high_52w = float(window_52w.max())
    low_52w = float(window_52w.min())

    # Cumulative return: P_t / P_0 - 1
    cum_returns = (price_series / price_series.iloc[0]) - 1.0

    # Drawdown series: (P_t - running_max) / running_max
    running_max = price_series.cummax()
    drawdowns = (price_series - running_max) / running_max
    max_drawdown = float(drawdowns.min()) * 100.0

    # Return distribution stats
    clean_rets = log_return_series.dropna()
    skewness = float(stats.skew(clean_rets))
    kurtosis = float(stats.kurtosis(clean_rets))  # Fisher excess kurtosis

    # Extreme sessions (|return| > 3 * std)
    ret_std = float(clean_rets.std())
    ret_mean = float(clean_rets.mean())
    extreme_mask = (clean_rets - ret_mean).abs() > (3.0 * ret_std)
    n_extreme_sessions = int(extreme_mask.sum())

    return {
        "latest_price": latest_price,
        "prev_price": prev_price,
        "price_change_1d": price_change_1d,
        "pct_change_1d": pct_change_1d,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "cum_returns": cum_returns,
        "drawdowns": drawdowns,
        "max_drawdown_pct": max_drawdown,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "return_std": ret_std,
        "n_extreme_sessions": n_extreme_sessions,
    }
