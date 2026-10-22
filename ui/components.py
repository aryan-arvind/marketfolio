"""
MarketFolio Reusable UI Components.
Header strip, KPI cards, badges, chips, takeaway callouts, and details wrappers.
Clean HTML/Streamlit components styled after institutional research terminals.
"""

from __future__ import annotations

from typing import Any
import streamlit as st

from ui.state import calculate_derived_stats
from ui.theme import (
    ACCENT_TEAL,
    BORDER_CARD,
    COLOR_GAIN,
    COLOR_LOSS,
    COLOR_WARNING,
    TEXT_DIM,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


def render_method_tag(technique_label: str) -> None:
    """Renders a subtle, non-intrusive gray method tag below page headers."""
    st.markdown(
        f'<div class="mf-method-tag">METHOD &bull; {technique_label}</div>',
        unsafe_allow_html=True,
    )


def render_takeaway_line(text: str) -> None:
    """Renders a single-line takeaway bar with teal accent border."""
    st.markdown(
        f'<div class="mf-takeaway">{text}</div>',
        unsafe_allow_html=True,
    )


def render_kpi_card(title: str, value: str, subtext: str = "", value_color: str | None = None) -> None:
    """Renders an institutional KPI metric card."""
    color_style = f' style="color: {value_color};"' if value_color else ""
    st.markdown(
        f"""
        <div class="mf-card">
            <div class="mf-card-title">{title}</div>
            <div class="mf-card-val"{color_style}>{value}</div>
            <div class="mf-card-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header_strip(data: dict[str, Any]) -> None:
    """
    Renders the Stocktwits/terminal style persistent header strip:
    Ticker, Asset Name, Big Latest Close/NAV, 1D % Change, Badges, and Thin Stats Strip.
    """
    meta = data["meta"]
    prices = data["clean_res"].price_series
    returns = data["clean_res"].log_return_series
    stats_dict = calculate_derived_stats(prices, returns)

    is_mf = meta.get("asset_type") == "mutual_fund" or meta.get("source") == "AMFI"
    price_label = "NAV" if is_mf else "Close"

    latest_price = stats_dict["latest_price"]
    price_chg = stats_dict["price_change_1d"]
    pct_chg = stats_dict["pct_change_1d"]
    is_positive = pct_chg >= 0

    chg_sign = "+" if is_positive else ""
    chg_class = "mf-change-pos" if is_positive else "mf-change-neg"

    is_stale = meta.get("is_stale", False)
    status_badge = (
        '<span class="mf-badge mf-badge-stale">Stale (Offline Cache)</span>'
        if is_stale
        else '<span class="mf-badge mf-badge-live">Live Feed</span>'
    )

    fetch_ts = meta.get("fetch_timestamp", "")
    fetch_info = f"&nbsp;&bull;&nbsp; Updated: {fetch_ts}" if fetch_ts else ""

    first_date = str(prices.index.min().date())
    last_date = meta.get("last_date", str(prices.index.max().date()))
    n_sessions = len(returns)

    st.markdown(
        f"""
        <div class="mf-header-strip">
            <div class="mf-header-left">
                <span class="mf-ticker">{data['symbol_code']}</span>
                <span class="mf-name">{data['display_name']}</span>
                {status_badge}
            </div>
            <div class="mf-header-price-block">
                <span style="font-size:0.8rem; color:{TEXT_DIM}; text-transform:uppercase;">{price_label}</span>
                <span class="mf-price">{latest_price:,.2f}</span>
                <span class="mf-change {chg_class}">
                    {chg_sign}{price_chg:,.2f} ({chg_sign}{pct_chg:.2f}%)
                </span>
            </div>
            <div class="mf-stats-strip">
                <div class="mf-stat-item">
                    <span class="mf-stat-label">Range:</span>
                    <span class="mf-stat-value">{first_date} &rarr; {last_date}</span>
                </div>
                <div class="mf-stat-item">
                    <span class="mf-stat-label">Sessions:</span>
                    <span class="mf-stat-value">{n_sessions:,}</span>
                </div>
                <div class="mf-stat-item">
                    <span class="mf-stat-label">52W Range:</span>
                    <span class="mf-stat-value">{stats_dict['low_52w']:,.2f} &ndash; {stats_dict['high_52w']:,.2f}</span>
                </div>
                <div class="mf-stat-item">
                    <span class="mf-stat-label">Source:</span>
                    <span class="mf-stat-value">{meta.get('source', 'Public API')}{fetch_info}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_notice_card(title: str, message: str, level: str = "info") -> None:
    """Renders a cleanly styled alert card."""
    border_color = COLOR_WARNING if level == "warning" else (COLOR_LOSS if level == "error" else ACCENT_TEAL)
    st.markdown(
        f"""
        <div style="background-color: #111827; border: 1px solid {BORDER_CARD}; border-left: 3px solid {border_color};
                    border-radius: 6px; padding: 0.85rem 1.15rem; margin-bottom: 1rem;">
            <div style="font-size: 0.85rem; font-weight: 600; color: {TEXT_PRIMARY}; margin-bottom: 0.25rem;">
                {title}
            </div>
            <div style="font-size: 0.82rem; color: {TEXT_MUTED}; line-height: 1.45;">
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
