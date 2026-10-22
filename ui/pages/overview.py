"""
MarketFolio Tab 1: Overview & Executive Risk Summary.
Provides top-line KPIs, full price history with range selector, and desk key readings.
"""

from __future__ import annotations

from typing import Any
import streamlit as st

from ui.charts import build_price_chart
from ui.components import render_kpi_card, render_method_tag, render_takeaway_line
from ui.readings import get_overview_bullet_list, get_volatility_regime
from ui.state import calculate_derived_stats
from ui.theme import ACCENT_TEAL, COLOR_GAIN, COLOR_LOSS, TEXT_MUTED


def render(data: dict[str, Any]) -> None:
    """Renders the Overview tab."""
    render_method_tag("Integrated Econometric Risk Architecture · Daily Frequency")

    prices = data["clean_res"].price_series
    returns = data["clean_res"].log_return_series
    vol_res = data["vol_res"]
    mkt_res = data["market_res"]
    meta = data["meta"]
    horizon = vol_res.horizon_days

    stats_dict = calculate_derived_stats(prices, returns)

    # Top KPI Row (6 cards)
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    latest_price = stats_dict["latest_price"]
    is_mf = meta.get("asset_type") == "mutual_fund" or meta.get("source") == "AMFI"
    price_title = "Latest NAV" if is_mf else "Latest Close"

    with c1:
        render_kpi_card(
            title=price_title,
            value=f"{latest_price:,.2f}",
            subtext=f"As of {meta.get('last_date')}",
        )

    with c2:
        render_kpi_card(
            title="Realized Vol (21D)",
            value=f"{vol_res.last_observed_vol_pct:.1f}%",
            subtext="Annualized rolling std",
        )

    with c3:
        render_kpi_card(
            title=f"{horizon}D Vol Forecast",
            value=f"{vol_res.point_forecast_pct:.1f}%",
            subtext="ARIMA median projection",
            value_color=ACCENT_TEAL,
        )

    with c4:
        lo, hi = vol_res.expected_risk_range_pct
        render_kpi_card(
            title=f"{horizon}D Risk Range",
            value=f"{lo:.0f}%–{hi:.0f}%",
            subtext="90% Residual Bootstrap",
        )

    with c5:
        if mkt_res:
            render_kpi_card(
                title=f"Beta ({mkt_res.benchmark_name})",
                value=f"{mkt_res.arma_beta:.2f}",
                subtext=f"HAC SE: {mkt_res.arma_beta_se:.2f}",
            )
        else:
            render_kpi_card(title="Market Beta", value="N/A", subtext="Benchmark unaligned")

    with c6:
        if mkt_res:
            alpha_val = mkt_res.arma_alpha_annualized * 100.0
            alpha_color = COLOR_GAIN if alpha_val >= 0 else COLOR_LOSS
            render_kpi_card(
                title="Annualized Alpha",
                value=f"{alpha_val:+.2f}%",
                subtext=f"p={mkt_res.arma_alpha_pvalue:.3f}",
                value_color=alpha_color,
            )
        else:
            render_kpi_card(title="Alpha", value="N/A", subtext="Benchmark unaligned")

    # Main view: Left price chart, Right Key Readings card
    col_chart, col_readings = st.columns([1.75, 1.25])

    status_chip, chip_class, vol_reading = get_volatility_regime(vol_res, horizon)

    with col_chart:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.25rem;">
                <span style="font-size: 0.88rem; font-weight: 600; color: #f8fafc;">Historical Price Trajectory</span>
                <span class="mf-badge {chip_class}">{status_chip}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_price = build_price_chart(prices, data["display_name"])
        st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False})

    with col_readings:
        bullets = get_overview_bullet_list(data)
        bullets_html = "".join([f'<li style="margin-bottom:0.75rem; line-height:1.45;">{b}</li>' for b in bullets])
        st.markdown(
            f"""
            <div class="mf-card" style="height: 420px; overflow-y: auto;">
                <div class="mf-card-title">Key Readings & Risk Posture</div>
                <ul style="padding-left: 1.15rem; margin-top: 0.5rem; font-size: 0.84rem; color: #cbd5e1;">
                    {bullets_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_takeaway_line(
        f"<b>Desk Takeaway</b>: {vol_reading} "
        f"Out-of-sample backtest indicates {data['eval_res'].summary_verdict.lower()}."
    )

    # Details Expander
    with st.expander("Technical Summary & Data Provenance"):
        anomalies = data["clean_res"].flagged_anomalies
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown(
                f"""
                <div style="font-size:0.82rem; color:{TEXT_MUTED}; line-height:1.6;">
                    <b>Sample Window</b>: {meta.get('first_date')} to {meta.get('last_date')}<br>
                    <b>Raw Points</b>: {data['clean_res'].n_original_obs:,} &nbsp;|&nbsp;
                    <b>Cleaned Sessions</b>: {data['clean_res'].n_cleaned_obs:,}<br>
                    <b>Gaps Forward-Filled</b>: {data['clean_res'].n_gaps_filled} &nbsp;|&nbsp;
                    <b>Excess Gaps Dropped</b>: {data['clean_res'].n_dropped_after_gap}<br>
                    <b>Stationarity Verdict</b>: {data['stat_suite']['log_returns'].verdict} (p={data['stat_suite']['log_returns'].adf_pvalue:.4f})
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_b:
            if not anomalies.empty:
                st.markdown(
                    f'<span class="mf-badge mf-badge-stale">Data Quality Alert: {len(anomalies)} Session(s) |r| > 15%</span>',
                    unsafe_allow_html=True,
                )
                st.dataframe(anomalies, use_container_width=True, hide_index=True)
            else:
                st.markdown(
                    '<span class="mf-badge mf-badge-live">Data Quality: 0 Extreme Return Anomalies Flagged</span>',
                    unsafe_allow_html=True,
                )
