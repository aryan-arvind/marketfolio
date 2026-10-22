"""
MarketFolio Tab 4: Volatility & Risk Modeling.
Realized volatility cone, empirical residual bootstrap (B=2000), drawdowns, and risk envelope.
"""

from __future__ import annotations

from typing import Any
import pandas as pd
import streamlit as st

from ui.charts import build_drawdown_chart, build_volatility_cone_chart
from ui.components import render_kpi_card, render_method_tag, render_takeaway_line
from ui.readings import get_volatility_regime
from ui.state import calculate_derived_stats
from ui.theme import ACCENT_TEAL, COLOR_GAIN, COLOR_LOSS, COLOR_WARNING, TEXT_MUTED


def render(data: dict[str, Any]) -> None:
    """Renders the Volatility & Risk tab."""
    vol_res = data["vol_res"]
    prices = data["clean_res"].price_series
    returns = data["clean_res"].log_return_series
    horizon = vol_res.horizon_days

    render_method_tag(f"Realized Volatility ARIMA{vol_res.best_order} & Residual Bootstrap (B=2,000, {horizon}D Horizon)")

    stats_dict = calculate_derived_stats(prices, returns)

    # Top KPI Row: Current Vol, {horizon}D Forecast, Risk Range, Max Drawdown
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            title="Current Realized Vol (21D)",
            value=f"{vol_res.last_observed_vol_pct:.1f}%",
            subtext="Annualized rolling standard deviation",
        )
    with c2:
        render_kpi_card(
            title=f"{horizon}D Median Vol Forecast",
            value=f"{vol_res.point_forecast_pct:.1f}%",
            subtext=f"ARIMA{vol_res.best_order} median projection",
            value_color=ACCENT_TEAL,
        )
    with c3:
        lo, hi = vol_res.expected_risk_range_pct
        render_kpi_card(
            title=f"{horizon}D Expected Risk Range",
            value=f"{lo:.0f}%–{hi:.0f}%",
            subtext="90% Confidence envelope (B=2000)",
        )
    with c4:
        dd = stats_dict["max_drawdown_pct"]
        render_kpi_card(
            title="Max Sample Drawdown",
            value=f"{dd:.1f}%",
            subtext="Peak-to-trough decline",
            value_color=COLOR_LOSS,
        )

    # Main Chart: Volatility Cone with Bootstrap Envelope
    st.markdown(
        f'<div style="font-size:0.88rem; font-weight:600; margin-top:0.5rem; margin-bottom:0.25rem;">'
        f'{horizon}-Day Realized Volatility Forecast Cone & 90% Empirical Envelope</div>',
        unsafe_allow_html=True,
    )
    fig_vol = build_volatility_cone_chart(vol_res, horizon)
    st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})

    # Secondary Chart: Underwater Drawdown Profile
    st.markdown(
        '<div style="font-size:0.88rem; font-weight:600; margin-top:0.75rem; margin-bottom:0.25rem;">'
        'Historical Drawdown Profile (%)</div>',
        unsafe_allow_html=True,
    )
    fig_dd = build_drawdown_chart(stats_dict["drawdowns"])
    st.plotly_chart(fig_dd, use_container_width=True, config={"displayModeBar": False})

    # Summary Risk Comparison Table
    st.markdown(
        f'<div style="font-size:0.88rem; font-weight:600; margin-top:0.75rem; margin-bottom:0.25rem;">'
        f'{horizon}-Day Horizon Volatility Benchmarking Table</div>',
        unsafe_allow_html=True,
    )
    lo, hi = vol_res.expected_risk_range_pct
    risk_df = pd.DataFrame([
        {
            "Forecast Benchmark": f"ARIMA{vol_res.best_order} Median Point Projection",
            "Annualized Value": f"{vol_res.point_forecast_pct:.2f}% p.a.",
            "Description": "Primary econometric model projection",
        },
        {
            "Forecast Benchmark": f"Empirical {horizon}D Risk Envelope (90% CI)",
            "Annualized Value": f"[{lo:.2f}% – {hi:.2f}%] p.a.",
            "Description": "5th to 95th percentile bounds from 2,000 bootstrap simulations",
        },
        {
            "Forecast Benchmark": "Naive Constant-Volatility Baseline",
            "Annualized Value": f"{vol_res.naive_forecast_pct:.2f}% p.a.",
            "Description": "Random-walk assumption (last observed rolling volatility persists)",
        },
        {
            "Forecast Benchmark": "21-Day Moving Average Baseline",
            "Annualized Value": f"{vol_res.rolling_mean_forecast_pct:.2f}% p.a.",
            "Description": "Historical moving average heuristic",
        },
    ])
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

    status_chip, _, vol_reading = get_volatility_regime(vol_res, horizon)
    render_takeaway_line(f"<b>Desk Takeaway</b>: {vol_reading}")

    # Details Expander: Model fit parameters and bootstrap fallback status
    with st.expander("Technical Model Specification & Bootstrap Verification"):
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown(
                f"""
                <div style="font-size:0.82rem; color:{TEXT_MUTED}; line-height:1.7;">
                    <b>Modeled Target</b>: Log Realized Volatility: ln(sigma_t)<br>
                    <b>Integration Order (d)</b>: {vol_res.d_order} (Selected via ADF/KPSS)<br>
                    <b>Optimal Specification</b>: ARIMA{vol_res.best_order}<br>
                    <b>Replications (B)</b>: 2,000 recursive residual bootstrap paths<br>
                    <b>Bootstrap Fallback Engaged</b>: {'Yes' if vol_res.boot_failed_or_fallback else 'No (Direct Empirical Simulation)'}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_b:
            if vol_res.boot_failed_or_fallback:
                st.caption(f"Fallback Reason: {vol_res.fallback_reason}")
            else:
                st.markdown(
                    '<span class="mf-badge mf-badge-live">Bootstrap Simulation: Fully Converged</span>',
                    unsafe_allow_html=True,
                )
