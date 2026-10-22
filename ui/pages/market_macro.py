"""
MarketFolio Tab 5: Market Sensitivity & Global Macro Linkages.
Single-Index regression with ARMA errors (SARIMAX) and 5-variable Vector Autoregression (VAR).
"""

from __future__ import annotations

from typing import Any
import pandas as pd
import streamlit as st

from ui.charts import build_fevd_stacked_chart, build_market_scatter_chart
from ui.components import (
    render_kpi_card,
    render_method_tag,
    render_notice_card,
    render_takeaway_line,
)
from ui.readings import get_macro_drivers_reading, get_market_sensitivity_reading
from ui.theme import ACCENT_TEAL, COLOR_GAIN, COLOR_LOSS, COLOR_WARNING, TEXT_MUTED


def render(data: dict[str, Any]) -> None:
    """Renders the Market & Macro tab."""
    render_method_tag("Single-Index SARIMAX (HAC & Autocorrelated Errors) & 5-Variable Vector Autoregression")

    mkt = data.get("market_res")
    var_res = data.get("var_res")
    returns = data["clean_res"].log_return_series
    bench_returns = data.get("bench_returns")
    bench_name = mkt.benchmark_name if mkt else "^NSEI"

    # SECTION 1: SYSTEMATIC MARKET SENSITIVITY
    st.markdown('<div style="font-size:0.95rem; font-weight:700; margin-bottom:0.4rem;">Market Sensitivity (Beta & Alpha)</div>', unsafe_allow_html=True)

    if mkt and bench_returns is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi_card(
                title=f"Market Beta vs {mkt.benchmark_name}",
                value=f"{mkt.arma_beta:.2f}x",
                subtext=f"95% CI: [{mkt.arma_beta_ci[0]:.2f}, {mkt.arma_beta_ci[1]:.2f}]",
                value_color=ACCENT_TEAL,
            )
        with c2:
            alpha_pa = mkt.arma_alpha_annualized * 100.0
            alpha_color = COLOR_GAIN if alpha_pa >= 0 else COLOR_LOSS
            render_kpi_card(
                title="Annualized Alpha",
                value=f"{alpha_pa:+.2f}%",
                subtext=f"p={mkt.arma_alpha_pvalue:.4f} (SARIMAX)",
                value_color=alpha_color,
            )
        with c3:
            render_kpi_card(
                title="R-Squared Fit",
                value=f"{mkt.ols_r_squared * 100:.1f}%",
                subtext="Variance explained by benchmark",
            )
        with c4:
            render_kpi_card(
                title="Residual Durbin-Watson",
                value=f"{mkt.ols_dw_stat:.2f}",
                subtext=f"Disturbance ARIMA{mkt.best_arma_order}",
                value_color=COLOR_WARNING if abs(mkt.ols_dw_stat - 2.0) > 0.4 else TEXT_MUTED,
            )

        col_scatter, col_desc = st.columns([1.6, 1.2])
        with col_scatter:
            fig_scatter = build_market_scatter_chart(returns, bench_returns, mkt)
            st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

        with col_desc:
            st.markdown(
                f"""
                <div class="mf-card" style="height: 340px; overflow-y: auto;">
                    <div class="mf-card-title">Empirical Sensitivity Finding</div>
                    <div style="font-size: 0.86rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 1rem;">
                        {mkt.interpretation}
                    </div>
                    <div style="font-size: 0.78rem; color: {TEXT_MUTED}; line-height: 1.6;">
                        <b>Aligned Trading Sessions</b>: {mkt.n_aligned_obs:,}<br>
                        <b>Observation Window</b>: {mkt.start_date} to {mkt.end_date}<br>
                        <b>HAC Standard Error</b>: beta SE = {mkt.arma_beta_se:.3f}<br>
                        <b>Residual Autocorrelation</b>: Modeled via SARIMAX with ARMA{mkt.best_arma_order} disturbances to prevent spurious t-ratios.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        render_notice_card(
            title="Market Sensitivity Unavailable",
            message=f"Could not calibrate benchmark regression against {bench_name}. Historical dates may not overlap.",
            level="warning",
        )

    render_takeaway_line(f"<b>Desk Takeaway</b>: {get_market_sensitivity_reading(mkt, bench_name)}")

    st.markdown("---")

    # SECTION 2: GLOBAL MACRO TRANSMISSION (VAR)
    st.markdown('<div style="font-size:0.95rem; font-weight:700; margin-bottom:0.4rem;">Global Macro Transmission (5-Variable Vector Autoregression)</div>', unsafe_allow_html=True)

    if var_res:
        st.markdown(
            f"""
            <div style="font-size:0.8rem; color:{TEXT_MUTED}; margin-bottom:0.75rem;">
                Evaluates transmission across <b>S&P 500, Brent Crude, USD/INR, India VIX</b> and <b>NIFTY 50</b> &nbsp;|&nbsp;
                Synchronized Sample: <b>{var_res.aligned_obs}</b> sessions &nbsp;|&nbsp;
                Selected Lag Order: <b>p={var_res.selected_lag}</b> &nbsp;|&nbsp;
                System Dynamic Stability: <b>{'Stable' if var_res.is_stable else 'Unstable'}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Macro Granger Table with strength badges
        granger_rows = []
        for g in var_res.granger_tests:
            if g.p_value < 0.01:
                strength_badge = '<span class="mf-badge mf-badge-live">Strong Lead (p < 0.01)</span>'
                leads_status = "Leads Nifty"
            elif g.p_value < 0.05:
                strength_badge = '<span class="mf-badge mf-badge-teal">Moderate Lead (p < 0.05)</span>'
                leads_status = "Leads Nifty"
            else:
                strength_badge = '<span class="mf-badge mf-badge-neutral">Independent</span>'
                leads_status = "No Lead"

            granger_rows.append({
                "Macro Transmission Factor": g.causing_variable,
                "Lead Status": leads_status,
                "Predictive Signal Strength": strength_badge,
                "Desk Transmission Finding": g.conclusion,
            })

        granger_df = pd.DataFrame(granger_rows)
        # Render clean HTML table for badges
        st.markdown(
            granger_df.to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )

        st.markdown('<div style="font-size:0.88rem; font-weight:600; margin-top:1.25rem; margin-bottom:0.25rem;">NIFTY 50 Forecast Error Variance Decomposition (FEVD, 10-Step Ahead)</div>', unsafe_allow_html=True)
        fig_fevd = build_fevd_stacked_chart(var_res.fevd_nifty_horizon10)
        st.plotly_chart(fig_fevd, use_container_width=True, config={"displayModeBar": False})

        render_takeaway_line(f"<b>Macro Transmission Finding</b>: {get_macro_drivers_reading(var_res)}")
    else:
        render_notice_card(
            title="Macro Transmission System Inactive",
            message="Vector Autoregression was skipped due to insufficient overlapping trading days across public macro data feeds.",
            level="info",
        )

    # Details Expander: Technical OLS vs SARIMAX & Companion Matrix Roots
    with st.expander("Technical Model Comparison & VAR Companion Matrix Diagnostics"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("##### OLS (HAC) vs SARIMAX ARMA Errors")
            if mkt:
                comp_df = pd.DataFrame([
                    {
                        "Estimator": "HAC OLS (Newey-West)",
                        "Alpha (p.a.)": f"{mkt.ols_alpha_annualized*100:+.2f}% (p={mkt.ols_alpha_pvalue:.3f})",
                        "Beta (Slope)": f"{mkt.ols_beta:.3f} (SE: {mkt.ols_beta_se_hac:.3f})",
                        "R2 / DW": f"{mkt.ols_r_squared:.3f} / {mkt.ols_dw_stat:.2f}",
                    },
                    {
                        "Estimator": f"SARIMAX + ARMA{mkt.best_arma_order}",
                        "Alpha (p.a.)": f"{mkt.arma_alpha_annualized*100:+.2f}% (p={mkt.arma_alpha_pvalue:.3f})",
                        "Beta (Slope)": f"{mkt.arma_beta:.3f} (SE: {mkt.arma_beta_se:.3f})",
                        "AICc": f"{mkt.arma_aicc:.1f}",
                    },
                ])
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
        with col_m2:
            st.markdown("##### VAR Stability & Stationarity Checks")
            if var_res:
                st.dataframe(var_res.stationarity_table, use_container_width=True, hide_index=True)
                st.caption(f"Cholesky Recursive Identification Ordering: {' > '.join(var_res.cholesky_order)}")
