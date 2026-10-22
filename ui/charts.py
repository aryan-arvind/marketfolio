"""
MarketFolio Plotly Chart Factory.
Constructs high-density, interactive financial charts with dark styling.
Zero matplotlib in main views; range selectors on price charts; no big titles.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from statsmodels.tsa.stattools import acf, pacf

from ui.theme import (
    ACCENT_TEAL,
    BORDER_COLOR,
    BORDER_HIGHLIGHT,
    COLOR_GAIN,
    COLOR_LOSS,
    COLOR_WARNING,
    GRID_COLOR,
    TEXT_DIM,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


def build_price_chart(price_series: pd.Series, display_name: str) -> go.Figure:
    """Builds interactive price/NAV chart with range selector buttons."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_series.index,
            y=price_series.values,
            mode="lines",
            name=display_name,
            line=dict(color=ACCENT_TEAL, width=1.75),
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: %{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        height=380,
        margin=dict(l=40, r=20, t=10, b=20),
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]),
                bgcolor="#111827",
                activecolor="#1e293b",
                font=dict(color=TEXT_MUTED, size=10),
                bordercolor=BORDER_COLOR,
                borderwidth=1,
            ),
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis=dict(
            side="right",
            tickformat=",.2f",
        ),
        showlegend=False,
    )
    return fig


def build_stl_decomposition_chart(stl_res: Any, display_name: str) -> go.Figure:
    """Builds a 4-panel synced time series showing Price, Trend, Seasonal, and Residual."""
    trend = stl_res.trend
    seasonal = stl_res.seasonal
    residual = stl_res.residual

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Trend Component", "Seasonal Cycle (21-Session)", "Residual / Irregular"),
    )

    fig.add_trace(
        go.Scatter(
            x=trend.index,
            y=trend.values,
            mode="lines",
            name="Trend",
            line=dict(color=ACCENT_TEAL, width=1.5),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=seasonal.index,
            y=seasonal.values,
            mode="lines",
            name="Seasonal",
            line=dict(color="#38bdf8", width=1.2),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=residual.index,
            y=residual.values,
            mode="lines",
            name="Residual",
            line=dict(color=TEXT_DIM, width=1.0),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.4f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=480,
        margin=dict(l=40, r=20, t=25, b=20),
        showlegend=False,
    )
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=11, color=TEXT_MUTED)
    return fig


def build_calendar_bars(cal_res: Any) -> tuple[go.Figure, go.Figure]:
    """Builds weekday and monthly return distribution bar charts with error intervals."""
    # Weekday figure
    dow_df = cal_res.dow_stats.copy()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    dow_df["order"] = dow_df["day_name"].apply(lambda x: dow_order.index(x) if x in dow_order else 99)
    dow_df = dow_df.sort_values("order")

    colors_dow = [COLOR_GAIN if m >= 0 else COLOR_LOSS for m in dow_df["mean"]]
    fig_dow = go.Figure()
    fig_dow.add_trace(
        go.Bar(
            x=dow_df["day_name"],
            y=dow_df["mean"] * 100.0,
            marker_color=colors_dow,
            error_y=dict(
                type="data",
                array=(dow_df["ci_upper"] - dow_df["mean"]) * 100.0,
                arrayminus=(dow_df["mean"] - dow_df["ci_lower"]) * 100.0,
                color=TEXT_DIM,
                thickness=1.2,
                width=4,
            ),
            hovertemplate="<b>%{x}</b><br>Mean Daily: %{y:+.3f}%<br>Count: %{customdata:,}<extra></extra>",
            customdata=dow_df["count"],
        )
    )
    fig_dow.update_layout(
        height=260,
        margin=dict(l=35, r=15, t=15, b=25),
        yaxis=dict(ticksuffix="%", zerolinecolor=BORDER_HIGHLIGHT),
        xaxis=dict(tickfont=dict(size=11)),
    )

    # Monthly figure
    m_df = cal_res.month_stats.copy()
    colors_m = [COLOR_GAIN if m >= 0 else COLOR_LOSS for m in m_df["mean"]]
    fig_m = go.Figure()
    fig_m.add_trace(
        go.Bar(
            x=m_df["month_name"],
            y=m_df["mean"] * 100.0,
            marker_color=colors_m,
            error_y=dict(
                type="data",
                array=(m_df["ci_upper"] - m_df["mean"]) * 100.0,
                arrayminus=(m_df["mean"] - m_df["ci_lower"]) * 100.0,
                color=TEXT_DIM,
                thickness=1.2,
                width=3,
            ),
            hovertemplate="<b>%{x}</b><br>Mean Daily: %{y:+.3f}%<br>Count: %{customdata:,}<extra></extra>",
            customdata=m_df["count"],
        )
    )
    fig_m.update_layout(
        height=260,
        margin=dict(l=35, r=15, t=15, b=25),
        yaxis=dict(ticksuffix="%", zerolinecolor=BORDER_HIGHLIGHT),
        xaxis=dict(tickfont=dict(size=11)),
    )

    return fig_dow, fig_m


def build_lag1_scatter_chart(returns: pd.Series) -> tuple[go.Figure, str]:
    """
    Builds lag-1 return scatter plot with trimmed axes (0.5th to 99.5th percentile)
    to prevent single corporate action / split outliers from distorting the view.
    """
    clean_rets = returns.dropna()
    r_t = clean_rets.iloc[1:].values * 100.0
    r_t_minus_1 = clean_rets.iloc[:-1].values * 100.0

    # Percentile clipping bounds (0.5% to 99.5%)
    p_low = float(np.percentile(clean_rets * 100.0, 0.5))
    p_high = float(np.percentile(clean_rets * 100.0, 99.5))

    # Identify trimmed points count
    outside_mask = (r_t < p_low) | (r_t > p_high) | (r_t_minus_1 < p_low) | (r_t_minus_1 > p_high)
    n_trimmed = int(outside_mask.sum())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=r_t_minus_1,
            y=r_t,
            mode="markers",
            marker=dict(
                size=4,
                color=ACCENT_TEAL,
                opacity=0.45,
                line=dict(width=0),
            ),
            hovertemplate="Lag 1 (t-1): %{x:.2f}%<br>Return (t): %{y:.2f}%<extra></extra>",
        )
    )

    # OLS fitted line on untrimmed data
    slope, intercept = np.polyfit(r_t_minus_1, r_t, 1)
    x_line = np.linspace(p_low, p_high, 50)
    y_line = slope * x_line + intercept

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color=COLOR_WARNING, width=1.5, dash="dash"),
            name=f"Fitted Slope (rho={slope:.3f})",
        )
    )

    pad = (p_high - p_low) * 0.08
    fig.update_layout(
        height=320,
        margin=dict(l=35, r=15, t=15, b=25),
        xaxis=dict(
            range=[p_low - pad, p_high + pad],
            title="Return t-1 (%)",
            ticksuffix="%",
        ),
        yaxis=dict(
            range=[p_low - pad, p_high + pad],
            title="Return t (%)",
            ticksuffix="%",
        ),
        showlegend=False,
    )

    note = f"Axes trimmed to 0.5th–99.5th percentile ({n_trimmed} outlier session(s) outside display box)."
    return fig, note


def build_returns_distribution_chart(returns: pd.Series) -> go.Figure:
    """Builds return histogram with parametric normal curve overlay."""
    clean_rets = returns.dropna().values * 100.0
    mu = float(np.mean(clean_rets))
    sigma = float(np.std(clean_rets))

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=clean_rets,
            histnorm="probability density",
            nbinsx=60,
            marker_color="rgba(6, 182, 212, 0.4)",
            marker_line=dict(color=ACCENT_TEAL, width=0.8),
            name="Empirical Density",
            hovertemplate="Return: %{x:.2f}%<br>Density: %{y:.4f}<extra></extra>",
        )
    )

    # Normal distribution overlay
    x_grid = np.linspace(min(clean_rets), max(clean_rets), 200)
    pdf = stats.norm.pdf(x_grid, mu, sigma)
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=pdf,
            mode="lines",
            line=dict(color="#f8fafc", width=1.5, dash="dot"),
            name="Normal Fit",
            hovertemplate="Normal Fit: %{y:.4f}<extra></extra>",
        )
    )

    # Limit display to 3.5 sigmas around mean
    fig.update_layout(
        height=320,
        margin=dict(l=35, r=15, t=15, b=25),
        xaxis=dict(
            range=[mu - 3.8 * sigma, mu + 3.8 * sigma],
            ticksuffix="%",
            title="Daily Log Return (%)",
        ),
        yaxis=dict(title="Density"),
        legend=dict(x=0.02, y=0.95),
    )
    return fig


def build_cumulative_returns_chart(cum_returns: pd.Series, display_name: str) -> go.Figure:
    """Builds cumulative return trajectory from session 0."""
    cum_pct = cum_returns * 100.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cum_pct.index,
            y=cum_pct.values,
            mode="lines",
            line=dict(color=ACCENT_TEAL, width=1.5),
            hovertemplate="%{x|%Y-%m-%d}: %{y:+.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=35, r=15, t=15, b=25),
        yaxis=dict(ticksuffix="%", zeroline=True, zerolinecolor=BORDER_HIGHLIGHT),
        showlegend=False,
    )
    return fig


def build_acf_pacf_tidy_chart(returns: pd.Series, nlags: int = 24) -> go.Figure:
    """Combines ACF and PACF into a single clean two-panel figure for Details."""
    clean_rets = returns.dropna()
    acf_vals = acf(clean_rets, nlags=nlags)
    pacf_vals = pacf(clean_rets, nlags=nlags)
    lags = list(range(nlags + 1))
    crit_bound = 1.96 / np.sqrt(len(clean_rets))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"),
        horizontal_spacing=0.08,
    )

    # ACF trace
    fig.add_trace(
        go.Bar(
            x=lags[1:],
            y=acf_vals[1:],
            marker_color=ACCENT_TEAL,
            name="ACF",
            hovertemplate="Lag %{x}: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    # 95% Bartlett bounds on ACF
    fig.add_hline(y=crit_bound, line=dict(color=TEXT_DIM, dash="dash", width=1), row=1, col=1)
    fig.add_hline(y=-crit_bound, line=dict(color=TEXT_DIM, dash="dash", width=1), row=1, col=1)

    # PACF trace
    fig.add_trace(
        go.Bar(
            x=lags[1:],
            y=pacf_vals[1:],
            marker_color="#38bdf8",
            name="PACF",
            hovertemplate="Lag %{x}: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=crit_bound, line=dict(color=TEXT_DIM, dash="dash", width=1), row=1, col=2)
    fig.add_hline(y=-crit_bound, line=dict(color=TEXT_DIM, dash="dash", width=1), row=1, col=2)

    fig.update_layout(
        height=260,
        margin=dict(l=35, r=15, t=25, b=25),
        showlegend=False,
    )
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=11, color=TEXT_MUTED)
    return fig


def build_volatility_cone_chart(vol_res: Any, horizon: int) -> go.Figure:
    """Builds realized volatility forecast cone with bootstrap 90% confidence envelope."""
    hist_vol = vol_res.realized_vol_series.dropna().iloc[-90:]  # Last 90 sessions for clear view
    last_date = hist_vol.index[-1]
    last_vol = float(hist_vol.iloc[-1])

    # Future dates (business days)
    future_dates = pd.bdate_range(start=last_date, periods=horizon + 1)[1:]

    # Forecast series
    pt_pred = np.insert(vol_res.analytic_point_pred, 0, last_vol)
    boot_med = np.insert(vol_res.boot_median, 0, last_vol)
    boot_lo = np.insert(vol_res.boot_lower_90, 0, last_vol)
    boot_hi = np.insert(vol_res.boot_upper_90, 0, last_vol)
    all_future_dates = pd.date_range(start=last_date, periods=horizon + 1)

    fig = go.Figure()

    # Historical realized volatility line
    fig.add_trace(
        go.Scatter(
            x=hist_vol.index,
            y=hist_vol.values,
            mode="lines",
            line=dict(color=TEXT_SECONDARY, width=1.5),
            name="Historical 21D Vol",
            hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}% p.a.<extra></extra>",
        )
    )

    # 90% Bootstrap envelope (shaded band)
    fig.add_trace(
        go.Scatter(
            x=all_future_dates,
            y=boot_hi,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=all_future_dates,
            y=boot_lo,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(6, 182, 212, 0.15)",
            name="90% Risk Cone (B=2000)",
            hovertemplate="Lower 90%: %{y:.1f}%<extra></extra>",
        )
    )

    # Bootstrap median forecast
    fig.add_trace(
        go.Scatter(
            x=all_future_dates,
            y=boot_med,
            mode="lines",
            line=dict(color=ACCENT_TEAL, width=2.0, dash="solid"),
            name="Forecast Median",
            hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}% p.a.<extra></extra>",
        )
    )

    # Naive baseline horizontal line
    fig.add_trace(
        go.Scatter(
            x=all_future_dates,
            y=[vol_res.naive_forecast_pct] * len(all_future_dates),
            mode="lines",
            line=dict(color=TEXT_DIM, width=1.2, dash="dot"),
            name="Naive Baseline",
            hovertemplate="Naive: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        height=360,
        margin=dict(l=35, r=15, t=15, b=25),
        yaxis=dict(ticksuffix="% p.a.", side="right"),
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def build_drawdown_chart(drawdowns: pd.Series) -> go.Figure:
    """Builds underwater drawdown chart."""
    dd_pct = drawdowns * 100.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd_pct.index,
            y=dd_pct.values,
            mode="lines",
            line=dict(color=COLOR_LOSS, width=1.2),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.15)",
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=35, r=15, t=15, b=25),
        yaxis=dict(ticksuffix="%", zeroline=True, zerolinecolor=BORDER_HIGHLIGHT),
        showlegend=False,
    )
    return fig


def build_market_scatter_chart(asset_rets: pd.Series, bench_rets: pd.Series, mkt_res: Any) -> go.Figure:
    """Builds return scatter against benchmark with fitted regression slope."""
    # Align dates
    df = pd.DataFrame({"asset": asset_rets, "bench": bench_rets}).dropna() * 100.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["bench"],
            y=df["asset"],
            mode="markers",
            marker=dict(
                size=4,
                color=ACCENT_TEAL,
                opacity=0.45,
            ),
            hovertemplate="Benchmark: %{x:+.2f}%<br>Asset: %{y:+.2f}%<extra></extra>",
            name="Daily Observations",
        )
    )

    # Fitted SARIMAX line
    x_vals = np.linspace(df["bench"].min(), df["bench"].max(), 50)
    beta = mkt_res.arma_beta
    alpha_daily_pct = mkt_res.arma_alpha_daily * 100.0
    y_vals = alpha_daily_pct + beta * x_vals

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            line=dict(color="#f8fafc", width=1.75),
            name=f"Fitted Beta ({beta:.2f}x)",
        )
    )

    fig.update_layout(
        height=340,
        margin=dict(l=35, r=15, t=15, b=25),
        xaxis=dict(title=f"{mkt_res.benchmark_name} Return (%)", ticksuffix="%"),
        yaxis=dict(title=f"{mkt_res.asset_name} Return (%)", ticksuffix="%"),
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def build_fevd_stacked_chart(fevd_df: pd.DataFrame) -> go.Figure:
    """Builds stacked 10-step Forecast Error Variance Decomposition area chart."""
    fig = go.Figure()
    palette = [ACCENT_TEAL, "#38bdf8", COLOR_WARNING, "#a855f7", "#ec4899", COLOR_GAIN]

    for idx, col in enumerate(fevd_df.columns):
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=fevd_df.index,
                y=fevd_df[col],
                mode="lines",
                stackgroup="one",
                name=col,
                line=dict(width=0.5, color=color),
                hovertemplate="Step %{x}: %{y:.1f}%<extra>" + col + "</extra>",
            )
        )

    fig.update_layout(
        height=300,
        margin=dict(l=35, r=15, t=15, b=25),
        yaxis=dict(title="Variance Contribution (%)", ticksuffix="%", range=[0, 100]),
        xaxis=dict(title="Forecast Step Ahead (Sessions)"),
        legend=dict(orientation="h", y=1.15, x=0),
    )
    return fig
