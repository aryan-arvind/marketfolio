"""
Calendar seasonality, subseries analysis, day-of-week tests, and STL decomposition (ATSA Unit 1).
Evaluates calendar anomalies (Monday effect, month-of-year effect), lag-1 dependence,
and quantifies trend and seasonal strength using robust STL decomposition.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL

logger = logging.getLogger("risklens")


class CalendarEffectResult(NamedTuple):
    """Statistical summary of calendar effect hypothesis tests."""
    ols_f_stat: float
    ols_f_pvalue: float
    kruskal_stat: float
    kruskal_pvalue: float
    is_day_of_week_significant: bool
    summary_verdict: str
    dow_stats: pd.DataFrame
    month_stats: pd.DataFrame


class STLDecompositionResult(NamedTuple):
    """Metrics and components from STL decomposition of log price/NAV."""
    trend_strength_f_t: float
    seasonal_strength_f_s: float
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    verdict: str


def analyze_calendar_effects(log_returns: pd.Series, asset_name: str) -> CalendarEffectResult:
    """
    Evaluates day-of-week and monthly calendar effects on log returns.
    Executes:
    1. OLS regression on day-of-week indicator variables with joint F-test.
    2. Non-parametric Kruskal-Wallis test across weekday return distributions.
    Computes sample means, standard errors, and 95% confidence intervals per calendar bucket.
    """
    df = pd.DataFrame({"return": log_returns.copy()})
    df["day_name"] = df.index.day_name()
    df["day_of_week"] = df.index.dayofweek  # 0=Monday, 4=Friday
    df["month"] = df.index.month
    df["month_name"] = df.index.strftime("%b")

    # Filter to standard trading week (Mon-Fri)
    df = df[df["day_of_week"] < 5]

    # Day-of-week stats
    dow_summary = (
        df.groupby(["day_of_week", "day_name"])["return"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    dow_summary["se"] = dow_summary["std"] / np.sqrt(dow_summary["count"])
    dow_summary["ci_lower"] = dow_summary["mean"] - 1.96 * dow_summary["se"]
    dow_summary["ci_upper"] = dow_summary["mean"] + 1.96 * dow_summary["se"]
    dow_summary["mean_annualized_pct"] = dow_summary["mean"] * 252 * 100

    # Month stats
    month_summary = (
        df.groupby(["month", "month_name"])["return"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .sort_values(by="month")
    )
    month_summary["se"] = month_summary["std"] / np.sqrt(month_summary["count"])
    month_summary["ci_lower"] = month_summary["mean"] - 1.96 * month_summary["se"]
    month_summary["ci_upper"] = month_summary["mean"] + 1.96 * month_summary["se"]
    month_summary["mean_annualized_pct"] = month_summary["mean"] * 252 * 100

    # 1. OLS regression on Day-of-Week dummies
    # r_t = gamma_0 + sum_{d=1..4} gamma_d * 1(Day=d) + e_t
    # Joint F-test tests whether all day dummies are zero (i.e. mean returns equal across weekdays)
    dow_dummies = pd.get_dummies(df["day_of_week"], drop_first=True, dtype=float)
    X = sm.add_constant(dow_dummies)
    ols_model = sm.OLS(df["return"], X).fit()
    f_stat = float(ols_model.fvalue)
    f_pvalue = float(ols_model.f_pvalue)

    # 2. Robust non-parametric Kruskal-Wallis Test
    groups = [group["return"].values for _, group in df.groupby("day_of_week")]
    kruskal_res = stats.kruskal(*groups)
    k_stat = float(kruskal_res.statistic)
    k_pvalue = float(kruskal_res.pvalue)

    is_significant = (f_pvalue < 0.05) or (k_pvalue < 0.05)
    if is_significant:
        verdict = (
            f"Statistically significant day-of-week calendar anomaly detected (OLS F-test p={f_pvalue:.4f}, "
            f"Kruskal-Wallis p={k_pvalue:.4f}). Returns differ across trading days."
        )
    else:
        verdict = (
            f"No statistically significant day-of-week calendar effect (OLS F-test p={f_pvalue:.4f}, "
            f"Kruskal-Wallis p={k_pvalue:.4f}). The hypothesis of equal mean returns across weekdays cannot be rejected."
        )

    logger.info("Calendar effect analysis for '%s': %s", asset_name, verdict)

    return CalendarEffectResult(
        ols_f_stat=f_stat,
        ols_f_pvalue=f_pvalue,
        kruskal_stat=k_stat,
        kruskal_pvalue=k_pvalue,
        is_day_of_week_significant=is_significant,
        summary_verdict=verdict,
        dow_stats=dow_summary,
        month_stats=month_summary,
    )


def run_stl_decomposition(
    price_series: pd.Series,
    asset_name: str,
    period: int = 21,
) -> STLDecompositionResult:
    """
    Performs robust LOESS-based STL decomposition (Cleveland et al. 1990) on log price/NAV.
    Period defaults to 21 trading days (~1 month).
    Calculates Hyndman & Athanasopoulos (FPP3) feature measures:
    - Trend Strength: F_T = max(0, 1 - Var(R) / Var(T + R))
    - Seasonal Strength: F_S = max(0, 1 - Var(R) / Var(S + R))
    """
    log_price = np.log(price_series).dropna()
    dates = log_price.index

    # Use RangeIndex to avoid statsmodels frequency ambiguity warnings
    s_clean = pd.Series(log_price.values, index=pd.RangeIndex(len(log_price)))

    stl = STL(s_clean, period=period, robust=True)
    res = stl.fit()

    trend = pd.Series(res.trend.values, index=dates)
    seasonal = pd.Series(res.seasonal.values, index=dates)
    resid = pd.Series(res.resid.values, index=dates)

    var_r = float(np.var(resid, ddof=1))
    var_tr = float(np.var(trend + resid, ddof=1))
    var_sr = float(np.var(seasonal + resid, ddof=1))

    f_t = max(0.0, 1.0 - (var_r / var_tr)) if var_tr > 1e-12 else 0.0
    f_s = max(0.0, 1.0 - (var_r / var_sr)) if var_sr > 1e-12 else 0.0

    verdict = (
        f"STL decomposition of {asset_name} log price/NAV shows dominant trend strength "
        f"F_T = {f_t:.4f} and negligible/minor seasonal strength F_S = {f_s:.4f} "
        "(values near 1 indicate strong signal, near 0 indicate noise dominance)."
    )
    logger.info(verdict)

    return STLDecompositionResult(
        trend_strength_f_t=f_t,
        seasonal_strength_f_s=f_s,
        trend=trend,
        seasonal=seasonal,
        residual=resid,
        verdict=verdict,
    )


def plot_seasonality_and_subseries(
    log_returns: pd.Series,
    calendar_res: CalendarEffectResult,
    asset_name: str,
    output_path: str | None = None,
) -> plt.Figure:
    """
    Plots:
    1. Day-of-week subseries bar chart with 95% confidence intervals.
    2. Month-of-year subseries bar chart with 95% confidence intervals.
    3. Lag-1 scatter plot of returns r_t vs r_{t-1}.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Calendar Return Patterns & Lag Structure: {asset_name}",
        fontsize=13,
        fontweight="bold",
    )

    # 1. Day of week
    dow_df = calendar_res.dow_stats
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    dow_means = [dow_df.loc[dow_df["day_of_week"] == d, "mean"].values[0] if (dow_df["day_of_week"] == d).any() else 0.0 for d in range(5)]
    dow_errs = [dow_df.loc[dow_df["day_of_week"] == d, "se"].values[0] * 1.96 if (dow_df["day_of_week"] == d).any() else 0.0 for d in range(5)]

    axes[0].bar(days, [m * 100 for m in dow_means], yerr=[e * 100 for e in dow_errs], capsize=4, color="#1f77b4", alpha=0.8)
    axes[0].axhline(0, color="black", linestyle="-", linewidth=0.8)
    axes[0].set_title(f"Day-of-Week Effect (OLS F-p={calendar_res.ols_f_pvalue:.3f})", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Mean Daily Return (%) +/- 95% CI")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Month of year
    mo_df = calendar_res.month_stats
    mo_names = mo_df["month_name"].tolist()
    mo_means = [m * 100 for m in mo_df["mean"].tolist()]
    mo_errs = [e * 1.96 * 100 for e in mo_df["se"].tolist()]

    axes[1].bar(mo_names, mo_means, yerr=mo_errs, capsize=4, color="#2ca02c", alpha=0.8)
    axes[1].axhline(0, color="black", linestyle="-", linewidth=0.8)
    axes[1].set_title("Month-of-Year Effect", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Mean Daily Return (%) +/- 95% CI")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # 3. Lag-1 Plot
    r = log_returns.dropna()
    r_t = r.iloc[1:].values
    r_prev = r.iloc[:-1].values
    corr = np.corrcoef(r_prev, r_t)[0, 1]

    axes[2].scatter(r_prev * 100, r_t * 100, alpha=0.3, edgecolors="none", color="#4c72b0", s=15)
    slope, intercept = np.polyfit(r_prev * 100, r_t * 100, 1)
    x_vals = np.linspace(np.min(r_prev) * 100, np.max(r_prev) * 100, 100)
    axes[2].plot(x_vals, intercept + slope * x_vals, color="crimson", linewidth=1.5, label=f"Fit (r = {corr:.3f})")
    axes[2].axhline(0, color="gray", linestyle=":", linewidth=0.8)
    axes[2].axvline(0, color="gray", linestyle=":", linewidth=0.8)
    axes[2].set_title(f"Lag-1 Return Scatter (rho_1 = {corr:.3f})", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Return r_{t-1} (%)")
    axes[2].set_ylabel("Return r_t (%)")
    axes[2].legend(loc="upper right", frameon=True)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved seasonality subseries chart to '%s'", output_path)

    return fig


def plot_stl_components(
    stl_res: STLDecompositionResult,
    asset_name: str,
    output_path: str | None = None,
) -> plt.Figure:
    """Plots the original log price/NAV, trend, seasonal, and residual components."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Trend & Seasonal Price Decomposition: {asset_name}\n"
        f"Trend Strength F_T = {stl_res.trend_strength_f_t:.4f} | Seasonal Strength F_S = {stl_res.seasonal_strength_f_s:.4f}",
        fontsize=13,
        fontweight="bold",
    )

    observed = stl_res.trend + stl_res.seasonal + stl_res.residual

    axes[0].plot(observed.index, observed.values, color="black", linewidth=1.2)
    axes[0].set_ylabel("Log Price", fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].plot(stl_res.trend.index, stl_res.trend.values, color="#1f77b4", linewidth=1.5)
    axes[1].set_ylabel("Trend", fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    axes[2].plot(stl_res.seasonal.index, stl_res.seasonal.values, color="#2ca02c", linewidth=1.0)
    axes[2].set_ylabel("Seasonal", fontweight="bold")
    axes[2].grid(True, linestyle=":", alpha=0.6)

    axes[3].scatter(stl_res.residual.index, stl_res.residual.values, color="crimson", s=10, alpha=0.5)
    axes[3].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[3].set_ylabel("Residual", fontweight="bold")
    axes[3].set_xlabel("Date", fontweight="bold")
    axes[3].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved STL decomposition chart to '%s'", output_path)

    return fig
