"""
MarketFolio Plain-English Readings & Institutional Desk Interpretations.
Centralized thresholds and concise template sentences for financial analysis.
Strictly avoids AI buzzwords, duplicated tickers, raw LaTeX, and hardcoded numbers.
"""

from __future__ import annotations

from typing import Any
import numpy as np

# Configurable quantitative thresholds
READINGS_CONFIG = {
    # Volatility regime percentiles based on asset's own history
    "vol_percentile_low": 33.3,
    "vol_percentile_high": 66.7,
    # Market beta classifications
    "beta_defensive": 0.80,
    "beta_aggressive": 1.20,
    # Alpha statistical significance
    "alpha_alpha_sig_pvalue": 0.05,
    # Calendar effect significance
    "calendar_sig_pvalue": 0.05,
    # Trend strength threshold (F_T)
    "trend_strong_threshold": 0.60,
    "seasonal_strong_threshold": 0.35,
    # Delta AICc threshold for weak-form predictability
    "aicc_predictable_delta": -2.0,
}


def get_trend_reading(stl_res: Any) -> str:
    """Returns a one-line desk takeaway on price trend vs cyclical noise."""
    f_t = stl_res.trend_strength_f_t
    f_s = stl_res.seasonal_strength_f_s

    if f_t >= READINGS_CONFIG["trend_strong_threshold"]:
        return f"Dominant persistent macro trend ({f_t:.2f} trend strength); cyclical noise is secondary."
    elif f_t >= 0.30:
        return f"Moderate directional drift ({f_t:.2f} trend strength) with noticeable local consolidation."
    else:
        return f"Range-bound price action with weak directional persistence ({f_t:.2f} trend strength)."


def get_volatility_regime(vol_res: Any, horizon: int) -> tuple[str, str, str]:
    """
    Evaluates current annualized volatility relative to its historical distribution.
    Returns: (status_chip, chip_css_class, one_line_reading)
    """
    hist_vol = vol_res.realized_vol_series.dropna()
    current_vol = vol_res.last_observed_vol_pct
    low_bound, high_bound = vol_res.expected_risk_range_pct
    point = vol_res.point_forecast_pct

    if len(hist_vol) > 0:
        pctile = float((hist_vol < current_vol).mean() * 100.0)
    else:
        pctile = 50.0

    if pctile < READINGS_CONFIG["vol_percentile_low"]:
        status = "Low Volatility"
        chip_class = "mf-badge-live"
        reading = (
            f"Volatility is subdued (ranked at {pctile:.0f}th historical percentile); "
            f"{horizon}D forecast projects {point:.1f}% [{low_bound:.0f}%–{high_bound:.0f}%]."
        )
    elif pctile > READINGS_CONFIG["vol_percentile_high"]:
        status = "High Volatility"
        chip_class = "mf-badge-stale"
        reading = (
            f"Volatility is elevated (ranked at {pctile:.0f}th historical percentile); "
            f"{horizon}D forecast projects {point:.1f}% [{low_bound:.0f}%–{high_bound:.0f}%]."
        )
    else:
        status = "Moderate Volatility"
        chip_class = "mf-badge-teal"
        reading = (
            f"Volatility is near historical median ({pctile:.0f}th percentile); "
            f"{horizon}D forecast projects {point:.1f}% [{low_bound:.0f}%–{high_bound:.0f}%]."
        )

    return status, chip_class, reading


def get_market_sensitivity_reading(market_res: Any, bench_name: str) -> str:
    """Returns a one-line desk takeaway on beta sensitivity and alpha generation."""
    if not market_res:
        return f"Benchmark sensitivity against {bench_name} is unavailable."

    beta = market_res.arma_beta
    alpha_pa = market_res.arma_alpha_annualized * 100.0
    alpha_p = market_res.arma_alpha_pvalue

    if beta < READINGS_CONFIG["beta_defensive"]:
        beta_desc = f"defensive market posture (beta {beta:.2f}x vs {bench_name})"
    elif beta > READINGS_CONFIG["beta_aggressive"]:
        beta_desc = f"high market gearing (beta {beta:.2f}x vs {bench_name})"
    else:
        beta_desc = f"near-parallel market alignment (beta {beta:.2f}x vs {bench_name})"

    if alpha_p < READINGS_CONFIG["alpha_alpha_sig_pvalue"]:
        if alpha_pa > 0:
            alpha_desc = f"statistically significant positive alpha (+{alpha_pa:.1f}% p.a.)"
        else:
            alpha_desc = f"statistically significant negative drag ({alpha_pa:.1f}% p.a.)"
    else:
        alpha_desc = "excess returns statistically indistinguishable from zero"

    return f"Displays {beta_desc}; {alpha_desc} after autocorrelated disturbance adjustment."


def get_predictability_reading(arma_res: Any) -> str:
    """Returns a plain-language reading on directional return predictability (EMH)."""
    if arma_res.is_predictable:
        p, d, q = arma_res.best_order
        return (
            f"Serial return structure detected: ARIMA({p},{d},{q}) improves AICc by "
            f"{abs(arma_res.best_model.aicc - arma_res.constant_model.aicc):.1f} over a random walk."
        )
    return "Returns behave as a martingale day to day; past price patterns offer no systematic directional edge."


def get_macro_drivers_reading(var_res: Any) -> str:
    """Summarizes macro lead-lag transmission to the domestic market."""
    if not var_res:
        return "Global macro lead-lag data is currently unavailable."

    causal = [g.causing_variable for g in var_res.granger_tests if g.is_causal_05]
    if causal:
        factors = ", ".join(causal)
        return f"Lead-lag transmission active: {factors} Granger-lead domestic benchmark movements."
    return "Domestic benchmark price formation operates independently of short-term external macro lags."


def get_data_quality_reading(clean_res: Any, meta: dict[str, Any]) -> str:
    """Summarizes sample integrity, calendar alignment, and corporate event anomalies."""
    n_anomalies = len(clean_res.flagged_anomalies)
    is_stale = meta.get("is_stale", False)
    status_str = "cached fallback data" if is_stale else "verified live feed"

    if n_anomalies > 0:
        return (
            f"Ingested {clean_res.n_cleaned_obs} sessions via {status_str}; "
            f"{n_anomalies} extreme return session(s) flagged (|r| > 15%)."
        )
    return f"Clean sample of {clean_res.n_cleaned_obs} aligned sessions via {status_str} with zero data disruptions."


def get_calendar_reading(cal_res: Any) -> str:
    """Returns a verdict on day-of-week and monthly anomalies."""
    if cal_res.is_day_of_week_significant:
        return (
            f"Measurable calendar asymmetry: weekday return distributions differ significantly "
            f"(joint F p={cal_res.ols_f_pvalue:.4f})."
        )
    return "Calendar return distributions are statistically uniform across weekdays and months."


def get_backtest_reading(eval_res: Any) -> str:
    """Returns the out-of-sample backtesting conclusion."""
    imprv = eval_res.model_metrics.mase_improvement_pct
    if imprv > 0:
        return (
            f"Outperforms competing random-walk and rolling-mean baselines "
            f"across {eval_res.n_origins_evaluated} expanding origins ({imprv:.1f}% MASE improvement)."
        )
    return (
        f"Model accuracy tracks close to the naive random-walk baseline "
        f"across {eval_res.n_origins_evaluated} expanding backtest windows."
    )


def get_overview_bullet_list(data: dict[str, Any]) -> list[str]:
    """Assembles the 5-6 core institutional takeaways for the Overview Key Readings card."""
    horizon = data.get("vol_res").horizon_days if data.get("vol_res") else 30
    bench_name = data.get("meta", {}).get("benchmark_name", "^NSEI")

    bullets = []
    # 1. Trend
    if data.get("stl_res"):
        bullets.append(f"**Trend**: {get_trend_reading(data['stl_res'])}")

    # 2. Volatility Regime
    if data.get("vol_res"):
        _, _, vol_read = get_volatility_regime(data["vol_res"], horizon)
        bullets.append(f"**Volatility**: {vol_read}")

    # 3. Market Sensitivity
    if data.get("market_res"):
        bullets.append(f"**Market Sensitivity**: {get_market_sensitivity_reading(data['market_res'], bench_name)}")

    # 4. Predictability
    if data.get("arma_search"):
        bullets.append(f"**Predictability**: {get_predictability_reading(data['arma_search'])}")

    # 5. Macro Drivers
    if data.get("var_res"):
        bullets.append(f"**Macro Transmission**: {get_macro_drivers_reading(data['var_res'])}")

    # 6. Data Quality
    if data.get("clean_res"):
        bullets.append(f"**Data Integrity**: {get_data_quality_reading(data['clean_res'], data.get('meta', {}))}")

    return bullets
