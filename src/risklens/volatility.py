"""
Realized volatility modeling, ARIMA estimation, analytic forecasting,
and residual bootstrap simulation (ATSA Units 4, 5, 8).
Forecasts log annualized realized volatility v_t = ln(sqrt(252) * std_21(r_t))
and generates empirical confidence intervals via residual bootstrap (Efron 1979).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from risklens.arma_models import calculate_aicc
from risklens.config import (
    ANNUALIZATION_FACTOR,
    BOOTSTRAP_RANDOM_SEED,
    BOOTSTRAP_REPLICATIONS,
    DEFAULT_FORECAST_HORIZON,
    REALIZED_VOLATILITY_WINDOW,
)
from risklens.diagnostics import run_unit_root_tests

logger = logging.getLogger("risklens")


class VolatilityForecastResult(NamedTuple):
    """Complete summary of volatility modeling and forecasting."""
    realized_vol_series: pd.Series
    log_vol_series: pd.Series
    d_order: int
    best_order: tuple[int, int, int]
    last_observed_vol_pct: float
    horizon_days: int
    analytic_point_pred: np.ndarray
    analytic_lower_90: np.ndarray
    analytic_upper_90: np.ndarray
    boot_median: np.ndarray
    boot_lower_90: np.ndarray
    boot_upper_90: np.ndarray
    boot_failed_or_fallback: bool
    fallback_reason: str
    expected_risk_range_pct: tuple[float, float]
    point_forecast_pct: float
    naive_forecast_pct: float
    rolling_mean_forecast_pct: float
    model_fit: Any


def compute_realized_volatility(
    log_returns: pd.Series,
    window: int = REALIZED_VOLATILITY_WINDOW,
    annualize: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """
    Computes annualized realized volatility and its natural logarithm:
    sigma_t = sqrt(252) * rolling_std(r_t, window=21, ddof=1)
    v_t = ln(sigma_t)
    
    Note: Overlapping rolling windows induce serial correlation in v_t by construction,
    making it an ideal stationary or persistent time series for ARMA forecasting.
    """
    rolling_std = log_returns.rolling(window=window).std(ddof=1).dropna()
    # Filter out any non-positive values
    rolling_std = rolling_std[rolling_std > 1e-8]

    if annualize:
        realized_vol = rolling_std * ANNUALIZATION_FACTOR
    else:
        realized_vol = rolling_std

    log_vol = np.log(realized_vol)
    return realized_vol, log_vol


def select_and_fit_volatility_arima(
    log_vol: pd.Series,
    asset_name: str,
) -> tuple[tuple[int, int, int], Any]:
    """
    Checks stationarity of log realized volatility, determines integration order d,
    and selects best ARIMA(p, d, q) by AICc over p in 0..3, q in 0..3.
    """
    # 1. Stationarity diagnostic
    diag = run_unit_root_tests(log_vol, f"{asset_name} Log Realized Vol")
    # If ADF rejects unit root (p < 0.05), use d=0; otherwise d=1
    d = 0 if diag.adf_pvalue < 0.05 else 1

    clean_vals = log_vol.dropna().to_numpy()
    search_vals = clean_vals[-1500:] if len(clean_vals) > 1500 else clean_vals
    nobs_search = len(search_vals)

    best_aicc = float("inf")
    best_order = (1, d, 1)

    for p in range(3):
        for q in range(3):
            order = (p, d, q)
            trend = "c" if d == 0 else None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(search_vals, order=order, trend=trend)
                    res = model.fit(method_kwargs={"maxiter": 25, "disp": False})
                aicc = calculate_aicc(float(res.aic), nobs_search, len(res.params))
                if aicc < best_aicc:
                    best_aicc = aicc
                    best_order = order
            except Exception:
                continue

    # Fit the chosen best order on the full historical series
    trend = "c" if d == 0 else None
    try:
        best_fit = ARIMA(clean_vals, order=best_order, trend=trend).fit(
            method_kwargs={"maxiter": 30, "disp": False}
        )
    except Exception:
        best_order = (1, d, 0)
        best_fit = ARIMA(clean_vals, order=best_order, trend=trend).fit(
            method_kwargs={"maxiter": 25, "disp": False}
        )

    logger.info("Selected Volatility ARIMA%s for '%s' (AICc=%.2f)", best_order, asset_name, best_aicc)
    return best_order, best_fit


def simulate_residual_bootstrap(
    model_fit: Any,
    observed_log_vol: np.ndarray,
    order: tuple[int, int, int],
    horizon: int = DEFAULT_FORECAST_HORIZON,
    b_replications: int = BOOTSTRAP_REPLICATIONS,
    seed: int = BOOTSTRAP_RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, str]:
    """
    Implements recursive residual bootstrap for ARIMA(p, d, q) (Efron 1979).
    
    Detailed handling:
    1. Extracts in-sample residuals e_t and centers them to mean zero.
    2. Identifies AR coefficients (phi_1..phi_p) and MA coefficients (theta_1..theta_q).
    3. For d=0 with trend='c', statsmodels parameter 'const' is the process mean mu,
       so the ARMA recursion operates on mean-deviations: x_t = y_t - mu.
    4. For d=1, the recursion operates on differenced series w_t = Delta y_t,
       and paths are reconstructed via y_{T+k} = y_T + sum_{m=1}^k w_{T+m}.
    5. Checks agreement between bootstrap median and analytic forecast.
    """
    p, d, q = order
    rng = np.random.default_rng(seed)

    raw_resid = model_fit.resid
    # Drop initial zeros/warmup
    clean_resid = raw_resid[p + d :] if len(raw_resid) > p + d else raw_resid
    centered_resid = clean_resid - np.mean(clean_resid)

    if len(centered_resid) < 20:
        return np.array([]), np.array([]), np.array([]), True, "Insufficient residuals for bootstrap."

    # Extract parameters
    param_names = list(model_fit.param_names)
    params = model_fit.params

    # Mean parameter mu
    mu = 0.0
    if "const" in param_names:
        const_idx = param_names.index("const")
        mu = float(params[const_idx])

    # AR coefficients
    ar_coeffs = []
    for i in range(1, p + 1):
        key = f"ar.L{i}"
        if key in param_names:
            ar_coeffs.append(float(params[param_names.index(key)]))
        else:
            ar_coeffs.append(0.0)

    # MA coefficients
    ma_coeffs = []
    for j in range(1, q + 1):
        key = f"ma.L{j}"
        if key in param_names:
            ma_coeffs.append(float(params[param_names.index(key)]))
        else:
            ma_coeffs.append(0.0)

    simulated_paths = np.zeros((b_replications, horizon))

    # Determine conditioning values
    if d == 0:
        # Recursion on mean-adjusted series: x_t = y_t - mu
        x_observed = observed_log_vol - mu
        last_resid = centered_resid[-max(q, 1) :] if q > 0 else np.array([])
        last_x = x_observed[-max(p, 1) :] if p > 0 else np.array([])

        for b in range(b_replications):
            # Sample future innovations with replacement
            innovations = rng.choice(centered_resid, size=horizon, replace=True)
            x_path = list(last_x)
            e_path = list(last_resid)

            path_vals = []
            for k in range(horizon):
                e_k = innovations[k]
                # AR component
                ar_term = sum(ar_coeffs[i] * x_path[-1 - i] for i in range(p)) if p > 0 else 0.0
                # MA component
                ma_term = sum(ma_coeffs[j] * e_path[-1 - j] for j in range(q)) if q > 0 else 0.0
                
                x_k = ar_term + ma_term + e_k
                x_path.append(x_k)
                e_path.append(e_k)
                path_vals.append(x_k + mu)

            simulated_paths[b, :] = path_vals

    else:
        # d == 1: Model fitted on differences w_t = Delta y_t
        w_observed = np.diff(observed_log_vol)
        last_y = observed_log_vol[-1]
        last_resid = centered_resid[-max(q, 1) :] if q > 0 else np.array([])
        last_w = w_observed[-max(p, 1) :] if p > 0 else np.array([])

        for b in range(b_replications):
            innovations = rng.choice(centered_resid, size=horizon, replace=True)
            w_path = list(last_w)
            e_path = list(last_resid)

            w_sim = []
            for k in range(horizon):
                e_k = innovations[k]
                ar_term = sum(ar_coeffs[i] * w_path[-1 - i] for i in range(p)) if p > 0 else 0.0
                ma_term = sum(ma_coeffs[j] * e_path[-1 - j] for j in range(q)) if q > 0 else 0.0
                w_k = ar_term + ma_term + e_k
                w_path.append(w_k)
                e_path.append(e_k)
                w_sim.append(w_k)

            # Reconstruct original levels by cumulating differences
            y_path = last_y + np.cumsum(w_sim)
            simulated_paths[b, :] = y_path

    # Percentiles: 5th, 50th (median), 95th across B paths
    boot_lower = np.percentile(simulated_paths, 5, axis=0)
    boot_median = np.percentile(simulated_paths, 50, axis=0)
    boot_upper = np.percentile(simulated_paths, 95, axis=0)

    return boot_lower, boot_median, boot_upper, False, ""


def forecast_volatility(
    log_returns: pd.Series,
    asset_name: str,
    horizon: int = DEFAULT_FORECAST_HORIZON,
) -> VolatilityForecastResult:
    """
    Orchestrates realized volatility modeling, estimation, and forecasting.
    Computes both analytic and residual bootstrap intervals, compares against
    naive and rolling-mean baselines, and generates the annualized risk range.
    """
    realized_vol, log_vol = compute_realized_volatility(log_returns)
    last_vol_pct = float(realized_vol.iloc[-1] * 100.0)

    order, model_fit = select_and_fit_volatility_arima(log_vol, asset_name)
    p, d, q = order

    # 1. Analytic Forecast
    forecast_obj = model_fit.get_forecast(steps=horizon)
    analytic_point_log = np.array(forecast_obj.predicted_mean)
    # 90% confidence interval (alpha=0.10) for comparison with 5th-95th bootstrap percentiles
    ci_df = forecast_obj.conf_int(alpha=0.10)
    analytic_lower_log = ci_df[:, 0] if isinstance(ci_df, np.ndarray) else ci_df.iloc[:, 0].values
    analytic_upper_log = ci_df[:, 1] if isinstance(ci_df, np.ndarray) else ci_df.iloc[:, 1].values

    # 2. Residual Bootstrap Forecast
    log_vol_vals = log_vol.dropna().to_numpy()
    boot_lower_log, boot_median_log, boot_upper_log, boot_failed, fallback_reason = simulate_residual_bootstrap(
        model_fit=model_fit,
        observed_log_vol=log_vol_vals,
        order=order,
        horizon=horizon,
    )

    # Self-Check: Verify bootstrap median is close to analytic forecast (gap < 10%)
    if not boot_failed and len(boot_median_log) == horizon:
        abs_relative_gap = np.max(np.abs((boot_median_log - analytic_point_log) / (np.abs(analytic_point_log) + 1e-6)))
        if abs_relative_gap > 0.10:
            logger.warning(
                "Bootstrap median diverged from analytic forecast by >10%% (max relative gap: %.2f%%). "
                "Logging warning as specified in ATSA protocol.",
                abs_relative_gap * 100.0
            )

    if boot_failed or len(boot_median_log) != horizon:
        logger.warning("Residual bootstrap failed (%s). Falling back to analytic intervals.", fallback_reason)
        boot_median_log = analytic_point_log
        boot_lower_log = analytic_lower_log
        boot_upper_log = analytic_upper_log
        boot_failed = True

    # Convert from log volatility back to annualized volatility percentages: exp(v) * 100
    analytic_point_pct = np.exp(analytic_point_log) * 100.0
    analytic_lower_pct = np.exp(analytic_lower_log) * 100.0
    analytic_upper_pct = np.exp(analytic_upper_log) * 100.0

    boot_median_pct = np.exp(boot_median_log) * 100.0
    boot_lower_pct = np.exp(boot_lower_log) * 100.0
    boot_upper_pct = np.exp(boot_upper_log) * 100.0

    # Baselines for evaluation
    # Naive baseline: last observed value constant forward
    naive_forecast_pct = last_vol_pct
    # Rolling-mean baseline: mean of last 21 days
    rolling_mean_forecast_pct = float(realized_vol.iloc[-REALIZED_VOLATILITY_WINDOW:].mean() * 100.0)

    # 30-day (horizon-step) expected risk range
    expected_risk_range = (float(boot_lower_pct[-1]), float(boot_upper_pct[-1]))
    final_point_pred = float(boot_median_pct[-1])

    logger.info(
        "Realized Volatility Forecast (%d days): Current=%.2f%%, Forecast Median=%.2f%%, 90%% Range=[%.2f%%, %.2f%%]",
        horizon, last_vol_pct, final_point_pred, expected_risk_range[0], expected_risk_range[1]
    )

    return VolatilityForecastResult(
        realized_vol_series=realized_vol,
        log_vol_series=log_vol,
        d_order=d,
        best_order=order,
        last_observed_vol_pct=last_vol_pct,
        horizon_days=horizon,
        analytic_point_pred=analytic_point_pct,
        analytic_lower_90=analytic_lower_pct,
        analytic_upper_90=analytic_upper_pct,
        boot_median=boot_median_pct,
        boot_lower_90=boot_lower_pct,
        boot_upper_90=boot_upper_pct,
        boot_failed_or_fallback=boot_failed,
        fallback_reason=fallback_reason,
        expected_risk_range_pct=expected_risk_range,
        point_forecast_pct=final_point_pred,
        naive_forecast_pct=naive_forecast_pct,
        rolling_mean_forecast_pct=rolling_mean_forecast_pct,
        model_fit=model_fit,
    )


def plot_volatility_forecast(
    vol_res: VolatilityForecastResult,
    asset_name: str,
    output_path: str | None = None,
) -> plt.Figure:
    """
    Plots the historical annualized realized volatility, the h-step point forecast,
    and the 90% analytic and bootstrap forecast intervals.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Show last 252 trading days (~1 year) of historical volatility for context
    hist_vol = vol_res.realized_vol_series.iloc[-252:] * 100.0
    h_steps = np.arange(1, vol_res.horizon_days + 1)

    ax.plot(range(-len(hist_vol) + 1, 1), hist_vol.values, color="#1f77b4", linewidth=1.5, label="Historical Realized Volatility (21-Day Rolling)")
    ax.scatter([0], [vol_res.last_observed_vol_pct], color="black", s=40, zorder=5, label=f"Last Observed ({vol_res.last_observed_vol_pct:.1f}%)")

    # Point forecast
    ax.plot(h_steps, vol_res.boot_median, color="crimson", linewidth=2.0, marker="o", markersize=3, label=f"ARIMA{vol_res.best_order} Forecast (Median)")

    # Bootstrap 90% interval
    ax.fill_between(h_steps, vol_res.boot_lower_90, vol_res.boot_upper_90, color="crimson", alpha=0.2, label="Bootstrap 90% Interval (5th-95th)")

    # Analytic 90% interval (dashed lines)
    ax.plot(h_steps, vol_res.analytic_lower_90, color="navy", linestyle="--", linewidth=1.2, label="Analytic 90% Bounds")
    ax.plot(h_steps, vol_res.analytic_upper_90, color="navy", linestyle="--", linewidth=1.2)

    # Baselines
    ax.axhline(vol_res.naive_forecast_pct, color="gray", linestyle=":", label=f"Naive Baseline ({vol_res.naive_forecast_pct:.1f}%)")
    ax.axhline(vol_res.rolling_mean_forecast_pct, color="darkorange", linestyle=":", label=f"21-Day Rolling Mean ({vol_res.rolling_mean_forecast_pct:.1f}%)")

    ax.set_title(
        f"Realized Volatility Forecast & Risk Cone: {asset_name}\n"
        f"Expected 30-Day Risk Range: [{vol_res.expected_risk_range_pct[0]:.1f}% - {vol_res.expected_risk_range_pct[1]:.1f}%] p.a.",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Trading Days Relative to Forecast Origin (0 = Latest Close)", fontsize=10)
    ax.set_ylabel("Annualized Volatility (% p.a.)", fontsize=10)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved volatility forecast chart to '%s'", output_path)

    return fig
