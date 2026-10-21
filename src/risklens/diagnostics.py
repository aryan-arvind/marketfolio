"""
Time series stationarity and diagnostic tests for RiskLens (ATSA Units 1 & 5).
Implements Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests
with joint decision synthesis, ACF/PACF analysis with confidence bands, and Ljung-Box tests.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, NamedTuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

logger = logging.getLogger("risklens")


class UnitRootTestResult(NamedTuple):
    """Encapsulates ADF and KPSS test metrics for a single time series."""
    series_name: str
    adf_stat: float
    adf_pvalue: float
    adf_lags: int
    adf_nobs: int
    adf_crit_values: dict[str, float]
    kpss_stat: float
    kpss_pvalue: float
    kpss_lags: int
    kpss_crit_values: dict[str, float]
    kpss_truncated: bool
    verdict: str
    explanation: str


def run_unit_root_tests(series: pd.Series, series_name: str) -> UnitRootTestResult:
    """
    Executes ADF (unit root null) and KPSS (stationarity null) tests on a series.
    Synthesizes results into a unified ATSA stationarity verdict.
    """
    clean_vals = series.dropna().to_numpy()
    if len(clean_vals) < 30:
        raise ValueError(f"Series '{series_name}' has insufficient observations ({len(clean_vals)}) for stationarity testing.")

    # 1. ADF Test (H0: Unit Root present, Ha: Stationary)
    adf_res = adfuller(clean_vals, autolag="AIC", result_object=False)
    adf_stat = float(adf_res[0])
    adf_pvalue = float(adf_res[1])
    adf_lags = int(adf_res[2])
    adf_nobs = int(adf_res[3])
    adf_crit = {k: float(v) for k, v in adf_res[4].items()}

    # 2. KPSS Test (H0: Trend/Level Stationary, Ha: Unit Root present)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        kpss_res = kpss(clean_vals, regression="c", nlags="auto")
        
        # Check if InterpolationWarning was issued (p-value bound hit)
        kpss_truncated = any("InterpolationWarning" in str(w.message) for w in caught_warnings)
        if kpss_truncated:
            logger.debug("KPSS p-value bounded at edge for '%s' (statsmodels asymptotic table bounds: 0.01 - 0.10).", series_name)

    kpss_stat = float(kpss_res[0])
    kpss_pvalue = float(kpss_res[1])
    kpss_lags = int(kpss_res[2])
    kpss_crit = {k: float(v) for k, v in kpss_res[3].items()}

    # Combined ATSA Decision Matrix at alpha = 0.05
    adf_rejects_null = adf_pvalue < 0.05
    kpss_rejects_null = kpss_pvalue < 0.05

    if adf_rejects_null and not kpss_rejects_null:
        verdict = "Stationary (Agreement)"
        explanation = (
            "Both tests agree: ADF rejects the unit root null (p < 0.05), and KPSS fails to reject "
            "the stationarity null (p >= 0.05). The series is covariance-stationary (I(0))."
        )
    elif not adf_rejects_null and kpss_rejects_null:
        verdict = "Unit Root / Non-Stationary (Agreement)"
        explanation = (
            "Both tests agree: ADF fails to reject the unit root null (p >= 0.05), and KPSS rejects "
            "the stationarity null (p < 0.05). The series has a stochastic trend / unit root (I(1))."
        )
    elif adf_rejects_null and kpss_rejects_null:
        verdict = "Conflicting: Trend- or Difference-Stationary Ambiguity"
        explanation = (
            "Conflicting results: ADF suggests stationarity (p < 0.05) while KPSS rejects stationarity "
            "(p < 0.05). This pattern commonly arises when the series exhibits structural breaks, "
            "long memory, or deterministic trend without unit root (trend-stationary ambiguity)."
        )
    else:
        verdict = "Conflicting / Insufficient Statistical Power"
        explanation = (
            "Indeterminate: ADF fails to reject unit root (p >= 0.05) and KPSS fails to reject "
            "stationarity (p >= 0.05). The sample size or signal-to-noise ratio is insufficient to distinguish "
            "a highly persistent stationary process from a genuine random walk."
        )

    logger.info(
        "Unit root diagnostics for '%s': ADF stat=%.4f (p=%.4f), KPSS stat=%.4f (p=%.4f) -> %s",
        series_name, adf_stat, adf_pvalue, kpss_stat, kpss_pvalue, verdict
    )

    return UnitRootTestResult(
        series_name=series_name,
        adf_stat=adf_stat,
        adf_pvalue=adf_pvalue,
        adf_lags=adf_lags,
        adf_nobs=adf_nobs,
        adf_crit_values=adf_crit,
        kpss_stat=kpss_stat,
        kpss_pvalue=kpss_pvalue,
        kpss_lags=kpss_lags,
        kpss_crit_values=kpss_crit,
        kpss_truncated=kpss_truncated,
        verdict=verdict,
        explanation=explanation,
    )


def run_stationarity_suite(
    price_series: pd.Series,
    log_return_series: pd.Series,
    asset_name: str,
) -> dict[str, UnitRootTestResult]:
    """
    Executes ADF and KPSS stationarity tests on:
    (a) Log Price / NAV
    (b) Log Returns
    """
    log_price = np.log(price_series).dropna()
    price_res = run_unit_root_tests(log_price, f"{asset_name} Log Price/NAV")
    return_res = run_unit_root_tests(log_return_series, f"{asset_name} Log Returns")

    return {
        "log_price": price_res,
        "log_returns": return_res,
    }


def compute_ljung_box(
    series: pd.Series, lags: list[int] = [10, 20], model_df: int = 0
) -> pd.DataFrame:
    """
    Executes Ljung-Box test for serial autocorrelation at specified lags.
    Handles degrees of freedom correction (model_df = p + q for residual tests).
    """
    clean_vals = series.dropna().to_numpy()
    lb_df = acorr_ljungbox(clean_vals, lags=lags, return_df=True, model_df=model_df)
    lb_df["significant_at_05"] = lb_df["lb_pvalue"] < 0.05
    return lb_df


def plot_acf_pacf_diagnostics(
    log_returns: pd.Series,
    asset_name: str,
    output_path: str | None = None,
    nlags: int = 40,
) -> plt.Figure:
    """
    Generates a 2x2 grid of ACF and PACF plots for:
    - Log returns r_t (tests weak-form predictability)
    - Squared log returns r_t^2 (tests volatility clustering / ARCH effects)
    Includes 95% Bartlett / Gaussian confidence bands (+/- 1.96 / sqrt(N)).
    """
    r = log_returns.dropna().to_numpy()
    r2 = r ** 2
    n = len(r)
    conf_band = 1.96 / np.sqrt(n)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"ATSA Unit 1 & 5 Diagnostic ACF/PACF Analysis: {asset_name}\n"
        f"Sample size N={n} | 95% Bartlett confidence threshold = +/- {conf_band:.4f}",
        fontsize=13,
        fontweight="bold",
    )

    # 1. ACF of Returns
    acf_r = acf(r, nlags=nlags, alpha=0.05)
    acf_vals = acf_r[0]
    axes[0, 0].stem(range(nlags + 1), acf_vals, basefmt="k-")
    axes[0, 0].axhline(0, color="gray", linestyle="-", linewidth=0.8)
    axes[0, 0].axhline(conf_band, color="crimson", linestyle="--", linewidth=1.2, label=f"95% CI (+/-{conf_band:.3f})")
    axes[0, 0].axhline(-conf_band, color="crimson", linestyle="--", linewidth=1.2)
    axes[0, 0].set_title("ACF of Log Returns r_t (Mean Autocorrelation)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Lag")
    axes[0, 0].set_ylabel("Autocorrelation")
    axes[0, 0].legend(loc="upper right", frameon=True)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # 2. PACF of Returns
    pacf_r = pacf(r, nlags=nlags, alpha=0.05)
    pacf_vals = pacf_r[0]
    axes[0, 1].stem(range(nlags + 1), pacf_vals, basefmt="k-")
    axes[0, 1].axhline(0, color="gray", linestyle="-", linewidth=0.8)
    axes[0, 1].axhline(conf_band, color="crimson", linestyle="--", linewidth=1.2)
    axes[0, 1].axhline(-conf_band, color="crimson", linestyle="--", linewidth=1.2)
    axes[0, 1].set_title("PACF of Log Returns r_t (Direct Lag Correlation)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Lag")
    axes[0, 1].set_ylabel("Partial Autocorrelation")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # 3. ACF of Squared Returns (Volatility Clustering)
    acf_r2 = acf(r2, nlags=nlags, alpha=0.05)
    acf_r2_vals = acf_r2[0]
    axes[1, 0].stem(range(nlags + 1), acf_r2_vals, basefmt="k-")
    axes[1, 0].axhline(0, color="gray", linestyle="-", linewidth=0.8)
    axes[1, 0].axhline(conf_band, color="crimson", linestyle="--", linewidth=1.2)
    axes[1, 0].axhline(-conf_band, color="crimson", linestyle="--", linewidth=1.2)
    axes[1, 0].set_title("ACF of Squared Returns r_t^2 (Volatility Clustering)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Lag")
    axes[1, 0].set_ylabel("Autocorrelation")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # 4. PACF of Squared Returns
    pacf_r2 = pacf(r2, nlags=nlags, alpha=0.05)
    pacf_r2_vals = pacf_r2[0]
    axes[1, 1].stem(range(nlags + 1), pacf_r2_vals, basefmt="k-")
    axes[1, 1].axhline(0, color="gray", linestyle="-", linewidth=0.8)
    axes[1, 1].axhline(conf_band, color="crimson", linestyle="--", linewidth=1.2)
    axes[1, 1].axhline(-conf_band, color="crimson", linestyle="--", linewidth=1.2)
    axes[1, 1].set_title("PACF of Squared Returns r_t^2 (ARCH Signatures)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Lag")
    axes[1, 1].set_ylabel("Partial Autocorrelation")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved ACF/PACF diagnostics chart to '%s'", output_path)

    return fig
