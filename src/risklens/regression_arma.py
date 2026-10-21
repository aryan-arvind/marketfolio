"""
Regression with ARMA errors (ATSA Units 2 & 5).
Estimates single-index market models (Asset Return ~ Alpha + Beta * Benchmark Return + ARMA(p, q) errors)
using statsmodels SARIMAX. Compares against Newey-West HAC OLS and provides econometric interpretation.
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
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from risklens.arma_models import calculate_aicc

logger = logging.getLogger("risklens")


class MarketRegressionResult(NamedTuple):
    """Detailed parameters and diagnostics from OLS and SARIMAX regressions."""
    asset_name: str
    benchmark_name: str
    n_aligned_obs: int
    start_date: str
    end_date: str
    # OLS metrics
    ols_alpha_daily: float
    ols_alpha_annualized: float
    ols_alpha_pvalue: float
    ols_beta: float
    ols_beta_se_hac: float
    ols_beta_pvalue: float
    ols_r_squared: float
    ols_dw_stat: float
    ols_lb_pvalue: float
    # SARIMAX ARMA error metrics
    best_arma_order: tuple[int, int, int]
    arma_alpha_daily: float
    arma_alpha_annualized: float
    arma_alpha_se: float
    arma_alpha_pvalue: float
    arma_alpha_ci: tuple[float, float]
    arma_beta: float
    arma_beta_se: float
    arma_beta_pvalue: float
    arma_beta_ci: tuple[float, float]
    arma_aicc: float
    sarimax_fit: Any
    interpretation: str


def fit_regression_arma(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    asset_name: str = "Asset",
    benchmark_name: str = "^NSEI",
    max_p: int = 2,
    max_q: int = 2,
) -> MarketRegressionResult:
    """
    Fits single-index market model:
    r_t^Asset = alpha + beta * r_t^Bench + eta_t,  eta_t ~ ARMA(p, q).
    
    1. Aligns returns on common dates (inner join).
    2. Runs benchmark OLS with Newey-West (HAC) standard errors.
    3. Calculates Durbin-Watson statistic and Ljung-Box test on OLS residuals.
    4. Searches (p, q) in [0..2]x[0..2] using SARIMAX with trend='c' and exog.
    5. Formulates econometric interpretation based on alpha significance and beta value.
    """
    # 1. Inner join on common dates
    aligned = pd.concat([asset_returns.rename("asset"), benchmark_returns.rename("bench")], axis=1, join="inner").dropna()
    n_obs = len(aligned)

    if n_obs < 100:
        raise ValueError(
            f"Insufficient common observations ({n_obs}) between '{asset_name}' and benchmark '{benchmark_name}'."
        )

    logger.info(
        "Aligned %d common return observations for '%s' vs '%s' (Dates: %s to %s). Note: Fund NAV is struck at EOD.",
        n_obs, asset_name, benchmark_name,
        aligned.index.min().strftime("%Y-%m-%d"),
        aligned.index.max().strftime("%Y-%m-%d"),
    )

    y = aligned["asset"].values
    X_scalar = aligned["bench"].values
    X_ols = sm.add_constant(X_scalar)

    # 2. OLS with Newey-West HAC covariance (maxlags=5)
    ols_model = sm.OLS(y, X_ols)
    ols_res = ols_model.fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    ols_alpha = float(ols_res.params[0])
    ols_beta = float(ols_res.params[1])
    ols_alpha_ann = ols_alpha * 252.0
    ols_alpha_p = float(ols_res.pvalues[0])
    ols_beta_se = float(ols_res.bse[1])
    ols_beta_p = float(ols_res.pvalues[1])
    ols_r2 = float(ols_res.rsquared)

    # OLS residual diagnostics
    ols_resid = ols_res.resid
    dw_val = float(durbin_watson(ols_resid))
    lb_ols = acorr_ljungbox(ols_resid, lags=[10], return_df=True)
    ols_lb_p = float(lb_ols.iloc[0]["lb_pvalue"])

    # 3. SARIMAX Grid Search for ARMA(p, q) errors
    best_aicc = float("inf")
    best_order = (0, 0, 0)
    best_sarimax = None

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        y,
                        exog=X_scalar,
                        order=(p, 0, q),
                        trend="c",
                        enforce_stationarity=True,
                        enforce_invertibility=False,
                    )
                    res = model.fit(disp=False)
                k_params = len(res.params)
                aicc = calculate_aicc(float(res.aic), n_obs, k_params)
                if aicc < best_aicc:
                    best_aicc = aicc
                    best_order = (p, 0, q)
                    best_sarimax = res
            except Exception:
                continue

    if best_sarimax is None:
        logger.warning("SARIMAX grid search failed. Using ARMA(0,0,0) baseline.")
        best_sarimax = SARIMAX(y, exog=X_scalar, order=(0, 0, 0), trend="c").fit(disp=False)
        best_order = (0, 0, 0)
        best_aicc = calculate_aicc(float(best_sarimax.aic), n_obs, len(best_sarimax.params))

    # Extract SARIMAX estimates
    # Parameter order in SARIMAX: intercept ('intercept'), beta ('x1'), then AR, then MA, then sigma2
    p_names = list(best_sarimax.param_names)
    params_s = best_sarimax.params
    bse_s = best_sarimax.bse
    pvals_s = best_sarimax.pvalues
    ci_s = best_sarimax.conf_int()

    # Intercept
    int_idx = 0
    if "intercept" in p_names:
        int_idx = p_names.index("intercept")
    elif "const" in p_names:
        int_idx = p_names.index("const")

    arma_alpha = float(params_s[int_idx])
    arma_alpha_ann = arma_alpha * 252.0
    arma_alpha_se = float(bse_s[int_idx])
    arma_alpha_p = float(pvals_s[int_idx])
    arma_alpha_ci = (float(ci_s[int_idx, 0]) * 252.0, float(ci_s[int_idx, 1]) * 252.0)

    # Beta
    beta_idx = 1
    if "x1" in p_names:
        beta_idx = p_names.index("x1")

    arma_beta = float(params_s[beta_idx])
    arma_beta_se = float(bse_s[beta_idx])
    arma_beta_p = float(pvals_s[beta_idx])
    arma_beta_ci = (float(ci_s[beta_idx, 0]), float(ci_s[beta_idx, 1]))

    # Econometric interpretation synthesis
    alpha_status = "statistically significant" if arma_alpha_p < 0.05 else "statistically indistinguishable from zero"
    
    if ols_r2 > 0.85 and abs(arma_beta - 1.0) < 0.15:
        tracking_text = "closely tracks the benchmark (high R^2 and beta near 1.0, typical of an index tracker or large-cap core fund)"
    elif arma_beta > 1.10:
        tracking_text = f"exhibits aggressive cyclical market sensitivity (high beta = {arma_beta:.2f})"
    elif arma_beta < 0.85:
        tracking_text = f"demonstrates defensive or low-beta posture (beta = {arma_beta:.2f})"
    else:
        tracking_text = f"shows moderate market co-movement (beta = {arma_beta:.2f}, R^2 = {ols_r2:.2f})"

    if arma_alpha_ann > 0 and arma_alpha_p < 0.05:
        alpha_text = f"generates positive annual alpha (+{arma_alpha_ann*100:.2f}% p.a., p={arma_alpha_p:.4f})"
    elif arma_alpha_ann < 0 and arma_alpha_p < 0.05:
        alpha_text = f"exhibits negative annual alpha ({arma_alpha_ann*100:.2f}% p.a., p={arma_alpha_p:.4f}, indicating fee drag or underperformance)"
    else:
        alpha_text = f"generates no statistically significant abnormal alpha (annual alpha = {arma_alpha_ann*100:.2f}%, p={arma_alpha_p:.4f})"

    interpretation = (
        f"The asset {tracking_text} and {alpha_text}. "
        f"ARMA errors order {best_order} was selected by AICc. "
        f"OLS Durbin-Watson statistic is {dw_val:.2f} (values near 2.0 indicate minimal autocorrelation; "
        f"OLS residual Ljung-Box p={ols_lb_p:.4f}). "
        f"Modeling autocorrelated errors ensures robust standard error estimation without downward bias."
    )

    logger.info("Market regression for '%s' vs '%s': %s", asset_name, benchmark_name, interpretation)

    return MarketRegressionResult(
        asset_name=asset_name,
        benchmark_name=benchmark_name,
        n_aligned_obs=n_obs,
        start_date=aligned.index.min().strftime("%Y-%m-%d"),
        end_date=aligned.index.max().strftime("%Y-%m-%d"),
        ols_alpha_daily=ols_alpha,
        ols_alpha_annualized=ols_alpha_ann,
        ols_alpha_pvalue=ols_alpha_p,
        ols_beta=ols_beta,
        ols_beta_se_hac=ols_beta_se,
        ols_beta_pvalue=ols_beta_p,
        ols_r_squared=ols_r2,
        ols_dw_stat=dw_val,
        ols_lb_pvalue=ols_lb_p,
        best_arma_order=best_order,
        arma_alpha_daily=arma_alpha,
        arma_alpha_annualized=arma_alpha_ann,
        arma_alpha_se=arma_alpha_se,
        arma_alpha_pvalue=arma_alpha_p,
        arma_alpha_ci=arma_alpha_ci,
        arma_beta=arma_beta,
        arma_beta_se=arma_beta_se,
        arma_beta_pvalue=arma_beta_p,
        arma_beta_ci=arma_beta_ci,
        arma_aicc=best_aicc,
        sarimax_fit=best_sarimax,
        interpretation=interpretation,
    )


def plot_market_regression(
    reg_res: MarketRegressionResult,
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    output_path: str | None = None,
) -> plt.Figure:
    """Plots the empirical return scatter, OLS fitted line, and SARIMAX model parameters."""
    aligned = pd.concat([asset_returns.rename("y"), benchmark_returns.rename("x")], axis=1, join="inner").dropna()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"ATSA Unit 2 & 5 Regression with ARMA Errors: {reg_res.asset_name} vs {reg_res.benchmark_name}\n"
        f"Beta = {reg_res.arma_beta:.3f} | Annual Alpha = {reg_res.arma_alpha_annualized*100:.2f}% (p={reg_res.arma_alpha_pvalue:.4f}) | R^2 = {reg_res.ols_r_squared:.3f}",
        fontsize=12,
        fontweight="bold",
    )

    # 1. Scatter and Fit
    x_vals = aligned["x"].values * 100
    y_vals = aligned["y"].values * 100
    axes[0].scatter(x_vals, y_vals, alpha=0.3, color="#1f77b4", s=15, edgecolors="none")

    x_line = np.linspace(np.min(x_vals), np.max(x_vals), 100)
    y_line = (reg_res.arma_alpha_daily * 100) + reg_res.arma_beta * x_line
    axes[0].plot(x_line, y_line, color="crimson", linewidth=2.0, label=f"SARIMAX{reg_res.best_arma_order} Line")
    axes[0].axhline(0, color="gray", linestyle=":", linewidth=0.8)
    axes[0].axvline(0, color="gray", linestyle=":", linewidth=0.8)
    axes[0].set_xlabel(f"{reg_res.benchmark_name} Daily Return (%)", fontweight="bold")
    axes[0].set_ylabel(f"{reg_res.asset_name} Daily Return (%)", fontweight="bold")
    axes[0].set_title("Empirical Return Scatter & Security Characteristic Line", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Residual Distribution and DW Check
    resid = reg_res.sarimax_fit.resid
    axes[1].hist(resid, bins=40, density=True, alpha=0.6, color="#4c72b0", edgecolor="black")
    mu, sigma = float(np.mean(resid)), float(np.std(resid))
    grid = np.linspace(mu - 3.5 * sigma, mu + 3.5 * sigma, 150)
    import scipy.stats as stats
    axes[1].plot(grid, stats.norm.pdf(grid, mu, sigma), color="crimson", linewidth=1.8, label="Normal Curve")
    axes[1].set_title(f"Model Residuals (OLS DW={reg_res.ols_dw_stat:.2f}, LB p={reg_res.ols_lb_pvalue:.4f})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Residual e_t")
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved market regression chart to '%s'", output_path)

    return fig
