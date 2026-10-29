"""
ARMA/ARIMA identification, grid search estimation, and diagnostic validation (ATSA Units 3, 4, 5).
Implements AICc-based order selection over p, q in 0..3, robust parameter estimation,
residual white-noise diagnostics (Ljung-Box with adjusted df, Jarque-Bera, QQ plots),
and formal Weak-Form Efficient Market Hypothesis predictability assessment.
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
import scipy.stats as stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import acf

logger = logging.getLogger("risklens")


class ARMAModelSummary(NamedTuple):
    """Detailed parameters and metrics for a fitted ARIMA model."""
    order: tuple[int, int, int]
    aic: float
    aicc: float
    bic: float
    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    conf_int: pd.DataFrame
    lb_df: pd.DataFrame
    jb_stat: float
    jb_pvalue: float
    jb_skew: float
    jb_kurtosis: float
    residuals: pd.Series
    fittedvalues: pd.Series
    model_result: Any


class GridSearchResult(NamedTuple):
    """Results from the ARIMA (p, d, q) model search."""
    best_order: tuple[int, int, int]
    best_model: ARMAModelSummary
    constant_model: ARMAModelSummary
    all_models_df: pd.DataFrame
    failed_orders: list[tuple[int, int, int]]
    predictability_verdict: str
    is_predictable: bool


def calculate_aicc(aic: float, nobs: int, k_params: int) -> float:
    """
    Computes small-sample corrected Akaike Information Criterion (AICc) (Hurvich & Tsai 1989):
    AICc = AIC + 2k(k+1) / (n - k - 1)
    """
    if nobs - k_params - 1 <= 0:
        return float("inf")
    correction = (2.0 * k_params * (k_params + 1.0)) / (nobs - k_params - 1.0)
    return float(aic + correction)


def fit_single_arima(
    series: pd.Series,
    order: tuple[int, int, int],
    trend: str | None = "c",
) -> ARMAModelSummary:
    """
    Fits an ARIMA(p, d, q) model to a RangeIndex series, extracts coefficients,
    and conducts residual diagnostic tests.
    """
    p, d, q = order
    actual_trend = "c" if (d == 0 and trend == "c") else None

    # Handle RangeIndex series to avoid freq warnings
    clean_vals = series.dropna().to_numpy()
    nobs = len(clean_vals)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        model = ARIMA(clean_vals, order=(p, d, q), trend=actual_trend)
        res: ARIMAResults = model.fit(method_kwargs={"maxiter": 25, "disp": False})

    k_params = len(res.params)
    aic = float(res.aic)
    aicc = calculate_aicc(aic, nobs, k_params)
    bic = float(res.bic)

    # Residuals
    residuals = pd.Series(res.resid, index=series.index)
    fitted = pd.Series(res.fittedvalues, index=series.index)

    # Ljung-Box test on residuals with model_df = p + q
    model_df = p + q
    # Ensure test lags exceed model_df
    test_lags = [10, 20] if 10 > model_df else [model_df + 2, model_df + 10]
    lb_df = acorr_ljungbox(res.resid, lags=test_lags, model_df=model_df, return_df=True)

    # Jarque-Bera test on residuals
    jb_res = jarque_bera(res.resid)
    jb_stat = float(jb_res[0])
    jb_pvalue = float(jb_res[1])
    jb_skew = float(jb_res[2])
    jb_kurt = float(jb_res[3])

    param_names = res.param_names
    params_s = pd.Series(res.params, index=param_names)
    bse_s = pd.Series(res.bse, index=param_names)
    pval_s = pd.Series(res.pvalues, index=param_names)
    ci_df = pd.DataFrame(res.conf_int(), index=param_names, columns=["lower_95", "upper_95"])

    return ARMAModelSummary(
        order=order,
        aic=aic,
        aicc=aicc,
        bic=bic,
        params=params_s,
        bse=bse_s,
        pvalues=pval_s,
        conf_int=ci_df,
        lb_df=lb_df,
        jb_stat=jb_stat,
        jb_pvalue=jb_pvalue,
        jb_skew=jb_skew,
        jb_kurtosis=jb_kurt,
        residuals=residuals,
        fittedvalues=fitted,
        model_result=res,
    )


def grid_search_arma(
    series: pd.Series,
    asset_name: str,
    d: int = 0,
    max_p: int = 2,
    max_q: int = 2,
) -> GridSearchResult:
    """
    Performs grid search over p in 0..max_p and q in 0..max_q.
    Selects best model via lowest AICc.
    Compares against constant-mean model ARIMA(0, d, 0) to evaluate return predictability.
    """
    records: list[dict[str, Any]] = []
    models_dict: dict[tuple[int, int, int], ARMAModelSummary] = {}
    failed_orders: list[tuple[int, int, int]] = []

    search_series = series.iloc[-500:] if len(series) > 500 else series

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            order = (p, d, q)
            try:
                mod_summary = fit_single_arima(search_series, order, trend="c")
                models_dict[order] = mod_summary
                records.append(
                    {
                        "p": p,
                        "d": d,
                        "q": q,
                        "order": f"({p},{d},{q})",
                        "aic": mod_summary.aic,
                        "aicc": mod_summary.aicc,
                        "bic": mod_summary.bic,
                        "jb_pvalue": mod_summary.jb_pvalue,
                        "lb_pvalue_lag10": mod_summary.lb_df.iloc[0]["lb_pvalue"] if len(mod_summary.lb_df) > 0 else np.nan,
                    }
                )
            except Exception as exc:
                logger.debug("ARIMA%s failed to converge or fit: %s", order, exc)
                failed_orders.append(order)

    if not records:
        raise RuntimeError(f"All ARIMA candidate models failed to fit for {asset_name}.")

    df_models = pd.DataFrame(records).sort_values(by="aicc").reset_index(drop=True)
    best_row = df_models.iloc[0]
    best_order = (int(best_row["p"]), int(best_row["d"]), int(best_row["q"]))
    
    # Fit best model on search series for responsive sub-second computation
    best_model = fit_single_arima(search_series, best_order, trend="c")
    models_dict[best_order] = best_model

    # Baseline constant-mean model: ARIMA(0, d, 0) on search series
    const_order = (0, d, 0)
    const_model = fit_single_arima(search_series, const_order, trend="c")
    models_dict[const_order] = const_model

    # Predictability Verdict (Fama 1970 Weak-Form EMH)
    # A model beats constant mean if delta_aicc < -2.0 and residual LB indicates white noise
    delta_aicc = best_model.aicc - const_model.aicc
    first_lb_p = best_model.lb_df.iloc[0]["lb_pvalue"] if len(best_model.lb_df) > 0 else 0.0
    is_resid_white_noise = first_lb_p > 0.05

    # Check if AR/MA coefficients are statistically significant (p < 0.05)
    ar_ma_pvals = [v for k, v in best_model.pvalues.items() if "ar." in k or "ma." in k]
    has_sig_ar_ma = any(p < 0.05 for p in ar_ma_pvals) if ar_ma_pvals else False

    is_predictable = (delta_aicc < -2.0) and is_resid_white_noise and has_sig_ar_ma

    if is_predictable:
        verdict = (
            f"Statistically significant predictability detected: Best model ARIMA{best_order} outperforms "
            f"the constant-mean benchmark (Delta AICc = {delta_aicc:.2f} < -2.0) with clean residuals "
            f"(Ljung-Box p = {first_lb_p:.4f} > 0.05) and significant AR/MA coefficients."
        )
    else:
        verdict = (
            f"No statistically significant return predictability: Best model ARIMA{best_order} does not "
            f"meaningfully outperform a constant-mean random walk benchmark (Delta AICc = {delta_aicc:.2f}). "
            f"Consistent with the Weak-Form Efficient Market Hypothesis (Fama 1970); conditional mean returns "
            "are white noise around an unconditional drift."
        )

    logger.info("ARMA Grid Search for '%s': Selected ARIMA%s (AICc=%.2f) -> %s", asset_name, best_order, best_model.aicc, verdict)

    return GridSearchResult(
        best_order=best_order,
        best_model=best_model,
        constant_model=const_model,
        all_models_df=df_models,
        failed_orders=failed_orders,
        predictability_verdict=verdict,
        is_predictable=is_predictable,
    )


def plot_arma_residual_diagnostics(
    model_summary: ARMAModelSummary,
    asset_name: str,
    output_path: str | None = None,
) -> plt.Figure:
    """
    Generates standard ATSA residual diagnostic plots:
    1. Residual time series path
    2. Residual histogram with normal KDE fit
    3. Normal QQ plot
    4. Residual ACF plot (lags 1 to 25)
    """
    resid = model_summary.residuals.dropna().values
    n = len(resid)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Model Residual Diagnostics: {asset_name} | ARIMA{model_summary.order}\n"
        f"Jarque-Bera p={model_summary.jb_pvalue:.4f} | Ljung-Box Lag10 p={model_summary.lb_df.iloc[0]['lb_pvalue']:.4f}",
        fontsize=13,
        fontweight="bold",
    )

    # 1. Residual time path
    axes[0, 0].plot(resid, color="#2b5c8f", linewidth=0.8)
    axes[0, 0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Standardized Residuals e_t", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Observation Index")
    axes[0, 0].set_ylabel("Residual")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # 2. Histogram and KDE vs Normal
    axes[0, 1].hist(resid, bins=50, density=True, alpha=0.6, color="#4c72b0", edgecolor="black")
    mu, sigma = float(np.mean(resid)), float(np.std(resid))
    x_grid = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    axes[0, 1].plot(x_grid, stats.norm.pdf(x_grid, mu, sigma), "crimson", linewidth=2.0, label="Gaussian N(mu, sigma)")
    axes[0, 1].set_title(f"Residual Distribution (Skew={model_summary.jb_skew:.2f}, Kurt={model_summary.jb_kurtosis:.2f})", fontsize=11, fontweight="bold")
    axes[0, 1].legend(loc="upper right", frameon=True)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # 3. Normal QQ Plot
    stats.probplot(resid, dist="norm", plot=axes[1, 0])
    axes[1, 0].get_lines()[0].set_markerfacecolor("#2b5c8f")
    axes[1, 0].get_lines()[0].set_markersize(4.0)
    axes[1, 0].get_lines()[0].set_alpha(0.6)
    axes[1, 0].get_lines()[1].set_color("crimson")
    axes[1, 0].get_lines()[1].set_linewidth(1.8)
    axes[1, 0].set_title("Normal Q-Q Plot", fontsize=11, fontweight="bold")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # 4. Residual ACF
    acf_resid = acf(resid, nlags=25, alpha=0.05)
    acf_vals = acf_resid[0][1:]  # Exclude lag 0
    conf_band = 1.96 / np.sqrt(n)
    axes[1, 1].stem(range(1, 26), acf_vals, basefmt="k-")
    axes[1, 1].axhline(0, color="gray", linestyle="-", linewidth=0.8)
    axes[1, 1].axhline(conf_band, color="crimson", linestyle="--", linewidth=1.2, label=f"95% CI (+/-{conf_band:.3f})")
    axes[1, 1].axhline(-conf_band, color="crimson", linestyle="--", linewidth=1.2)
    axes[1, 1].set_title("Residual ACF (Lags 1-25)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Lag")
    axes[1, 1].set_ylabel("Autocorrelation")
    axes[1, 1].legend(loc="upper right", frameon=True)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Saved ARMA residual diagnostics chart to '%s'", output_path)

    return fig
