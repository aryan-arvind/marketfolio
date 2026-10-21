"""
Out-of-sample time series evaluation and expanding-window backtesting (ATSA Unit 4 & 5).
Implements strictly time-ordered rolling-origin forecasting without look-ahead bias,
formal data leakage assertions, and standardized metric evaluation (MAE, RMSE, MASE).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from risklens.config import (
    DEFAULT_FORECAST_HORIZON,
    EVAL_INITIAL_WINDOW,
    EVAL_MAX_ORIGINS,
    EVAL_STEP_SIZE,
)

logger = logging.getLogger("risklens")


class RollingOriginMetrics(NamedTuple):
    """Evaluation summary metrics for a model and its competing baselines."""
    model_name: str
    mae: float
    rmse: float
    mase: float
    mase_improvement_pct: float


class BacktestResult(NamedTuple):
    """Comprehensive container for out-of-sample backtesting results."""
    target_name: str
    n_origins_evaluated: int
    failed_origins_count: int
    horizon: int
    initial_window: int
    step_size: int
    metrics_table: pd.DataFrame
    model_metrics: RollingOriginMetrics
    naive_metrics: RollingOriginMetrics
    rolling_mean_metrics: RollingOriginMetrics
    summary_verdict: str


def assert_no_leakage(train_idx: np.ndarray | list[int], test_idx: np.ndarray | list[int]) -> None:
    """
    Enforces strict temporal ordering and non-overlap between training and test sets.
    Raises AssertionError if test indices precede or overlap with train indices.
    """
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise AssertionError("Train and test index sets must be non-empty.")

    train_set = set(train_idx)
    test_set = set(test_idx)

    overlap = train_set.intersection(test_set)
    if overlap:
        raise AssertionError(f"DATA LEAKAGE DETECTED: Overlap found between train and test sets: {sorted(list(overlap))[:5]}")

    max_train = int(np.max(train_idx))
    min_test = int(np.min(test_idx))

    if max_train >= min_test:
        raise AssertionError(
            f"TEMPORAL LEAKAGE DETECTED: Max train index ({max_train}) is not strictly less "
            f"than min test index ({min_test}). Time series must preserve chronological causality."
        )


def compute_mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    train_history: np.ndarray,
) -> float:
    """
    Computes Mean Absolute Scaled Error (MASE) (Hyndman & Koehler 2006).
    Scales out-of-sample MAE by the in-sample one-step naive forecast MAE:
    Scale = (1 / (T - 1)) * sum_{t=2}^T |y_t - y_{t-1}|
    """
    if len(actual) == 0 or len(train_history) < 2:
        return float("nan")

    mae_test = float(np.mean(np.abs(actual - predicted)))

    # In-sample naive scale
    naive_diffs = np.abs(np.diff(train_history))
    scale = float(np.mean(naive_diffs))

    if scale < 1e-12:
        return float("inf") if mae_test > 1e-12 else 1.0

    return float(mae_test / scale)


def run_rolling_origin_evaluation(
    series: pd.Series,
    target_name: str = "Realized Volatility",
    model_order: tuple[int, int, int] = (1, 0, 1),
    horizon: int = DEFAULT_FORECAST_HORIZON,
    initial_window: int = EVAL_INITIAL_WINDOW,
    step_size: int = EVAL_STEP_SIZE,
    max_origins: int = EVAL_MAX_ORIGINS,
) -> BacktestResult:
    """
    Executes expanding-window rolling-origin evaluation on a time series.
    At each origin t:
    1. Training set: indices 0 .. t-1
    2. Test set: indices t .. t+horizon-1
    3. Asserts no temporal or data leakage.
    4. Refits model and computes h-step path.
    5. Computes competing baseline forecasts (naive last value and rolling mean).
    6. Accumulates forecast errors across all origins and computes MAE, RMSE, MASE.
    """
    clean_vals = series.dropna().to_numpy()
    n_total = len(clean_vals)

    # Validate sufficient length
    min_required = initial_window + horizon
    if n_total < min_required:
        # Scale down initial window dynamically if series length is constrained
        initial_window = max(100, n_total - (horizon * 2))
        logger.warning(
            "Series length (%d) is shorter than standard initial window + horizon (%d). "
            "Adjusted initial window to %d.",
            n_total, min_required, initial_window
        )

    # Determine origin split points
    possible_origins = list(range(initial_window, n_total - horizon + 1, step_size))
    if len(possible_origins) > max_origins:
        # Subsample evenly to cap runtime
        indices = np.linspace(0, len(possible_origins) - 1, max_origins, dtype=int)
        origins = [possible_origins[i] for i in indices]
    else:
        origins = possible_origins

    if not origins:
        raise ValueError(
            f"Unable to construct rolling origins for '{target_name}' with length={n_total}, "
            f"initial_window={initial_window}, horizon={horizon}."
        )

    logger.info(
        "Starting rolling-origin evaluation for '%s': %d origins, horizon=%d, initial window=%d, step=%d",
        target_name, len(origins), horizon, initial_window, step_size
    )

    actuals: list[float] = []
    preds_model: list[float] = []
    preds_naive: list[float] = []
    preds_mean: list[float] = []
    train_scales: list[float] = []

    failed_origins = 0

    for origin in origins:
        train_idx = np.arange(0, origin)
        test_idx = np.arange(origin, origin + horizon)

        # Enforce temporal integrity
        assert_no_leakage(train_idx, test_idx)

        y_train = clean_vals[train_idx]
        y_test = clean_vals[test_idx]

        # In-sample naive scale for this training window
        scale = float(np.mean(np.abs(np.diff(y_train))))
        train_scales.append(scale)

        # 1. Model forecast (refit at each origin)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trend = "c" if model_order[1] == 0 else None
                mod = ARIMA(y_train, order=model_order, trend=trend)
                fit = mod.fit()
                pred_path = fit.forecast(steps=horizon)
        except Exception as exc:
            logger.debug("Fit failed at origin %d: %s. Using naive fallback.", origin, exc)
            failed_origins += 1
            pred_path = np.full(horizon, y_train[-1])

        # 2. Naive baseline: repeat last known observation
        naive_path = np.full(horizon, y_train[-1])

        # 3. Rolling-mean baseline: mean of last 21 training days
        window_len = min(21, len(y_train))
        mean_path = np.full(horizon, np.mean(y_train[-window_len:]))

        actuals.extend(y_test)
        preds_model.extend(pred_path)
        preds_naive.extend(naive_path)
        preds_mean.extend(mean_path)

    y_act = np.array(actuals)
    y_mod = np.array(preds_model)
    y_nai = np.array(preds_naive)
    y_mea = np.array(preds_mean)

    # Compute aggregate metrics
    mae_mod = float(np.mean(np.abs(y_act - y_mod)))
    rmse_mod = float(np.sqrt(np.mean((y_act - y_mod) ** 2)))

    mae_nai = float(np.mean(np.abs(y_act - y_nai)))
    rmse_nai = float(np.sqrt(np.mean((y_act - y_nai) ** 2)))

    mae_mea = float(np.mean(np.abs(y_act - y_mea)))
    rmse_mea = float(np.sqrt(np.mean((y_act - y_mea) ** 2)))

    # Global MASE denominator
    avg_scale = float(np.mean(train_scales)) if train_scales else 1.0
    mase_mod = mae_mod / avg_scale if avg_scale > 1e-12 else float("nan")
    mase_nai = mae_nai / avg_scale if avg_scale > 1e-12 else float("nan")
    mase_mea = mae_mea / avg_scale if avg_scale > 1e-12 else float("nan")

    # Percentage improvement relative to naive baseline
    # Negative improvement means better (lower MASE)
    pct_imp_mase = ((mase_nai - mase_mod) / mase_nai) * 100.0 if mase_nai > 1e-12 else 0.0
    pct_imp_mea = ((mase_nai - mase_mea) / mase_nai) * 100.0 if mase_nai > 1e-12 else 0.0

    mod_metrics = RollingOriginMetrics(
        model_name=f"ARIMA{model_order}",
        mae=mae_mod,
        rmse=rmse_mod,
        mase=mase_mod,
        mase_improvement_pct=pct_imp_mase,
    )

    nai_metrics = RollingOriginMetrics(
        model_name="Naive (Last Value)",
        mae=mae_nai,
        rmse=rmse_nai,
        mase=mase_nai,
        mase_improvement_pct=0.0,
    )

    mea_metrics = RollingOriginMetrics(
        model_name="Rolling Mean (21d)",
        mae=mae_mea,
        rmse=rmse_mea,
        mase=mase_mea,
        mase_improvement_pct=pct_imp_mea,
    )

    # Summary table
    table_df = pd.DataFrame(
        [
            {
                "Model": mod_metrics.model_name,
                "MAE": mod_metrics.mae,
                "RMSE": mod_metrics.rmse,
                "MASE": mod_metrics.mase,
                "MASE Improvement vs Naive (%)": f"{pct_imp_mase:+.2f}%",
            },
            {
                "Model": nai_metrics.model_name,
                "MAE": nai_metrics.mae,
                "RMSE": nai_metrics.rmse,
                "MASE": nai_metrics.mase,
                "MASE Improvement vs Naive (%)": "0.00% (Baseline)",
            },
            {
                "Model": mea_metrics.model_name,
                "MAE": mea_metrics.mae,
                "RMSE": mea_metrics.rmse,
                "MASE": mea_metrics.mase,
                "MASE Improvement vs Naive (%)": f"{pct_imp_mea:+.2f}%",
            },
        ]
    )

    if pct_imp_mase > 0:
        verdict = (
            f"ARIMA{model_order} demonstrates superior out-of-sample forecasting skill, "
            f"reducing MASE by {pct_imp_mase:.2f}% relative to the naive baseline across "
            f"{len(origins)} expanding-window origins."
        )
    else:
        verdict = (
            f"ARIMA{model_order} exhibits comparable or slightly weaker out-of-sample accuracy "
            f"to the naive random walk baseline (MASE change: {pct_imp_mase:+.2f}%). "
            "Reflecting the ATSA principle: financial time series often approach martingale differences "
            "where parameter estimation uncertainty attenuates theoretical out-of-sample gains."
        )

    logger.info("Evaluation complete for '%s': %s", target_name, verdict)

    return BacktestResult(
        target_name=target_name,
        n_origins_evaluated=len(origins),
        failed_origins_count=failed_origins,
        horizon=horizon,
        initial_window=initial_window,
        step_size=step_size,
        metrics_table=table_df,
        model_metrics=mod_metrics,
        naive_metrics=nai_metrics,
        rolling_mean_metrics=mea_metrics,
        summary_verdict=verdict,
    )
