"""
MarketFolio Advanced Machine Learning: Shared Financial-Data Pipeline.
Provides leak-free feature engineering, chronological train/val/test splitting
with embargo purging for overlapping horizons, train-only scalers, regime targets,
and benchmark baselines (Majority, Persistence, Logistic Regression, Z-Score, Isolation Forest).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from risklens.config import (
    ANNUALIZATION_FACTOR,
    ML_LOOKBACK_WINDOW,
    ML_RANDOM_SEED,
    ML_TEST_RATIO,
    ML_TRAIN_RATIO,
    ML_VAL_RATIO,
)

logger = logging.getLogger("risklens")


@dataclass
class RegimeThresholds:
    low_cutoff: float
    high_cutoff: float
    quantiles: tuple[float, float] = (0.333, 0.667)
    labels: tuple[str, str, str] = ("Low Volatility", "Normal Volatility", "High Volatility")


@dataclass
class DatasetSplits:
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    X_train_raw: pd.DataFrame
    X_val_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train_scaled: np.ndarray
    X_val_scaled: np.ndarray
    X_test_scaled: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler
    y_train: np.ndarray | None = None
    y_val: np.ndarray | None = None
    y_test: np.ndarray | None = None
    regime_thresholds: RegimeThresholds | None = None
    embargo_days: int = 0
    train_purge_dates: tuple[str, str] | None = None
    val_purge_dates: tuple[str, str] | None = None


class BaselineComparison(NamedTuple):
    name: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    predictions: np.ndarray
    description: str


def build_multivariate_feature_matrix(
    price_series: pd.Series,
    return_series: pd.Series,
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Constructs a rich, stationary feature matrix from primary price & returns,
    strictly utilizing causal historical information up to time t (zero lookahead).

    Features:
    1. log_return: Daily logarithmic return r_t.
    2. vol_5d: 5-day rolling annualized realized volatility (ln).
    3. vol_21d: 21-day rolling annualized realized volatility (ln).
    4. momentum_10d: 10-day cumulative return.
    5. momentum_50d: 50-day cumulative return.
    6. price_zscore_21d: 21-day Z-score of price relative to moving average.
    7. return_skew_21d: 21-day rolling skewness (tail asymmetry).
    8. return_kurt_21d: 21-day rolling kurtosis (fat-tail indicator).
    9. Optional macro covariates (NIFTY, USDINR, Brent, VIX) if aligned.
    """
    p = price_series.sort_index().copy()
    r = return_series.sort_index().copy()

    df = pd.DataFrame(index=r.index)
    df["log_return"] = r

    # Rolling realized volatility (annualized log form)
    r_sq = r ** 2
    vol_5 = np.sqrt(r_sq.rolling(5).mean() * 252.0).clip(lower=1e-4)
    vol_21 = np.sqrt(r_sq.rolling(21).mean() * 252.0).clip(lower=1e-4)
    df["vol_5d"] = np.log(vol_5)
    df["vol_21d"] = np.log(vol_21)

    # Momentum signals
    df["momentum_10d"] = r.rolling(10).sum()
    df["momentum_50d"] = r.rolling(50).sum()

    # Mean reversion / price distance
    ma_21 = p.rolling(21).mean()
    sd_21 = p.rolling(21).std().clip(lower=1e-6)
    df["price_zscore_21d"] = ((p - ma_21) / sd_21).reindex(r.index)

    # Higher moments
    df["return_skew_21d"] = r.rolling(21).skew().fillna(0.0)
    df["return_kurt_21d"] = r.rolling(21).kurt().fillna(0.0)

    # Macro series integration if provided
    if macro_df is not None and not macro_df.empty:
        common_idx = df.index.intersection(macro_df.index)
        if len(common_idx) > 200:
            aligned_macro = macro_df.loc[common_idx]
            for col in aligned_macro.columns:
                m_series = aligned_macro[col]
                # If series already represents returns (e.g. from align_macro_series with negative values), use directly
                if (m_series < 0).any():
                    df[f"macro_{col.lower()}_ret"] = m_series.reindex(df.index)
                elif "VIX" in col.upper():
                    df[f"macro_{col.lower()}"] = m_series.reindex(df.index)
                else:
                    ret = np.log(m_series.clip(lower=1e-4) / m_series.shift(1).clip(lower=1e-4))
                    df[f"macro_{col.lower()}_ret"] = ret.reindex(df.index)

    # Clean infinite values and drop warm-up NaN rows (from 50-day rolling window)
    df = df.replace([np.inf, -np.inf], np.nan)
    clean_df = df.dropna().copy()
    logger.info("Constructed feature matrix: %d rows, %d features.", len(clean_df), clean_df.shape[1])
    return clean_df


def construct_volatility_regime_targets(
    return_series: pd.Series,
    train_end_idx: int,
    horizon: int = 10,
    quantiles: tuple[float, float] = (0.333, 0.667),
) -> tuple[pd.Series, pd.Series, RegimeThresholds]:
    """
    Constructs future realized volatility over [t+1, t+H] and assigns discrete regimes:
    0: Low Volatility
    1: Normal Volatility
    2: High Volatility

    CRITICAL ACADEMIC INTEGRITY RULE:
    Quantile cutoffs are calibrated EXCLUSIVELY on the training slice (0 to train_end_idx).
    Silent full-series fallback is strictly prohibited.
    """
    r = return_series.sort_index().copy()
    
    # Forward rolling realized volatility over horizon H
    # forward_vol[t] = sqrt(252 / H * sum_{i=1}^H r_{t+i}^2)
    r_sq = r ** 2
    forward_sum = r_sq.iloc[::-1].rolling(window=horizon).sum().iloc[::-1].shift(-1)
    forward_vol = np.sqrt((forward_sum / horizon) * 252.0)

    # Drop tail observations where forward horizon is incomplete
    valid_vol = forward_vol.dropna()

    # STRICT VALIDATION: Require explicit, valid training boundary
    if train_end_idx is None or not isinstance(train_end_idx, int) or train_end_idx <= 0 or train_end_idx >= len(valid_vol):
        raise ValueError(
            f"train_end_idx must be an explicit positive integer strictly less than valid target length ({len(valid_vol)}), "
            f"got {train_end_idx}. Full-series threshold calculation is strictly prohibited to prevent lookahead leakage."
        )

    # Determine thresholds strictly on training subset
    train_slice = valid_vol.iloc[:train_end_idx]
    low_cutoff = float(train_slice.quantile(quantiles[0]))
    high_cutoff = float(train_slice.quantile(quantiles[1]))

    thresholds = RegimeThresholds(
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        quantiles=quantiles,
    )

    regimes = pd.Series(index=valid_vol.index, dtype=int)
    regimes[valid_vol <= low_cutoff] = 0
    regimes[(valid_vol > low_cutoff) & (valid_vol <= high_cutoff)] = 1
    regimes[valid_vol > high_cutoff] = 2

    logger.info(
        "Volatility Regimes calibrated on train slice (N=%d): Low <= %.2f%%, High > %.2f%%",
        len(train_slice), low_cutoff * 100.0, high_cutoff * 100.0
    )
    return valid_vol, regimes, thresholds


def create_chronological_splits(
    features_df: pd.DataFrame,
    target_series: pd.Series | None = None,
    horizon: int = 10,
    train_ratio: float = ML_TRAIN_RATIO,
    val_ratio: float = ML_VAL_RATIO,
) -> DatasetSplits:
    """
    Partitions features and optional target into Chronological Train, Validation, and Test sets.
    Implements a strict embargo/purge of length `horizon` between splits to eliminate
    forward-looking label overlap leakage.
    Scalers are fitted EXCLUSIVELY on the Training set.
    """
    # Align features and target
    if target_series is not None:
        common_idx = features_df.index.intersection(target_series.index)
        X_df = features_df.loc[common_idx]
        y_s = target_series.loc[common_idx]
    else:
        X_df = features_df.copy()
        y_s = None

    n_total = len(X_df)
    n_train_raw = int(n_total * train_ratio)
    n_val_raw = int(n_total * val_ratio)

    # 1. Train split
    train_idx = slice(0, n_train_raw)
    train_df = X_df.iloc[train_idx]
    train_dates = train_df.index

    # 2. Embargo after train (purge horizon days)
    purge_train_start = n_train_raw
    purge_train_end = min(n_train_raw + horizon, n_total)
    train_purge = (
        str(X_df.index[purge_train_start].date()),
        str(X_df.index[purge_train_end - 1].date()) if purge_train_end > purge_train_start else "",
    )

    # 3. Validation split
    val_start = purge_train_end
    val_end = min(val_start + n_val_raw, n_total)
    val_df = X_df.iloc[val_start:val_end]
    val_dates = val_df.index

    # 4. Embargo after validation (purge horizon days)
    purge_val_start = val_end
    purge_val_end = min(val_end + horizon, n_total)
    val_purge = (
        str(X_df.index[purge_val_start].date()) if purge_val_start < n_total else "",
        str(X_df.index[purge_val_end - 1].date()) if purge_val_end > purge_val_start and purge_val_end <= n_total else "",
    )

    # 5. Test split
    test_start = min(purge_val_end, n_total)
    test_df = X_df.iloc[test_start:]
    test_dates = test_df.index

    if len(test_df) < 20:
        raise ValueError(
            f"Insufficient test observations ({len(test_df)}) after applying horizon={horizon} embargo. "
            "Ensure total observation count is sufficient."
        )

    # Fit scaler strictly on Train features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df.values)
    X_val_scaled = scaler.transform(val_df.values)
    X_test_scaled = scaler.transform(test_df.values)

    y_train = y_s.iloc[train_idx].values if y_s is not None else None
    y_val = y_s.iloc[val_start:val_end].values if y_s is not None else None
    y_test = y_s.iloc[test_start:].values if y_s is not None else None

    logger.info(
        "Chronological Splits (Embargo=%d days): Train=%d, Purge1=%d, Val=%d, Purge2=%d, Test=%d",
        horizon, len(train_df), (purge_train_end - purge_train_start), len(val_df), (purge_val_end - purge_val_start), len(test_df)
    )

    splits = DatasetSplits(
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        X_train_raw=train_df,
        X_val_raw=val_df,
        X_test_raw=test_df,
        X_train_scaled=X_train_scaled,
        X_val_scaled=X_val_scaled,
        X_test_scaled=X_test_scaled,
        feature_names=list(X_df.columns),
        scaler=scaler,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        embargo_days=horizon,
        train_purge_dates=train_purge,
        val_purge_dates=val_purge,
    )
    return splits


def prepare_ml_dataset(
    features_df: pd.DataFrame,
    return_series: pd.Series,
    horizon: int = 10,
    train_ratio: float = ML_TRAIN_RATIO,
    val_ratio: float = ML_VAL_RATIO,
    quantiles: tuple[float, float] = (0.333, 0.667),
) -> tuple[DatasetSplits, RegimeThresholds]:
    """
    Unified end-to-end dataset preparation guaranteeing zero forward-looking leakage:
    1. Computes forward realized volatility over horizon H.
    2. Pre-aligns features and forward target on their exact common business day intersection.
    3. Derives the training boundary strictly as int(len(aligned) * train_ratio).
    4. Calibrates quantile regime cutoffs EXCLUSIVELY on aligned training observations.
    5. Discretizes forward regimes (0: Low, 1: Normal, 2: High).
    6. Partitions into Chronological Train, Val, Test with H-day embargo/purge gaps.
    7. Fits StandardScaler strictly on training features.
    """
    r = return_series.sort_index().copy()
    r_sq = r ** 2
    fwd_sum = r_sq.iloc[::-1].rolling(window=horizon).sum().iloc[::-1].shift(-1)
    fwd_vol = np.sqrt((fwd_sum / horizon) * 252.0).dropna()

    common_idx = features_df.index.intersection(fwd_vol.index)
    if len(common_idx) < 200:
        raise ValueError(f"Insufficient aligned feature-target observations ({len(common_idx)}).")

    # Limit to most recent 1,250 trading days (~5 years) for computational speed and regime relevance
    if len(common_idx) > 1250:
        common_idx = common_idx[-1250:]

    aligned_features = features_df.loc[common_idx]
    aligned_vol = fwd_vol.loc[common_idx]

    n_train = int(len(common_idx) * train_ratio)
    train_slice = aligned_vol.iloc[:n_train]

    low_cutoff = float(train_slice.quantile(quantiles[0]))
    high_cutoff = float(train_slice.quantile(quantiles[1]))
    thresholds = RegimeThresholds(
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        quantiles=quantiles,
    )

    regimes = pd.Series(index=aligned_vol.index, dtype=int)
    regimes[aligned_vol <= low_cutoff] = 0
    regimes[(aligned_vol > low_cutoff) & (aligned_vol <= high_cutoff)] = 1
    regimes[aligned_vol > high_cutoff] = 2

    splits = create_chronological_splits(
        features_df=aligned_features,
        target_series=regimes,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    splits.regime_thresholds = thresholds
    return splits, thresholds


def create_sliding_windows(
    X: np.ndarray,
    y: np.ndarray | None = None,
    lookback: int = ML_LOOKBACK_WINDOW,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Transforms 2D tabular features into 3D sliding sequence windows:
    Output shape: (num_windows, lookback, num_features).
    Returns (X_seq, y_seq, window_end_indices).
    """
    n_samples, n_features = X.shape
    if n_samples < lookback:
        raise ValueError(f"Sequence length {n_samples} is shorter than lookback window {lookback}.")

    num_windows = n_samples - lookback + 1
    X_windows = np.zeros((num_windows, lookback, n_features), dtype=np.float32)

    for i in range(num_windows):
        X_windows[i] = X[i : i + lookback]

    # Target corresponds to the label assigned at the end of the lookback window
    window_end_indices = np.arange(lookback - 1, n_samples)
    y_windows = y[window_end_indices] if y is not None else None

    return X_windows, y_windows, window_end_indices


# =========================================================================
# BASELINE EVALUATORS FOR VOLATILITY REGIMES
# =========================================================================

def evaluate_majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> BaselineComparison:
    """Naive majority class persistence baseline."""
    values, counts = np.unique(y_train, return_counts=True)
    majority_class = values[np.argmax(counts)]
    y_pred = np.full_like(y_test, fill_value=majority_class)

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    return BaselineComparison(
        name="Majority Class Baseline",
        accuracy=acc,
        balanced_accuracy=bal_acc,
        macro_f1=f1,
        predictions=y_pred,
        description=f"Always predicts majority training regime (Class {majority_class}).",
    )


def evaluate_persistence_baseline(
    return_series: pd.Series,
    test_dates: pd.DatetimeIndex,
    thresholds: RegimeThresholds,
    horizon: int = 10,
    y_test: np.ndarray | None = None,
) -> BaselineComparison:
    """
    Martingale Persistence Baseline:
    Predicts the future regime to be the regime observed over the preceding H days [t-H+1, t].
    """
    r_sq = (return_series.sort_index() ** 2)
    past_vol = np.sqrt((r_sq.rolling(horizon).mean()) * 252.0)

    # Align with test dates
    aligned_past_vol = past_vol.reindex(test_dates).fillna(thresholds.low_cutoff)
    y_pred = np.zeros(len(aligned_past_vol), dtype=int)
    y_pred[aligned_past_vol.values <= thresholds.low_cutoff] = 0
    y_pred[(aligned_past_vol.values > thresholds.low_cutoff) & (aligned_past_vol.values <= thresholds.high_cutoff)] = 1
    y_pred[aligned_past_vol.values > thresholds.high_cutoff] = 2

    # Truncate to y_test length if provided
    if y_test is not None:
        y_pred = y_pred[: len(y_test)]
        acc = float(accuracy_score(y_test, y_pred))
        bal_acc = float(balanced_accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    else:
        acc, bal_acc, f1 = 0.0, 0.0, 0.0

    return BaselineComparison(
        name="Last-Known Regime Persistence",
        accuracy=acc,
        balanced_accuracy=bal_acc,
        macro_f1=f1,
        predictions=y_pred,
        description="Predicts the volatility regime observed over the prior H-day window.",
    )


def evaluate_logistic_regression_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> BaselineComparison:
    """
    Classical Multiclass Logistic Regression Baseline on standardized tabular features.
    """
    clf = LogisticRegression(
        max_iter=1000,
        random_state=ML_RANDOM_SEED,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    return BaselineComparison(
        name="L2 Logistic Regression Baseline",
        accuracy=acc,
        balanced_accuracy=bal_acc,
        macro_f1=f1,
        predictions=y_pred,
        description="Classical multinomial logistic model with balanced class weights.",
    )
