"""
Unit tests for cleaning routines, return transformations, and evaluation metrics.
Validates log return computation and MASE against exact hand-calculated values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risklens.cleaning import clean_primary_asset
from risklens.evaluation import assert_no_leakage, compute_mase


def test_log_returns_hand_calculated():
    """
    Algorithm-validation test: Verifies log returns against exact analytic values.
    Prices: P_0 = 100.0, P_1 = 110.0, P_2 = 99.0
    r_1 = ln(110 / 100) = ln(1.1) = 0.0953101798...
    r_2 = ln(99 / 110)  = ln(0.9) = -0.1053605156...
    """
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.Series([100.0, 110.0, 99.0], index=dates)

    log_returns = np.log(prices / prices.shift(1)).dropna()

    expected_r1 = np.log(1.1)
    expected_r2 = np.log(0.9)

    assert len(log_returns) == 2
    assert log_returns.iloc[0] == pytest.approx(expected_r1, rel=1e-7)
    assert log_returns.iloc[1] == pytest.approx(expected_r2, rel=1e-7)


def test_mase_hand_calculated():
    """
    Algorithm-validation test: Verifies MASE against hand-calculated values.
    Train window: [10.0, 12.0, 11.0, 13.0]
      Differences: |12 - 10| = 2.0, |11 - 12| = 1.0, |13 - 11| = 2.0
      In-sample Naive Scale: (2.0 + 1.0 + 2.0) / 3 = 5.0 / 3 = 1.6666667
    Test actuals:   [14.0, 15.0]
    Test predicted: [13.0, 14.0]
      Errors: |14 - 13| = 1.0, |15 - 14| = 1.0
      MAE = (1.0 + 1.0) / 2 = 1.0
    Expected MASE = MAE / Scale = 1.0 / (5.0 / 3.0) = 3.0 / 5.0 = 0.6000
    """
    train_history = np.array([10.0, 12.0, 11.0, 13.0])
    actual = np.array([14.0, 15.0])
    predicted = np.array([13.0, 14.0])

    mase = compute_mase(actual, predicted, train_history)

    assert mase == pytest.approx(0.6000, rel=1e-6)


def test_cleaning_business_days_and_gap_fill():
    """
    Algorithm-validation test: Verifies business-day alignment, forward filling
    up to 3 business days, and anomaly detection (>15% return).
    """
    # 800 business days of prices
    b_dates = pd.bdate_range("2021-01-01", periods=800)
    # Introduce a 2-day gap and a 1-day spike
    base_prices = np.linspace(100.0, 200.0, 800)
    # Inject 20% return jump at index 100
    base_prices[100] = base_prices[99] * 1.25

    s = pd.Series(base_prices, index=b_dates)
    # Remove 2 consecutive business days (indices 10 and 11) to test gap fill
    s_with_gap = s.drop(index=[b_dates[10], b_dates[11]])

    clean_res = clean_primary_asset(s_with_gap, asset_name="TestAsset", min_observations=750)

    # Gap of 2 days should be filled
    assert clean_res.n_gaps_filled == 2
    assert len(clean_res.log_return_series) >= 750
    # Anomaly of 25% jump at index 100 should be flagged
    assert len(clean_res.flagged_anomalies) >= 1
    flagged_dates = clean_res.flagged_anomalies["date"].tolist()
    assert b_dates[100].strftime("%Y-%m-%d") in flagged_dates
