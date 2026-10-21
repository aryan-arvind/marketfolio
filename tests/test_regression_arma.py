"""
Unit tests for regression with ARMA errors.
Verifies algorithm recovery of true market beta from a synthetic process with ARMA errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risklens.regression_arma import fit_regression_arma


def test_regression_arma_recovers_known_beta():
    """
    Algorithm-validation test: Generates synthetic benchmark X_t and asset y_t:
    y_t = alpha + beta * X_t + eta_t
    where beta = 1.2, alpha = 0.0002, and eta_t is an AR(1) error process with phi=0.4.
    Asserts that fit_regression_arma recovers beta within close tolerance.
    """
    rng = np.random.default_rng(42)
    n = 1200
    dates = pd.bdate_range("2020-01-01", periods=n)

    # Benchmark returns X_t
    X = rng.standard_normal(n) * 0.015

    # Autocorrelated errors eta_t ~ AR(1)
    phi = 0.4
    e = rng.standard_normal(n) * 0.008
    eta = np.zeros(n)
    for t in range(1, n):
        eta[t] = phi * eta[t - 1] + e[t]

    true_alpha = 0.0002
    true_beta = 1.20
    y = true_alpha + true_beta * X + eta

    asset_series = pd.Series(y, index=dates)
    bench_series = pd.Series(X, index=dates)

    res = fit_regression_arma(asset_series, bench_series, asset_name="SyntheticAsset", benchmark_name="SyntheticBench")

    # Estimated beta should be close to 1.20
    assert res.arma_beta == pytest.approx(true_beta, abs=0.08)
    assert res.ols_beta == pytest.approx(true_beta, abs=0.08)
    # p-value for beta should be highly significant (< 0.001)
    assert res.arma_beta_pvalue < 0.001
