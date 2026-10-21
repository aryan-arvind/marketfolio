"""
Unit tests for residual bootstrap simulation.
Verifies that bootstrap simulated median path matches analytic point forecasts.
"""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.tsa.arima.model import ARIMA

from risklens.volatility import simulate_residual_bootstrap


def test_residual_bootstrap_recovers_analytic_forecast():
    """
    Algorithm-validation test: On a synthetic stationary AR(1) series,
    verifies that the residual bootstrap median path closely tracks
    the statsmodels analytic point forecast within a tight margin.
    """
    rng = np.random.default_rng(42)
    n = 1500
    phi = 0.5
    const_mean = 2.5
    e = rng.standard_normal(n) * 0.2
    y = np.zeros(n)
    y[0] = const_mean
    for t in range(1, n):
        y[t] = const_mean + phi * (y[t - 1] - const_mean) + e[t]

    horizon = 10
    model = ARIMA(y, order=(1, 0, 0), trend="c")
    fit = model.fit()

    analytic_point = fit.get_forecast(steps=horizon).predicted_mean

    boot_lower, boot_median, boot_upper, failed, _ = simulate_residual_bootstrap(
        model_fit=fit,
        observed_log_vol=y,
        order=(1, 0, 0),
        horizon=horizon,
        b_replications=1500,
        seed=42,
    )

    assert not failed
    assert len(boot_median) == horizon
    # Bootstrap median should track analytic forecast closely (within 5%)
    rel_gap = np.abs((boot_median - analytic_point) / analytic_point)
    assert np.max(rel_gap) < 0.05
    # Bootstrap bounds should encompass the median
    assert (boot_lower <= boot_median).all()
    assert (boot_median <= boot_upper).all()
