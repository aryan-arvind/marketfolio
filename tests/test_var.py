"""
Unit tests for Vector Autoregression (VAR) stability and Granger causality.
Verifies algorithm detection of causal linkages on a synthetic bivariate system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.api import VAR


def test_var_stability_and_granger_causality_synthetic():
    """
    Algorithm-validation test: Constructs a bivariate VAR(1) system where X causes Y:
      X_t = 0.40 * X_{t-1} + e_{1,t}
      Y_t = 0.50 * X_{t-1} + 0.30 * Y_{t-1} + e_{2,t}
    Asserts that:
    1. The VAR system is mathematically stable (roots inside unit circle).
    2. Granger-causality test rejects H0 (X causes Y, p < 0.05).
    3. Reverse Granger-causality test fails to reject H0 (Y does not cause X, p >= 0.05).
    """
    rng = np.random.default_rng(42)
    n = 1500
    dates = pd.bdate_range("2020-01-01", periods=n)

    e1 = rng.standard_normal(n)
    e2 = rng.standard_normal(n)

    x = np.zeros(n)
    y = np.zeros(n)

    for t in range(1, n):
        x[t] = 0.40 * x[t - 1] + e1[t]
        y[t] = 0.50 * x[t - 1] + 0.30 * y[t - 1] + e2[t]

    df = pd.DataFrame({"X": x, "Y": y}, index=dates)

    var_model = VAR(df)
    var_fit = var_model.fit(1)

    # 1. Stability check
    assert var_fit.is_stable()

    # 2. Granger Causality: X causes Y
    g_x_causes_y = var_fit.test_causality("Y", "X", kind="f")
    assert g_x_causes_y.pvalue < 0.001  # Highly significant causality

    # 3. Granger Causality: Y causes X (should not be significant)
    g_y_causes_x = var_fit.test_causality("X", "Y", kind="f")
    assert g_y_causes_x.pvalue > 0.05  # Non-causal
