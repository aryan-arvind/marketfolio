"""
Unit tests for ARMA model estimation and order selection.
Verifies algorithm recovery of true AR(1) order from a known synthetic process.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risklens.arma_models import grid_search_arma


def test_arma_order_selection_recovers_known_ar1():
    """
    Algorithm-validation test: Generates a synthetic AR(1) with phi=0.6 and N=3000
    using a fixed seed, and asserts that grid search over p, q in 0..3 selects order (1, 0, 0).
    """
    rng = np.random.default_rng(0)
    n = 3000
    phi = 0.6
    e = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]

    series = pd.Series(y, index=pd.RangeIndex(n))

    # Grid search over p in 0..3, q in 0..3 with d=0
    search_res = grid_search_arma(series, "Synthetic AR(1)", d=0, max_p=3, max_q=3)

    best_p, best_d, best_q = search_res.best_order
    # Must recover true AR(1) order (p=1, q=0)
    assert best_p == 1
    assert best_d == 0
    assert best_q == 0

    # Best model parameter for ar.L1 should be close to 0.6
    param_names = list(search_res.best_model.params.index)
    ar1_idx = param_names.index("ar.L1")
    estimated_phi = search_res.best_model.params.iloc[ar1_idx]
    assert estimated_phi == pytest.approx(0.6, abs=0.08)
