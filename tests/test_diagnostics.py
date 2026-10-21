"""
Unit tests for ATSA diagnostics, leakage assertions, and unit root test logic.
Uses synthetic benchmarks with known ground-truth parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risklens.diagnostics import run_unit_root_tests
from risklens.evaluation import assert_no_leakage


def test_assert_no_leakage():
    """
    Algorithm-validation test: Verifies that assert_no_leakage validates proper
    strictly increasing, non-overlapping train and test index partitions,
    and raises AssertionError when temporal leakage occurs.
    """
    train_valid = np.arange(0, 100)
    test_valid = np.arange(100, 121)

    # Should pass without error
    assert_no_leakage(train_valid, test_valid)

    # Overlap test: test starts before train finishes
    train_overlap = np.arange(0, 105)
    test_overlap = np.arange(100, 121)
    with pytest.raises(AssertionError, match="DATA LEAKAGE DETECTED"):
        assert_no_leakage(train_overlap, test_overlap)

    # Inversion test: test precedes train
    train_rev = np.arange(120, 200)
    test_rev = np.arange(50, 100)
    with pytest.raises(AssertionError, match="TEMPORAL LEAKAGE DETECTED"):
        assert_no_leakage(train_rev, test_rev)


def test_unit_root_on_synthetic_series():
    """
    Algorithm-validation test:
    1. Random Walk: y_t = y_{t-1} + e_t (Known Unit Root / Non-Stationary).
       ADF should fail to reject H0 (p >= 0.05) or KPSS should reject H0 (p < 0.05).
    2. AR(1) with phi=0.6: y_t = 0.6 * y_{t-1} + e_t (Known Stationary).
       ADF should reject H0 (p < 0.05) and KPSS should fail to reject H0 (p >= 0.05).
    """
    rng = np.random.default_rng(42)
    n = 2000

    # 1. Synthetic Random Walk
    innovations = rng.standard_normal(n)
    random_walk = pd.Series(np.cumsum(innovations))
    rw_res = run_unit_root_tests(random_walk, "Synthetic Random Walk")

    # A random walk must NOT be identified as "Stationary (Agreement)"
    assert rw_res.verdict != "Stationary (Agreement)"
    assert "Unit Root" in rw_res.verdict or rw_res.adf_pvalue > 0.05

    # 2. Synthetic Stationary AR(1) with phi = 0.6
    phi = 0.6
    ar1_vals = np.zeros(n)
    for t in range(1, n):
        ar1_vals[t] = phi * ar1_vals[t - 1] + innovations[t]
    ar1_series = pd.Series(ar1_vals)
    ar1_res = run_unit_root_tests(ar1_series, "Synthetic AR(1)")

    # AR(1) with phi=0.6 must be recognized as stationary
    assert ar1_res.adf_pvalue < 0.01
    assert ar1_res.verdict == "Stationary (Agreement)"
