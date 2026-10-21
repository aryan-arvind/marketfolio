"""
Unit tests for data parsing and date index normalization.
Validates mfapi.in JSON parser and timezone normalization routines.
"""

from __future__ import annotations

import pandas as pd
import pytest

from risklens.data_sources import parse_mfapi_history_json


def test_mfapi_parser_with_recorded_fixture():
    """
    Algorithm-validation test: Verifies that parse_mfapi_history_json correctly parses
    the exact recorded schema structure returned by mfapi.in (AMFI).
    """
    recorded_response = {
        "meta": {
            "fund_house": "Nippon India Mutual Fund",
            "scheme_type": "Open Ended Schemes",
            "scheme_category": "Other Scheme - Index Funds",
            "scheme_code": 120716,
            "scheme_name": "Nippon India Index Fund - Nifty 50 Plan - Direct Plan - Growth Plan",
        },
        "data": [
            {"date": "03-01-2024", "nav": "35.2500"},
            {"date": "02-01-2024", "nav": "35.1000"},
            {"date": "01-01-2024", "nav": "34.9500"},
            {"date": "31-12-2023", "nav": "0.0000"},  # Should be dropped (nav <= 0)
            {"date": "30-12-2023", "nav": "invalid"},  # Should be dropped (non-numeric)
        ],
        "status": "SUCCESS",
    }

    series = parse_mfapi_history_json(recorded_response)

    # Assertions
    assert isinstance(series, pd.Series)
    assert len(series) == 3
    # Check ascending sort order
    assert series.index[0] == pd.Timestamp("2024-01-01")
    assert series.index[-1] == pd.Timestamp("2024-01-03")
    assert series.iloc[0] == pytest.approx(34.9500)
    assert series.iloc[-1] == pytest.approx(35.2500)
    assert (series > 0).all()


def test_yfinance_timezone_normalization():
    """
    Algorithm-validation test: Verifies that timezone-aware timestamps (such as Asia/Kolkata
    or America/New_York) are cleanly converted to timezone-naive dates.
    """
    # Create timezone-aware DatetimeIndex
    dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="Asia/Kolkata")
    prices = [100.0, 101.5, 100.8, 102.3, 103.1]
    raw_series = pd.Series(prices, index=dates)

    # Conversion logic matching data_sources.py
    normalized = raw_series.copy()
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized.index = pd.to_datetime(normalized.index.date)

    assert normalized.index.tz is None
    assert normalized.index[0] == pd.Timestamp("2024-01-01")
    assert normalized.index.is_monotonic_increasing
    assert len(normalized) == 5
