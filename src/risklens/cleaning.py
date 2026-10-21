"""
Time series cleaning, business day calendar alignment, return calculations,
and anomaly detection for RiskLens.
Adheres strictly to the ATSA cleaning specification.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple, Tuple

import numpy as np
import pandas as pd

from risklens.config import (
    ANOMALY_ABS_RETURN_THRESHOLD,
    MAX_BUSINESS_DAY_GAP_FILL,
    MIN_MACRO_OBSERVATIONS,
    MIN_PRIMARY_OBSERVATIONS,
)

logger = logging.getLogger("risklens")


class CleanedSeriesResult(NamedTuple):
    """Container for cleaned primary asset prices, returns, and metadata."""
    price_series: pd.Series
    log_return_series: pd.Series
    modeling_series: pd.Series
    dates: np.ndarray
    flagged_anomalies: pd.DataFrame
    n_original_obs: int
    n_cleaned_obs: int
    n_gaps_filled: int
    n_dropped_after_gap: int


def clean_primary_asset(
    raw_series: pd.Series,
    asset_name: str = "Asset",
    min_observations: int = MIN_PRIMARY_OBSERVATIONS,
) -> CleanedSeriesResult:
    """
    Cleans a primary price or NAV series:
    1. Reindexes to complete business days (pd.bdate_range).
    2. Forward-fills gaps of at most MAX_BUSINESS_DAY_GAP_FILL (3) days.
    3. Drops remaining NaN gaps.
    4. Computes log returns r_t = ln(P_t / P_{t-1}).
    5. Flags large returns |r_t| > 0.15 as potential corporate-action / data artefacts.
    6. Verifies that the cleaned sample size meets the ATSA minimum threshold (>= 750).
    7. Produces a RangeIndex series for statsmodels compatibility alongside an explicit dates array.
    """
    n_orig = len(raw_series)
    if n_orig < min_observations:
        raise ValueError(
            f"Asset '{asset_name}' has only {n_orig} raw observations, which is below "
            f"the required minimum of {min_observations}. Aborting analysis."
        )

    # Ensure index is sorted DatetimeIndex
    s = raw_series.sort_index().copy()
    start_date = s.index.min()
    end_date = s.index.max()

    # Generate business day range
    b_calendar = pd.bdate_range(start=start_date, end=end_date, freq="B")
    reindexed = s.reindex(b_calendar)

    # Count gaps before fill
    initial_nans = int(reindexed.isna().sum())

    # Forward fill limited to 3 days
    filled = reindexed.ffill(limit=MAX_BUSINESS_DAY_GAP_FILL)
    nans_after_fill = int(filled.isna().sum())
    n_gaps_filled = initial_nans - nans_after_fill

    # Drop any remaining unfillable gaps
    cleaned_price = filled.dropna()
    n_dropped = nans_after_fill

    logger.info(
        "Cleaned '%s': original obs=%d, bday range=%d, filled gaps=%d, dropped remaining=%d, final clean obs=%d",
        asset_name, n_orig, len(b_calendar), n_gaps_filled, n_dropped, len(cleaned_price)
    )

    if len(cleaned_price) < min_observations:
        raise ValueError(
            f"Asset '{asset_name}' has {len(cleaned_price)} observations after business day "
            f"cleaning and gap dropping, falling below required minimum of {min_observations}."
        )

    # Compute log returns: r_t = ln(P_t / P_{t-1})
    log_returns = np.log(cleaned_price / cleaned_price.shift(1)).dropna()

    # Flag absolute log returns > 15%
    anomaly_mask = log_returns.abs() > ANOMALY_ABS_RETURN_THRESHOLD
    flagged_indices = log_returns.index[anomaly_mask]
    flagged_df = pd.DataFrame(
        {
            "date": flagged_indices.strftime("%Y-%m-%d"),
            "log_return": log_returns.loc[anomaly_mask].values,
            "pct_return": (np.exp(log_returns.loc[anomaly_mask].values) - 1.0) * 100.0,
            "price": cleaned_price.loc[flagged_indices].values,
        }
    )

    if not flagged_df.empty:
        logger.warning(
            "Flagged %d potential return anomalies (|log return| > 15%%) for '%s':\n%s",
            len(flagged_df), asset_name, flagged_df.to_string(index=False)
        )
    else:
        logger.info("No return anomalies exceeding 15%% detected for '%s'.", asset_name)

    # Prepare statsmodels modeling series with RangeIndex
    dates = log_returns.index.to_numpy()
    modeling_series = pd.Series(log_returns.values, index=pd.RangeIndex(len(log_returns)))

    return CleanedSeriesResult(
        price_series=cleaned_price,
        log_return_series=log_returns,
        modeling_series=modeling_series,
        dates=dates,
        flagged_anomalies=flagged_df,
        n_original_obs=n_orig,
        n_cleaned_obs=len(log_returns),
        n_gaps_filled=n_gaps_filled,
        n_dropped_after_gap=n_dropped,
    )


def align_macro_series(
    macro_dict: dict[str, pd.Series],
    min_obs: int = MIN_MACRO_OBSERVATIONS,
) -> tuple[pd.DataFrame, bool, str]:
    """
    Transforms macro price/index series into stationary return representations
    and aligns them across common dates via inner join.
    - S&P 500, Brent, USD/INR, NIFTY 50 -> Log returns
    - India VIX -> Log differences: ln(VIX_t) - ln(VIX_{t-1})
    
    Returns:
        (aligned_df, is_valid, reason)
    """
    expected_keys = ["SP500", "Brent", "USDINR", "IndiaVIX", "NIFTY50"]
    missing = [k for k in expected_keys if k not in macro_dict or macro_dict[k].empty]
    if missing:
        msg = f"Missing required macro series: {missing}. Macro/VAR analysis cannot proceed."
        logger.warning(msg)
        return pd.DataFrame(), False, msg

    transformed: dict[str, pd.Series] = {}
    for key in ["SP500", "Brent", "USDINR", "NIFTY50"]:
        s = macro_dict[key].sort_index()
        ret = np.log(s / s.shift(1)).dropna()
        transformed[key] = ret

    # India VIX log difference
    vix = macro_dict["IndiaVIX"].sort_index()
    vix_ret = np.log(vix / vix.shift(1)).dropna()
    transformed["IndiaVIX"] = vix_ret

    # Inner join on common dates
    aligned_df = pd.DataFrame(transformed).dropna()

    if len(aligned_df) < min_obs:
        msg = (
            f"Aligned macro dataset has only {len(aligned_df)} observations across common dates, "
            f"which is fewer than the required {min_obs}. Differing international trading holidays "
            "and truncated data spans constrain the sample."
        )
        logger.warning(msg)
        return aligned_df, False, msg

    logger.info(
        "Successfully aligned 5 macro series across %d common dates (%s to %s).",
        len(aligned_df),
        aligned_df.index.min().strftime("%Y-%m-%d"),
        aligned_df.index.max().strftime("%Y-%m-%d"),
    )
    return aligned_df, True, "Aligned successfully."


def map_range_to_dates(values: np.ndarray | pd.Series, dates: np.ndarray) -> pd.Series:
    """Utility to map modeling outputs from a RangeIndex back to DatetimeIndex for plotting."""
    if len(values) != len(dates):
        raise ValueError(f"Length mismatch: {len(values)} values vs {len(dates)} dates.")
    return pd.Series(values, index=pd.to_datetime(dates))
