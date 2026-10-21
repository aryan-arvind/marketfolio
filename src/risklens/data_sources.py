"""
Data ingestion and caching layer for RiskLens.
Fetches real AMFI mutual fund data via mfapi.in and real market/macro series via yfinance.
Implements disk caching with 6-hour TTL, automatic retries with exponential backoff,
stale cache fallback on network failure, and data validation.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
import requests
import yfinance as yf

from risklens.config import (
    CACHE_DIR,
    CACHE_TTL_SECONDS,
    DEFAULT_BENCHMARK_SYMBOL,
    HTTP_BACKOFF_FACTOR,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT_SECONDS,
    HTTP_USER_AGENT,
    MACRO_SYMBOLS,
    MF_HISTORY_URL,
    MF_SEARCH_URL,
    setup_logging,
)

logger = logging.getLogger("risklens")


def _get_cache_paths(cache_key: str) -> Tuple[Path, Path]:
    """Generates sanitized filesystem paths for cached CSV data and metadata JSON."""
    safe_key = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in cache_key)
    csv_path = CACHE_DIR / f"{safe_key}.csv"
    meta_path = CACHE_DIR / f"{safe_key}_meta.json"
    return csv_path, meta_path


def _load_from_cache(
    cache_key: str, force_refresh: bool = False
) -> Tuple[pd.Series | None, dict[str, Any] | None]:
    """
    Attempts to read cached series and metadata if present.
    Returns (series, metadata) if valid and not expired, or (series, metadata) marked as stale if expired.
    If no cache exists, returns (None, None).
    """
    csv_path, meta_path = _get_cache_paths(cache_key)
    if not (csv_path.exists() and meta_path.exists()):
        return None, None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        series = df["price"]
        series = series.sort_index()

        fetch_time = datetime.fromisoformat(metadata.get("fetch_timestamp", "1970-01-01T00:00:00+00:00"))
        age_seconds = (datetime.now(timezone.utc) - fetch_time).total_seconds()

        metadata["is_stale"] = age_seconds > CACHE_TTL_SECONDS

        if not force_refresh and not metadata["is_stale"]:
            logger.info("Loaded valid cached data for '%s' (age: %.1f hours)", cache_key, age_seconds / 3600.0)
            return series, metadata

        # Return expired cache in case caller needs to fall back if network fails
        return series, metadata
    except Exception as exc:
        logger.warning("Failed to read cache for '%s': %s", cache_key, exc)
        return None, None


def _save_to_cache(cache_key: str, series: pd.Series, metadata: dict[str, Any]) -> None:
    """Saves a price/NAV series to disk as CSV along with a JSON metadata sidecar."""
    csv_path, meta_path = _get_cache_paths(cache_key)
    try:
        df = series.to_frame(name="price")
        df.index.name = "date"
        df.to_csv(csv_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.debug("Successfully saved '%s' to cache", cache_key)
    except Exception as exc:
        logger.error("Failed to write cache for '%s': %s", cache_key, exc)


def _http_get_with_retry(url: str) -> requests.Response:
    """Executes an HTTP GET request with retries, exponential backoff, and custom User-Agent."""
    headers = {"User-Agent": HTTP_USER_AGENT, "Accept": "application/json"}
    last_error: Exception | None = None

    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response
        except (requests.RequestException, requests.Timeout) as exc:
            last_error = exc
            backoff = HTTP_BACKOFF_FACTOR ** (attempt - 1)
            logger.warning(
                "HTTP request to %s failed on attempt %d/%d (%s). Retrying in %.1fs...",
                url, attempt, HTTP_MAX_RETRIES, exc, backoff
            )
            time.sleep(backoff)

    raise ConnectionError(f"HTTP GET failed after {HTTP_MAX_RETRIES} attempts for URL {url}: {last_error}")


def search_mutual_fund(query: str) -> list[dict[str, Any]]:
    """
    Queries the mfapi.in search endpoint and returns a list of matching funds.
    Each item contains 'schemeCode' and 'schemeName'.
    """
    if not query.strip():
        raise ValueError("Fund query text cannot be empty.")

    url = f"{MF_SEARCH_URL}{requests.utils.quote(query)}"
    logger.info("Searching mutual funds with query: '%s'", query)
    response = _http_get_with_retry(url)
    try:
        data = response.json()
    except json.JSONDecodeError as err:
        raise ValueError(f"Malformed JSON returned by mfapi search for query '{query}': {err}")

    if not isinstance(data, list):
        raise ValueError(f"Unexpected response structure from mfapi search: expected list, got {type(data)}")

    return data


def resolve_mutual_fund(query: str) -> tuple[int, str]:
    """
    Resolves the target scheme code and scheme name from a search query.
    Prints candidates and applies priority logic:
    1. First candidate containing both 'Direct' and 'Growth' (case-insensitive).
    2. Otherwise, the first candidate.
    Logs warnings regarding IDCW / dividend plans.
    """
    candidates = search_mutual_fund(query)
    if not candidates:
        raise ValueError(f"No mutual funds found matching query '{query}'. Please refine your search.")

    logger.info("Found %d matching schemes for query '%s'. Top candidates:", len(candidates), query)
    print(f"\nFound {len(candidates)} matching schemes for '{query}'. Top candidates:")
    for idx, cand in enumerate(candidates[:10], start=1):
        print(f"  [{idx}] Code: {cand.get('schemeCode')} - {cand.get('schemeName')}")

    # Prioritize Direct and Growth plans
    selected = None
    for cand in candidates:
        name = cand.get("schemeName", "")
        name_lower = name.lower()
        if "direct" in name_lower and "growth" in name_lower:
            selected = cand
            break

    if selected is None:
        selected = candidates[0]
        logger.warning(
            "No 'Direct Growth' option matched. Selected first available candidate: %s (%s)",
            selected.get("schemeName"), selected.get("schemeCode")
        )
    else:
        logger.info(
            "Selected Direct Growth candidate: %s (%s)",
            selected.get("schemeName"), selected.get("schemeCode")
        )

    scheme_name = str(selected.get("schemeName", "Unknown Scheme"))
    scheme_code = int(selected.get("schemeCode"))

    # Dividend / IDCW check and warning
    name_lower = scheme_name.lower()
    if "dividend" in name_lower or "idcw" in name_lower:
        warning_msg = (
            f"WARNING: The selected fund '{scheme_name}' appears to be a Dividend/IDCW plan. "
            "Dividend payouts cause artificial drops in NAV that distort time series return calculations. "
            "A Growth plan is strongly recommended for accurate ATSA modeling."
        )
        logger.warning(warning_msg)
        print(f"\n{warning_msg}\n")

    return scheme_code, scheme_name


def parse_mfapi_history_json(raw_json: dict[str, Any]) -> pd.Series:
    """
    Parses a raw mfapi.in scheme JSON dictionary into a cleaned, sorted pd.Series.
    Expected keys: 'meta' (dict) and 'data' (list of {'date': 'dd-mm-yyyy', 'nav': '12.34'}).
    """
    if "data" not in raw_json or not isinstance(raw_json["data"], list):
        raise ValueError("Invalid mfapi history response: missing or malformed 'data' list.")

    items = raw_json["data"]
    if not items:
        raise ValueError("mfapi returned an empty 'data' list for the scheme.")

    records: list[tuple[pd.Timestamp, float]] = []
    for item in items:
        date_str = item.get("date")
        nav_val = item.get("nav")
        if not date_str or nav_val is None:
            continue
        try:
            dt = pd.to_datetime(date_str, format="%d-%m-%Y")
            val = float(nav_val)
            if val > 0:
                records.append((dt, val))
        except (ValueError, TypeError):
            continue

    if not records:
        raise ValueError("No valid positive NAV records parsed from mfapi history.")

    df = pd.DataFrame(records, columns=["date", "price"])
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values(by="date")
    series = df.set_index("date")["price"]
    return series


def fetch_mutual_fund_nav(
    scheme_code: int, scheme_name: str = "", force_refresh: bool = False
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Retrieves the historical NAV time series for an Indian mutual fund by scheme code.
    Checks disk cache first. If refresh requested or cache expired, queries mfapi.in.
    Falls back gracefully to stale cache on network failure.
    """
    cache_key = f"mf_{scheme_code}"
    cached_series, cached_meta = _load_from_cache(cache_key, force_refresh=force_refresh)

    if cached_series is not None and cached_meta is not None and not force_refresh and not cached_meta.get("is_stale", True):
        return cached_series, cached_meta

    url = f"{MF_HISTORY_URL}{scheme_code}"
    logger.info("Fetching mutual fund history from live API: %s", url)
    try:
        response = _http_get_with_retry(url)
        raw_json = response.json()
        series = parse_mfapi_history_json(raw_json)

        meta = raw_json.get("meta", {})
        fund_name = scheme_name or meta.get("scheme_name") or f"MF Scheme {scheme_code}"
        fetch_time = datetime.now(timezone.utc).isoformat()
        metadata: dict[str, Any] = {
            "source": "mfapi.in (AMFI Official Data)",
            "scheme_code": scheme_code,
            "scheme_name": fund_name,
            "fund_house": meta.get("fund_house", "Unknown"),
            "scheme_type": meta.get("scheme_type", "Unknown"),
            "scheme_category": meta.get("scheme_category", "Unknown"),
            "last_date": series.index.max().strftime("%Y-%m-%d"),
            "first_date": series.index.min().strftime("%Y-%m-%d"),
            "n_obs": len(series),
            "fetch_timestamp": fetch_time,
            "is_stale": False,
        }
        _save_to_cache(cache_key, series, metadata)
        return series, metadata
    except Exception as exc:
        logger.error("Network fetch failed for MF scheme %d: %s", scheme_code, exc)
        if cached_series is not None and cached_meta is not None:
            logger.warning(
                "Falling back to STALE CACHE for MF scheme %d (last date: %s)",
                scheme_code, cached_meta.get("last_date")
            )
            cached_meta["is_stale"] = True
            cached_meta["stale_reason"] = f"Live fetch failed: {exc}"
            return cached_series, cached_meta
        raise RuntimeError(f"Failed to fetch mutual fund {scheme_code} from live API and no cache exists: {exc}")


def fetch_yfinance_ticker(
    symbol: str, force_refresh: bool = False, is_primary: bool = True
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Fetches daily historical close price for a stock, index, or commodity via yfinance.
    Uses yf.Ticker(symbol).history(period='max', auto_adjust=True, interval='1d').
    Strips timezones, validates monotonically increasing dates and positive prices.
    Caches results and supports stale fallback on network interruption.
    """
    cache_key = f"yf_{symbol}"
    cached_series, cached_meta = _load_from_cache(cache_key, force_refresh=force_refresh)

    if cached_series is not None and cached_meta is not None and not force_refresh and not cached_meta.get("is_stale", True):
        return cached_series, cached_meta

    logger.info("Fetching yfinance series for symbol: '%s'", symbol)
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="max", auto_adjust=True, interval="1d")

        if hist.empty or "Close" not in hist.columns:
            raise ValueError(f"yfinance returned empty dataset or missing 'Close' column for '{symbol}'.")

        series = hist["Close"].copy()

        # Normalize index to timezone-naive dates
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        series.index = pd.to_datetime(series.index.date)

        # Remove duplicate dates and NaN
        series = series[~series.index.duplicated(keep="last")]
        series = series.dropna()
        series = series[series > 0]
        series = series.sort_index()

        if series.empty:
            raise ValueError(f"No positive valid prices remaining for '{symbol}' after cleaning.")

        fetch_time = datetime.now(timezone.utc).isoformat()
        metadata: dict[str, Any] = {
            "source": f"Yahoo Finance ({symbol})",
            "symbol": symbol,
            "last_date": series.index.max().strftime("%Y-%m-%d"),
            "first_date": series.index.min().strftime("%Y-%m-%d"),
            "n_obs": len(series),
            "fetch_timestamp": fetch_time,
            "is_stale": False,
        }

        _save_to_cache(cache_key, series, metadata)
        return series, metadata
    except Exception as exc:
        logger.error("yfinance download failed for '%s': %s", symbol, exc)
        if cached_series is not None and cached_meta is not None:
            logger.warning("Falling back to STALE CACHE for '%s' (last date: %s)", symbol, cached_meta.get("last_date"))
            cached_meta["is_stale"] = True
            cached_meta["stale_reason"] = f"Live fetch failed: {exc}"
            return cached_series, cached_meta
        if is_primary:
            raise RuntimeError(f"Primary asset '{symbol}' download failed and no cache is available: {exc}")
        else:
            logger.warning("Secondary series '%s' skipped due to download error: %s", symbol, exc)
            raise


def fetch_all_macro_series(force_refresh: bool = False) -> tuple[dict[str, pd.Series], dict[str, dict[str, Any]]]:
    """
    Fetches all standard macro series defined in MACRO_SYMBOLS:
    S&P 500, Brent Crude, USD/INR, India VIX, NIFTY 50.
    Skips any non-critical failure while logging clear warnings.
    """
    series_dict: dict[str, pd.Series] = {}
    meta_dict: dict[str, dict[str, Any]] = {}

    for name, sym in MACRO_SYMBOLS.items():
        try:
            s, m = fetch_yfinance_ticker(sym, force_refresh=force_refresh, is_primary=False)
            series_dict[name] = s
            meta_dict[name] = m
            logger.info("Successfully ingested macro variable '%s' (%s): %d observations", name, sym, len(s))
        except Exception as exc:
            logger.warning("Failed to ingest macro series '%s' (%s): %s", name, sym, exc)

    return series_dict, meta_dict
