"""
Live API Smoke Test for RiskLens.
Exercises real public endpoints:
1. mfapi.in search and history (AMFI official data)
2. Yahoo Finance equities and indices (^NSEI, RELIANCE.NS)
Verifies live connectivity, response formats, positive prices, and date structures.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from risklens.config import setup_logging
from risklens.data_sources import (
    fetch_mutual_fund_nav,
    fetch_yfinance_ticker,
    resolve_mutual_fund,
    search_mutual_fund,
)


def run_smoke_test() -> bool:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logger = setup_logging()
    print("=" * 70)
    print(" RUNNING RISKLENS LIVE PUBLIC API SMOKE TEST")
    print("=" * 70)

    success = True

    # 1. Test mfapi.in Search
    print("\n[1/4] Testing mfapi.in search endpoint...")
    try:
        search_query = "Nifty 50 Index Fund Direct Growth"
        candidates = search_mutual_fund(search_query)
        assert len(candidates) > 0, "mfapi search returned empty candidate list"
        print(f"  [PASS] Successfully received {len(candidates)} candidates for query '{search_query}'.")
        scheme_code, scheme_name = resolve_mutual_fund(search_query)
        print(f"  [PASS] Resolved scheme: {scheme_name} (Code: {scheme_code})")
    except Exception as exc:
        print(f"  [FAIL] mfapi.in search failed: {exc}")
        success = False
        scheme_code = 120716  # fallback for next step if search fails

    # 2. Test mfapi.in History Fetch
    print(f"\n[2/4] Testing mfapi.in history fetch for scheme {scheme_code}...")
    try:
        series, meta = fetch_mutual_fund_nav(scheme_code, force_refresh=True)
        assert len(series) > 100, f"NAV history unexpectedly short: {len(series)} points"
        assert (series > 0).all(), "Non-positive NAV values detected"
        print(f"  [PASS] Ingested {len(series)} historical NAV records ({series.index.min().date()} to {series.index.max().date()}).")
        print(f"  [PASS] Latest NAV: {series.iloc[-1]:.4f} as of {meta.get('last_date')}")
    except Exception as exc:
        print(f"  [FAIL] mfapi.in history fetch failed: {exc}")
        success = False

    # 3. Test yfinance Benchmark (^NSEI)
    print("\n[3/4] Testing Yahoo Finance index fetch for NIFTY 50 (^NSEI)...")
    try:
        nifty, nifty_meta = fetch_yfinance_ticker("^NSEI", force_refresh=True, is_primary=False)
        assert len(nifty) > 500, f"NIFTY 50 history unexpectedly short: {len(nifty)} points"
        assert (nifty > 0).all(), "Non-positive price values in NIFTY 50"
        assert nifty.index.tz is None, "NIFTY 50 index failed to normalize to timezone-naive"
        print(f"  [PASS] Ingested {len(nifty)} daily close prices for ^NSEI ({nifty.index.min().date()} to {nifty.index.max().date()}).")
        print(f"  [PASS] Latest Close: {nifty.iloc[-1]:.2f} as of {nifty_meta.get('last_date')}")
    except Exception as exc:
        print(f"  [FAIL] Yahoo Finance NIFTY 50 fetch failed: {exc}")
        success = False

    # 4. Test yfinance Stock (RELIANCE.NS)
    print("\n[4/4] Testing Yahoo Finance equity fetch for RELIANCE.NS...")
    try:
        rel, rel_meta = fetch_yfinance_ticker("RELIANCE.NS", force_refresh=True, is_primary=True)
        assert len(rel) > 500, f"RELIANCE.NS history unexpectedly short: {len(rel)} points"
        assert (rel > 0).all(), "Non-positive price values in RELIANCE.NS"
        print(f"  [PASS] Ingested {len(rel)} daily close prices for RELIANCE.NS ({rel.index.min().date()} to {rel.index.max().date()}).")
        print(f"  [PASS] Latest Close: {rel.iloc[-1]:.2f} as of {rel_meta.get('last_date')}")
    except Exception as exc:
        print(f"  [FAIL] Yahoo Finance RELIANCE.NS fetch failed: {exc}")
        success = False

    print("\n" + "=" * 70)
    if success:
        print(" ALL LIVE API CHECKS PASSED: Environment ready for quantitative analysis.")
    else:
        print(" SMOKE TEST DETECTED FAILURES: Check logs and network status.")
    print("=" * 70 + "\n")

    return success


if __name__ == "__main__":
    ok = run_smoke_test()
    sys.exit(0 if ok else 1)
