"""
RiskLens configuration, constants, and logging setup.
Contains no global mutable state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final
import numpy as np

# Base directory paths
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent
CACHE_DIR: Final[Path] = BASE_DIR / ".cache"
REPORTS_DIR: Final[Path] = BASE_DIR / "reports"
LOGS_DIR: Final[Path] = BASE_DIR / "logs"

# Ensure runtime directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Data fetching configuration
CACHE_TTL_HOURS: Final[int] = 6
CACHE_TTL_SECONDS: Final[int] = CACHE_TTL_HOURS * 3600
HTTP_TIMEOUT_SECONDS: Final[int] = 20
HTTP_MAX_RETRIES: Final[int] = 3
HTTP_BACKOFF_FACTOR: Final[float] = 1.5
HTTP_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RiskLens/1.0 "
    "(Applied Time Series Analysis University Project)"
)

# Public API endpoints
MF_SEARCH_URL: Final[str] = "https://api.mfapi.in/mf/search?q="
MF_HISTORY_URL: Final[str] = "https://api.mfapi.in/mf/"

# Financial & Macro ticker mapping
DEFAULT_BENCHMARK_SYMBOL: Final[str] = "^NSEI"
MACRO_SYMBOLS: Final[dict[str, str]] = {
    "SP500": "^GSPC",
    "Brent": "BZ=F",
    "USDINR": "INR=X",
    "IndiaVIX": "^INDIAVIX",
    "NIFTY50": "^NSEI",
}
VAR_CHOLESKY_ORDER: Final[list[str]] = [
    "SP500",
    "Brent",
    "USDINR",
    "IndiaVIX",
    "NIFTY50",
]

# Cleaning and threshold parameters
MIN_PRIMARY_OBSERVATIONS: Final[int] = 750
MIN_MACRO_OBSERVATIONS: Final[int] = 500
MAX_BUSINESS_DAY_GAP_FILL: Final[int] = 3
ANOMALY_ABS_RETURN_THRESHOLD: Final[float] = 0.15

# Quantitative modeling parameters
ANNUALIZATION_TRADING_DAYS: Final[int] = 252
ANNUALIZATION_FACTOR: Final[float] = float(np.sqrt(ANNUALIZATION_TRADING_DAYS))
REALIZED_VOLATILITY_WINDOW: Final[int] = 21

DEFAULT_FORECAST_HORIZON: Final[int] = 21
BOOTSTRAP_REPLICATIONS: Final[int] = 2000
BOOTSTRAP_RANDOM_SEED: Final[int] = 42

# Rolling-origin evaluation defaults
EVAL_INITIAL_WINDOW: Final[int] = 756
EVAL_STEP_SIZE: Final[int] = 21
EVAL_MAX_ORIGINS: Final[int] = 36


def setup_logging(
    log_file: Path | None = None,
    log_level: int = logging.INFO,
    quiet_console: bool = False,
) -> logging.Logger:
    """
    Configures and returns the application logger writing to both console and file.
    Follows standard logging best practices with formatted timestamps.
    """
    if log_file is None:
        log_file = LOGS_DIR / "risklens.log"

    logger = logging.getLogger("risklens")
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File Handler
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console Handler
        if not quiet_console:
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)

    return logger
