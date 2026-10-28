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

# Advanced Machine Learning Parameters
ML_LOOKBACK_WINDOW: Final[int] = 20
ML_TRAIN_RATIO: Final[float] = 0.70
ML_VAL_RATIO: Final[float] = 0.15
ML_TEST_RATIO: Final[float] = 0.15
ML_RANDOM_SEED: Final[int] = 42

# VAE Anomaly Detection
VAE_LATENT_DIM: Final[int] = 4
VAE_HIDDEN_DIM: Final[int] = 64
VAE_EPOCHS: Final[int] = 40
VAE_LEARNING_RATE: Final[float] = 0.002
VAE_ANOMALY_PERCENTILE: Final[float] = 95.0

# Transformer Volatility Regime Classification
TRANSFORMER_D_MODEL: Final[int] = 64
TRANSFORMER_NHEAD: Final[int] = 4
TRANSFORMER_LAYERS: Final[int] = 2
TRANSFORMER_DIM_FEEDFORWARD: Final[int] = 128
TRANSFORMER_EPOCHS: Final[int] = 35
TRANSFORMER_LEARNING_RATE: Final[float] = 0.001
TRANSFORMER_DROPOUT: Final[float] = 0.1

# Graph Convolutional Network (GCN) Multi-Asset
GCN_HIDDEN_DIM: Final[int] = 32
GCN_EPOCHS: Final[int] = 40
GCN_LEARNING_RATE: Final[float] = 0.005
GCN_CORR_THRESHOLD: Final[float] = 0.25


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
