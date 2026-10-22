# MarketFolio: Real-Data Risk & Macro-Linkage Monitor for Indian Mutual Funds and Equities

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10--3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Course: ATSA](https://img.shields.io/badge/University%20Project-Applied%20Time%20Series%20Analysis-purple)]()

**MarketFolio** is an end-to-end quantitative time series analysis system designed for Indian mutual funds and equities. Developed as a capstone project for **Applied Time Series Analysis (ATSA)**, it connects directly to live public financial APIs, processes real daily time series, executes rigorous econometric routines, and produces production-grade diagnostic reports, visualizations, and an interactive Streamlit dashboard.

---

## 1. What MarketFolio Does

MarketFolio investigates the statistical and economic properties of asset prices and returns across several core dimensions:
1. **Stationarity & Integration**: Distinguishes stochastic trends from stationary processes using joint ADF and KPSS tests on both log price levels and log returns.
2. **Seasonality & Calendar Anomalies**: Evaluates day-of-week and month-of-year return distributions via OLS dummy regressions and non-parametric Kruskal-Wallis tests, and decomposes price dynamics via robust STL decomposition ($F_T, F_S$).
3. **Conditional Mean Predictability & Market Efficiency**: Executes AICc-guided grid search over candidate $\text{ARMA}(p, q)$ structures, benchmarking against a constant-drift martingale to empirically evaluate Fama's (1970) Weak-Form Efficient Market Hypothesis.
4. **Realized Volatility & Residual Bootstrap Risk Ranges**: Models natural log annualized realized volatility $v_t = \ln(\sqrt{252}\cdot \text{std}_{21}(r_t))$, generating 30-day risk range forecasts via empirical residual bootstrap ($B = 2000$) and analytic bounds.
5. **Systematic Market Linkage**: Estimates single-index market models with autocorrelated errors ($\text{SARIMAX}$) to compute unbiased alpha and beta estimates relative to market benchmarks (e.g. NIFTY 50), comparing against Newey-West HAC OLS.
6. **Global Macro Linkages (Vector Autoregression)**: Ingests 5 macroeconomic variables (S&P 500, Brent Crude, USD/INR, India VIX, NIFTY 50) into an identified $\text{VAR}(p)$ system to compute Granger causality, orthogonalized impulse response functions (Cholesky decomposition), and forecast error variance decompositions (FEVD).
7. **Strict Out-of-Sample Evaluation**: Backtests models using an expanding-window rolling origin protocol without look-ahead bias (`assert_no_leakage`), reporting MAE, RMSE, and MASE relative to naive and rolling-mean baselines.

---

## 2. ATSA Syllabus Alignment

The architecture of RiskLens directly maps each module to core units of the Applied Time Series Analysis (ATSA) syllabus:

| ATSA Syllabus Unit | Core Quantitative Topics | Implementation File | Key Outputs & Diagnostics |
|:---|:---|:---|:---|
| **Unit 1: Introduction to Time Series** | Time series plots, calendar effects, subseries, STL decomposition, feature measures ($F_T, F_S$) | [`src/risklens/seasonality.py`](src/risklens/seasonality.py) | Day-of-week OLS dummy F-test, Kruskal-Wallis test, STL trend/seasonal strength, lag-1 scatter |
| **Unit 1 & 5: Stationarity & Integration** | Unit root testing, difference vs trend stationarity, autocorrelation (ACF/PACF), Ljung-Box test | [`src/risklens/diagnostics.py`](src/risklens/diagnostics.py) | Joint ADF & KPSS decision synthesis, 40-lag ACF/PACF plots with Bartlett bands, Ljung-Box on $r_t$ and $r_t^2$ |
| **Unit 3 & 4: ARMA Identification & Estimation** | Model identification, AICc/BIC model selection, maximum likelihood estimation, residual diagnostics | [`src/risklens/arma_models.py`](src/risklens/arma_models.py) | AICc grid search over $p, q \in [0..3]$, Ljung-Box adjusted residual tests, Jarque-Bera, normal Q-Q plots |
| **Unit 2 & 5: Transfer Functions & Regression with ARMA Errors** | Autocorrelated regression disturbances, Cochrane-Orcutt / SARIMAX representation, single-index models | [`src/risklens/regression_arma.py`](src/risklens/regression_arma.py) | Market beta, annualized alpha, Durbin-Watson statistic, SARIMAX vs HAC OLS comparison |
| **Unit 4, 5 & 8: Realized Volatility & Bootstrap** | Volatility clustering, multi-step point forecasting, recursive residual bootstrap | [`src/risklens/volatility.py`](src/risklens/volatility.py) | Realized volatility modeling, $B=2000$ residual bootstrap percentile paths (5th/50th/95th), expected risk range |
| **Unit 8: Vector Autoregression (VAR)** | Multivariate time series, lag order selection, structural stability, Granger causality, Cholesky IRF, FEVD | [`src/risklens/var_analysis.py`](src/risklens/var_analysis.py) | 5-variable macro VAR system, Granger F-tests, 10-day orthogonalized IRF, FEVD tables, 1-step rolling forecast |
| **Unit 4 & 5: Forecast Evaluation** | Out-of-sample backtesting, expanding-window evaluation, scale-free accuracy metrics | [`src/risklens/evaluation.py`](src/risklens/evaluation.py) | `assert_no_leakage`, rolling-origin backtest, MAE, RMSE, MASE vs naive and rolling-mean baselines |

---

## 3. Real Data Sources & Operational Limits

RiskLens runs **exclusively on live real data** fetched at runtime. No synthetic data, random numbers, or fabricated outputs are used outside the algorithm validation tests in `tests/`.

### Sources
1. **Indian Mutual Funds (AMFI Official Data)**:
   - Accessed via the public community API at `https://api.mfapi.in/mf/`.
   - Resolved at runtime via query matching (`--fund-query "text"`). Prioritizes Direct-Growth options.
   - *Limitation*: IDCW / Dividend payout plans distort NAV return time series due to artificial cash drops; Growth plans are strongly recommended and warned in logs.
2. **Equities, Indices & Commodities (Yahoo Finance)**:
   - Accessed via `yfinance` (`yf.Ticker(symbol).history(period="max", auto_adjust=True, interval="1d")`).
   - Macro system symbols: NIFTY 50 (`^NSEI`), S&P 500 (`^GSPC`), Brent Crude (`BZ=F`), USD/INR (`INR=X`), India VIX (`^INDIAVIX`).
   - *Limitation*: `yfinance` is an unofficial public interface intended for personal, research, and educational purposes. Data is daily close, not high-frequency tick data.

### Caching and Freshness
- Downloaded series are cached locally as CSV files with a sidecar JSON in `.cache/` with a **6-hour TTL**.
- Use the `--refresh` flag to bypass the cache and force live re-download.
- If the public network is unreachable but a cache exists, the system gracefully degrades to the stale cache and stamps `[STALE CACHE]` across all generated outputs.

---

## 4. Installation & Quick Start

### Prerequisites
- Python 3.10 to 3.12
- Git

### Setup Instructions

```bash
# 1. Clone repository
git clone https://github.com/aryan-academic/risklens.git
cd ATSAProj

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 4. Install production dependencies
pip install -r requirements.txt
```

---

## 5. Usage

### Command-Line Interface (CLI)

```bash
# Analyze an Indian Mutual Fund (AMFI runtime query resolution):
python -m risklens.cli --fund-query "Nifty 50 Index Fund Direct Growth"

# Analyze an Indian Stock (Yahoo Finance symbol):
python -m risklens.cli --ticker RELIANCE.NS

# Custom benchmark, horizon, and forced fresh download:
python -m risklens.cli --ticker HDFCBANK.NS --benchmark ^NSEI --horizon 21 --refresh --no-llm
```

Every CLI run generates a complete artifact bundle in `reports/<asset>_<YYYY-MM-DD>/`:
- `report.md`: Complete human-readable ATSA analytical report.
- `results.json`: Full dictionary of computed metrics, p-values, and model parameters.
- `*.csv`: Tables for flagged outliers, ARMA model comparisons, calendar effects, VAR FEVD, and rolling evaluation.
- `*.png`: Publication-grade diagnostic charts (ACF/PACF, STL decomposition, residual Q-Q, volatility forecast, market regression, VAR IRF/FEVD).

### Interactive Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

The multi-tab web dashboard provides interactive visualizations, data freshness banners, full statistical tables, and downloadable markdown reports.

### Running Tests

```bash
# Run all offline algorithm-validation tests (synthetic ground-truth benchmarks):
pytest -q

# Run live network smoke test against real public APIs:
python scripts/smoke_test.py
```

---

## 6. Econometric Interpretation Guide

### Why Direction (Return) is Unpredictable while Volatility is Forecastable
- **Weak-Form Efficient Market Hypothesis (Fama 1970)**: In liquid equity markets, information is rapidly incorporated into prices. Expected future log returns conditional on past returns fluctuate around an unconditional drift $E[r_t \mid r_{t-1}, \dots] \approx \mu$. Hence, ARMA grid searches frequently select $(0,0,0)$ or yield negligible $\Delta \text{AICc} > -2.0$ relative to a constant mean. A null return predictability finding is an economically valid result, not an analytical failure.
- **Volatility Clustering (Mandelbrot 1963; Engle 1982)**: While the *sign* of returns is uncorrelated, the *magnitude* of returns ($|r_t|$ and $r_t^2$) exhibits substantial autocorrelation (ARCH effects). Market uncertainty and information arrival occur in clusters. Consequently, modeling realized volatility $v_t$ with ARMA and residual bootstrap provides strong out-of-sample explanatory and forecasting skill.

---

## 7. Methodological Limitations

1. **Cholesky Causal Ordering**: Orthogonalized impulse response functions and variance decompositions assume a lower-triangular causal ordering: `[S&P 500 -> Brent -> USD/INR -> India VIX -> NIFTY 50]`. Alternative orderings reflect different structural assumptions and will yield altered contemporaneous shock distributions.
2. **Overlapping Rolling Windows**: Realized volatility is calculated over a rolling 21-day window. Overlapping windows induce autocorrelation into $v_t$ by construction, which assists ARMA persistence modeling but requires careful out-of-sample evaluation.
3. **Calendar Misalignment**: Global macro factors (US equities, European crude, Indian equities) operate on different national holiday calendars. Inner-join date alignment discards non-coincident dates and entails lead-lag timing differences (e.g. US trading closes after the Indian trading day ends).
4. **Survivorship Bias**: Public live data reflects actively surviving mutual funds and stocks; discontinued or merged funds are absent from public query endpoints.

---

## 8. Disclaimer

> **ACADEMIC AND EDUCATIONAL USE ONLY**: RiskLens is built strictly for educational and coursework purposes in university Applied Time Series Analysis (ATSA). Nothing produced by this repository constitutes financial, investment, legal, or tax advice. Past statistical performance is no guarantee of future returns.

---

## 9. Academic References

- **Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990)**. STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6(1), 3–73.
- **Efron, B. (1979)**. Bootstrap methods: Another look at the jackknife. *The Annals of Statistics*, 7(1), 1–26.
- **Engle, R. F. (1982)**. Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007.
- **Fama, E. F. (1970)**. Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383–417.
- **Granger, C. W. (1969)**. Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424–438.
- **Hurvich, C. M., & Tsai, C. L. (1989)**. Regression and time series model selection in small samples. *Biometrika*, 76(2), 297–307.
- **Hyndman, R. J., & Athanasopoulos, G. (2021)**. *Forecasting: Principles and Practice* (3rd ed.). OTexts: Melbourne, Australia. [otexts.com/fpp3](https://otexts.com/fpp3).
- **Hyndman, R. J., & Koehler, A. B. (2006)**. Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
- **Mandelbrot, B. (1963)**. The variation of certain speculative prices. *The Journal of Business*, 36(4), 394–419.
- **Sims, C. A. (1980)**. Macroeconomics and reality. *Econometrica*, 48(1), 1–48.
