# MarketFolio: Quantitative Risk, Econometric Diagnostics & Advanced ML Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10--3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![PyTorch: 2.15](https://img.shields.io/badge/PyTorch-CPU%20Verified-ee4c2c)](https://pytorch.org/)
[![Test Suite: 19 Passed](https://img.shields.io/badge/Tests-19%2F19%20Passing-10b981)](tests/)

**MarketFolio** is an institutional quantitative analytics system and advanced machine learning platform for Indian equities, mutual funds, and macroeconomic transmission. It combines rigorous financial econometrics (joint ADF/KPSS unit roots, Cochrane-Orcutt SARIMAX regressions, 5-variable macro VAR systems, expanding-window rolling backtests) with academically defensible deep neural architectures (Variational Autoencoders, Multi-Head Self-Attention Transformers, and Spectral Graph Convolutional Networks).

The platform features both a multi-tab **Streamlit Financial Terminal** and a modern **React + FastAPI Web Trading Terminal**, accompanied by an integrated research governance tracker.

---

## 1. Advanced Machine Learning Suite

MarketFolio enforces strict academic integrity: **zero lookahead bias**, **chronological splits**, **train-only threshold calibration**, and **purged embargo windows** for overlapping horizons.

| Advanced Model | Neural Architecture | Primary Objective | Comparative Baselines | Key Academic Safeguard |
|:---|:---|:---|:---|:---|
| **Deep Anomaly Engine (VAE)** | Multi-channel $\beta$-VAE ($d_z=4$, LeakyReLU, LayerNorm) | Reconstructs normal stationary covariance; flags temporal market dislocations via reconstruction error. | 1. 3-Sigma Rolling Z-score<br>2. Isolation Forest | Anomaly threshold $\tau$ calibrated **exclusively** on pre-split training reconstruction loss (P95). |
| **Volatility Regime Transformer** | Multi-Head Self-Attention ($d_{\text{model}}=64$, 4 heads, attention pooling) | Classifies future $H$-day realized volatility into Low, Normal, and High regimes. | 1. Naive Majority Class<br>2. Last-Known Persistence (Martingale)<br>3. L2 Logistic Regression | $H$-day embargo gap between Train, Val, and Test splits eliminates forward-looking label overlap. Dynamic failure-mode narrative. |
| **Inter-Market Relational Graph (GCN)** | 2-Layer Spectral Graph Convolution (Kipf & Welling 2017) | Models spatial feature aggregation and systemic risk transmission across 6 synchronized cross-market nodes. | 1. Static Correlation Network<br>2. Majority Static Baseline | Normalized Laplacian $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ derived strictly from training correlation matrix. |
| **Research Governance Tracker** | Decoupled Trello Integration | Tracks experiment implementation, validation tasks, blockers, and completion evidence. | Offline Documented Audit Roadmap | Handles missing API credentials gracefully; completely decoupled from core analytics pipeline. |

---

## 2. Classical Econometric Diagnostics (Consolidated Reference Desk)

Classical time-series methods are preserved and consolidated to serve as rigorous benchmark baselines:
1. **Stationarity & Integration**: Joint Augmented Dickey-Fuller (ADF) and KPSS tests on both log price levels and log returns.
2. **Seasonality & Calendar Anomalies**: OLS day-of-week dummy regressions, Kruskal-Wallis non-parametric tests, and monthly STL decomposition ($F_T, F_S$).
3. **Conditional Mean Predictability & Market Efficiency**: AICc-guided grid search over candidate $\text{ARMA}(p, q)$ structures benchmarked against constant-drift martingales to test Fama's (1970) Weak-Form Efficient Market Hypothesis.
4. **Realized Volatility & Residual Bootstrap Risk Ranges**: Models natural log annualized volatility $v_t = \ln(\sqrt{252}\cdot \text{std}_{21}(r_t))$, generating 30-day risk range forecasts via empirical residual bootstrap ($B = 2000$).
5. **Systematic Market Linkage**: Single-index market models with autocorrelated errors ($\text{SARIMAX}$) comparing unbiased alpha and beta estimates against Newey-West HAC OLS.
6. **Global Macro Linkages (Vector Autoregression)**: 5 macroeconomic variables (S&P 500, Brent Crude, USD/INR, India VIX, NIFTY 50) in an identified $\text{VAR}(p)$ system computing Granger causality, Cholesky orthogonalized impulse responses, and FEVD.
7. **Strict Out-of-Sample Evaluation**: Backtests models using expanding-window rolling origins (`assert_no_leakage`), reporting MAE, RMSE, and MASE relative to naive baselines.

---

## 3. Deliberate UI Hierarchy & Classification

Rather than maintaining redundant diagnostic pages, the application interface has been deliberately restructured:
- **Retained**: Executive Desk Overview, Price & Return empirical distributions, Realized Volatility Cone & Bootstrap simulation, and Institutional Research Report.
- **Relocated & Consolidated**: Stationarity (ADF/KPSS), Autocorrelation (ACF/PACF), Calendar anomalies, Cochrane-Orcutt SARIMAX beta, and 5-variable Macro VAR have been consolidated into **"Classical Econometrics & Baseline Diagnostics"**.
- **Promoted**: Dedicated institutional desks for **Deep Anomaly (VAE)**, **Volatility Regimes (Transformer)**, **Inter-Market Topology (GCN)**, and **Research Tracker (Trello)**.
- **Simplified**: Eliminates duplicate narrative text and redundant diagnostic callouts across tabs.

---

## 4. Academic Defensibility & Anti-Leakage Safeguards

1. **Chronological Partitioning**: Data is split chronologically into 70% Train, 15% Validation, and 15% Test. Shuffling is strictly prohibited.
2. **Horizon Embargo & Purging**: For forward-looking target horizon $H$, an embargo period of $H$ trading days is enforced between Train and Validation, and between Validation and Test, ensuring no future information bleeds across boundaries.
3. **Training-Only Feature Preprocessing**: `StandardScaler` parameters ($\mu, \sigma$) and volatility regime quantiles are fitted **exclusively on the training partition** and applied out-of-sample.
4. **Dynamic Failure-Mode Reporting**: When the Transformer fails to beat the Last-Known Persistence baseline out-of-sample, the interface and research memo explicitly state that the martingale null hypothesis cannot be rejected.

---

## 5. Installation & Quick Start

### Prerequisites
- Python 3.10 to 3.12
- Node.js 18+ (for frontend development)
- Git

### Setup Instructions

```bash
# 1. Clone repository
git clone https://github.com/aryan-arvind/marketfolio.git
cd marketfolio

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 4. Install production dependencies (including PyTorch CPU & scikit-learn)
pip install -r requirements.txt
```

---

## 6. Running the Applications

### Interactive Streamlit Financial Desk

```bash
streamlit run app/streamlit_app.py
```

### TradingView-Inspired Web Terminal (FastAPI + React)

```bash
# Start FastAPI backend (serves both API endpoints and built React frontend):
uvicorn backend.api:app --reload --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

*(Optional: For live frontend development with hot-reloading)*:
```bash
cd frontend
npm install
npm run dev
```

---

## 7. Running the Verification & Test Suite

The repository contains 19 automated tests validating classical algorithms, deep neural architectures, and anti-leakage invariants:

```bash
# Run complete test suite (all 19 tests):
pytest -v

# Run Advanced ML tests only:
pytest tests/test_advanced_ml.py -v

# Run live network smoke test against real public APIs:
python scripts/smoke_test.py
```

---

## 8. Optional Trello Research Tracker Setup

To connect live Trello boards to the research governance desk, configure these optional environment variables:

```bash
export TRELLO_API_KEY="your_api_key"
export TRELLO_TOKEN="your_token"
export TRELLO_BOARD_ID="your_board_id"
```

If unconfigured, MarketFolio automatically activates the **Local Research Audit Roadmap**, providing full visibility into validation tasks and leakage checklists without API requirements.

---

## 9. Academic References

- **Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990)**. STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics*, 6(1), 3–73.
- **Efron, B. (1979)**. Bootstrap methods: Another look at the jackknife. *The Annals of Statistics*, 7(1), 1–26.
- **Fama, E. F. (1970)**. Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383–417.
- **Granger, C. W. (1969)**. Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424–438.
- **Hurvich, C. M., & Tsai, C. L. (1989)**. Regression and time series model selection in small samples. *Biometrika*, 76(2), 297–307.
- **Hyndman, R. J., & Athanasopoulos, G. (2021)**. *Forecasting: Principles and Practice* (3rd ed.). OTexts: Melbourne, Australia.
- **Kingma, D. P., & Welling, M. (2014)**. Auto-encoding variational Bayes. *arXiv preprint arXiv:1312.6114*.
- **Kipf, T. N., & Welling, M. (2017)**. Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations (ICLR)*.
- **Sims, C. A. (1980)**. Macroeconomics and reality. *Econometrica*, 48(1), 1–48.
- **Vaswani, A., et al. (2017)**. Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998–6008.

---

## 10. Disclaimer

> **ACADEMIC AND RESEARCH USE ONLY**: MarketFolio is developed for quantitative research and educational analysis. None of the statistical models, deep neural forecasts, or interpretations constitute investment advice, financial recommendations, or trading signals. Past statistical performance is no guarantee of future returns.
