# MARKETFOLIO: MASTER ATML STUDY & DEFENSE GUIDE
**A Complete Conceptual, Technical, and Practical Guide for Faculty Evaluation**
*Course: Advanced Techniques in Machine Learning (ATML) | Academic Year: 2025–2026*

---

## Table of Contents
1. [The Problem Statement — Why Does MarketFolio Exist?](#1-the-problem-statement--why-does-marketfolio-exist)
2. [How MarketFolio Works — Step-by-Step System Flow](#2-how-marketfolio-works--step-by-step-system-flow)
3. [Core ATML Concepts Explained in Plain English (With Math & Intuition)](#3-core-atml-concepts-explained-in-plain-english)
   - [3.1 Variational Autoencoder (VAE) for Anomaly & Tail Risk](#31-variational-autoencoder-vae-for-anomaly--tail-risk)
   - [3.2 Multi-Head Self-Attention Transformer for Volatility Regimes](#32-multi-head-self-attention-transformer-for-volatility-regimes)
   - [3.3 Spectral Graph Convolutional Network (GCN) for Inter-Market Systemic Risk](#33-spectral-graph-convolutional-network-gcn-for-inter-market-systemic-risk)
4. [What the Backend Actually Does (Under-the-Hood Code & Execution)](#4-what-the-backend-actually-does)
5. [The Outputs & End-Goal Explanation — What Does It All Mean?](#5-the-outputs--end-goal-explanation)
6. [Obvious & Tough Viva Questions & Bulletproof Answers](#6-obvious--tough-viva-questions--bulletproof-answers)
7. [SEPARATE SECTION: Research Governance & Trello Tracker](#7-separate-section-research-governance--trello-tracker)

---

# 1. The Problem Statement — Why Does MarketFolio Exist?

### The Myth of Predicting Tomorrow's Stock Price
When most engineering students do a project on financial data, they almost always do the same thing:
They take historical daily closing prices, feed them into a basic LSTM or Linear Regression, and predict tomorrow's price with a graph showing two overlapping lines that look "99% accurate."

**In the academic world of Financial Econometrics and Advanced ML, this is completely wrong.**
Here is why:
1. **Eugene Fama's Weak-Form Efficient Market Hypothesis (Nobel Prize, 2013):**
   Stock price movements follow a Markovian random walk. All known historical price information is already baked into today's price. The next tick is driven by unpredictable news and noise.
2. **The "Overfitting to Lag-1" Optical Illusion:**
   If a model simply predicts $\hat{y}_{t+1} = y_t$ (tomorrow's price equals today's price), the plot will look almost identical with an apparent 98% $R^2$, but the model has zero trading or predictive edge—it's just a 1-day delayed mirror!
3. **Severe Lookahead Data Leakage:**
   Students frequently use `train_test_split(shuffle=True)` or fit a `StandardScaler` across the entire 10-year dataset. By doing this, future market crashes leak into the training features, fabricating false high performance that fails instantly in the real world.

### The Real Problem We Solve
Instead of trying to predict unpredictable price noise, institutional risk managers and quantitative researchers care about three fundamental questions:
1. **Anomaly & Tail Risk:** Has the market entered an unprecedented crisis or liquidity dislocation (e.g., March 2020 COVID shock, 2008 Lehman crisis) where historical correlation patterns break down?
2. **Volatility Regimes:** Will the market enter a state of **High Turbulence**, **Normal Drift**, or **Low Volatility** over the next 10 trading days? (While price direction is random, price *volatility* clusters intensely in time).
3. **Cross-Market Systemic Contagion:** How do macroeconomic shocks—such as a spike in Brent Crude Oil, a crash in the US S&P 500, a depreciation in USD/INR, or a surge in the India VIX—propagate through a network to hit an Indian equity like Reliance?

**MarketFolio solves these three real problems** using three distinct Advanced Machine Learning architectures from the ATML syllabus, validated under strict **Zero-Leakage** constraints on **4,362 real trading days** (2008–2026).

---

# 2. How MarketFolio Works — Step-by-Step System Flow

Here is the exact step-by-step pipeline from raw market data to interactive visual insights:

```
[ Step 1: Real Market Data Ingestion ]
  Target: RELIANCE.NS (NSE India)
  Macro Series: NIFTY 50 (^NSEI), S&P 500 (^GSPC), Brent Crude (BZ=F), USD/INR (USDINR=X), India VIX (^INDIAVIX)
                 │
                 ▼
[ Step 2: Econometric Preprocessing & Stationarity ]
  • Log Returns: r_t = ln(P_t / P_{t-1})
  • Joint ADF & KPSS Unit-Root Tests: Proves returns are I(0) stationary
  • Date Alignment: Synchronizes trading holidays across India and the US
                 │
                 ▼
[ Step 3: Zero-Leakage Chronological Partitioning ]
  • Train Set (70%): 2008-01-01 to 2020-05-18 (3,053 days)
  • Validation Set (15%): 2020-05-19 to 2022-12-28 (654 days)
  • Held-Out Test Set (15%): 2022-12-29 to 2026-03-31 (655 days)
  • Scaler Frozen: StandardScaler fit exclusively on Train Set
  • Embargo Gap: H = 10 days purged between boundaries to prevent forward overlap
                 │
                 ▼
[ Step 4: Triple Advanced Machine Learning Engine ]
   ┌───────────────────────────┬───────────────────────────┬───────────────────────────┐
   │ 1. TimeSeriesVAE          │ 2. Volatility Transformer │ 3. Inter-Market GCN       │
   │ Generative Latent Model   │ Self-Attention Sequence   │ Relational Graph Network  │
   │ Task: Anomaly Detection   │ Task: 3-Class Regime      │ Task: Systemic Stress     │
   └───────────────────────────┴───────────────────────────┴───────────────────────────┘
                 │
                 ▼
[ Step 5: Econometric Baseline Benchmarking ]
  • VAE benchmarked against 3-Sigma Rolling Gaussian Z-scores
  • Transformer benchmarked against Martingale Persistence & Naive Majority
  • GCN benchmarked against Single-Asset Logistic Regression (No Topology)
                 │
                 ▼
[ Step 6: Streamlit Interactive Terminal & Visualization ]
  • Active on http://localhost:8501
  • Real-time regime gauges, interactive network graphs, reconstruction loss curves
```

---

# 3. Core ATML Concepts Explained in Plain English

---

## 3.1 Variational Autoencoder (VAE) for Anomaly & Tail Risk

### The Intuition (The Financial "Polygraph")
Imagine you are training a lie detector on an individual's normal speaking voice. After listening to hours of normal speech, the detector knows exactly what their voice sounds like. If the person suddenly panics, screams, or speaks in an unnatural frequency, the detector cannot reconstruct the audio cleanly—the reconstruction error spikes!

That is exactly how our **TimeSeriesVAE** works:
- We train it on 10-dimensional rolling multivariate financial features during normal market conditions.
- The VAE compresses this data into a smooth, 4-dimensional latent probability space and then reconstructs it.
- In normal markets, the reconstruction error is very low (it accurately reproduces the inputs).
- When a severe dislocation occurs (e.g., the COVID crash in March 2020, or sudden macro shocks), the correlations behave completely outside normal bounds. The VAE fails to reconstruct the input, and its **Reconstruction Error spikes**.

### The Math & Architecture

```
Input x (10-dim) ──► Encoder [Linear 32 -> 16] ──► Mean (μ) & Log-Var (log σ²)
                                                           │
                                             Reparameterization Trick:
                                             z = μ + σ ⊙ ε   (ε ~ N(0, I))
                                                           │
                                                           ▼
Output x̂ (10-dim) ◄── Decoder [Linear 16 -> 32] ◄── Latent Vector z (4-dim)
```

1. **The Encoder Network $q_\phi(z|x)$:**
   Maps the 10-dimensional feature vector $x$ into two vectors: the mean $\mu \in \mathbb{R}^4$ and log-variance $\log \sigma^2 \in \mathbb{R}^4$.
2. **The Reparameterization Trick:**
   *The Problem:* We need to draw a random sample $z$ from $\mathcal{N}(\mu, \sigma^2)$. But random sampling is a stochastic, non-differentiable operation—gradients cannot flow back through random chance!
   *The Solution:* We sample random noise $\epsilon$ from a standard normal distribution $\mathcal{N}(0, I)$ independently, and compute:
   $$z = \mu + \sigma \odot \epsilon$$
   Now, the operation is completely differentiable with respect to $\mu$ and $\sigma$. Backpropagation works seamlessly!
3. **The Loss Function (Negative ELBO — Evidence Lower Bound):**
   $$\mathcal{L}_{\text{VAE}} = \underbrace{\frac{1}{D}\sum_{j=1}^{D}(x_j - \hat{x}_j)^2}_{\text{Reconstruction Loss (MSE)}} + \beta \cdot \underbrace{\left(-\frac{1}{2}\sum_{k=1}^{d_z}\left(1 + \log\sigma_k^2 - \mu_k^2 - \sigma_k^2\right)\right)}_{\text{Kullback-Leibler (KL) Divergence}}$$
   - **Reconstruction Loss:** Measures how accurately the decoder reproduces input $x$.
   - **KL Divergence:** Measures how close the learned latent distribution is to a standard Gaussian prior $\mathcal{N}(0, I)$. This prevents the latent space from fragmenting into isolated clusters.
   - **$\beta$-weight ($\beta = 0.05$):** Prevents "posterior collapse," where the model ignores the latent space and only minimizes reconstruction error.
4. **The Academic Safeguard (Zero Leakage):**
   We calculate the reconstruction error for all training days and find the **95th percentile ($\text{P95}$)**. This becomes our fixed anomaly threshold $\tau = 1.60267$. 
   If an out-of-sample test day has a reconstruction loss $> \tau$, it is flagged as an anomaly. We **never** look at the test set to determine this threshold!

---

## 3.2 Multi-Head Self-Attention Transformer for Volatility Regimes

### The Intuition (The Attentive Financial Historian)
Standard Recurrent Neural Networks (RNNs and LSTMs) read time-series data one day at a time from left to right. By day 20, the model has largely forgotten subtle nuances from day 1, and it cannot parallelize computations.

A **Transformer** behaves like a seasoned financial historian. It looks at the entire 20-day historical window simultaneously. If day 18 had a sudden surge in trading volume and day 2 had a minor currency dip, the **Self-Attention mechanism** directly links day 18 and day 2, assigning a high mathematical attention weight to that interaction regardless of how far apart they occurred.

### Why Volatility Regimes Instead of Stock Prices?
- Stock price returns have almost zero autocorrelation (they are white noise).
- But **volatility is conditionally heteroskedastic**—it exhibits long memory and clustering (Mandelbrot, 1963). Periods of turbulence are followed by turbulence; periods of calm are followed by calm.
- We formulate the task as predicting the **future 10-day realized volatility regime**:
  - **Class 0 (Low Volatility):** Realized volatility in the bottom 33.3% of historical training distribution.
  - **Class 1 (Normal Volatility):** Middle 33.3%.
  - **Class 2 (High Volatility):** Top 33.3% (turbulent, high-risk conditions).

### The Math & Architecture

```
Input Sequence (20 days × 10 features)
       │
  [+] Sinusoidal Positional Encoding (Injects temporal order)
       │
  ┌────▼────────────────────────────────────────────┐
  │ Transformer Encoder Layer 1                     │
  │ • Multi-Head Self-Attention (4 Heads, d_model=32)│
  │ • LayerNorm + Dropout (0.10) + Residual Add     │
  │ • Feed-Forward Network (d_ff=64) + Residual Add │
  └────┬────────────────────────────────────────────┘
       │
  ┌────▼────────────────────────────────────────────┐
  │ Transformer Encoder Layer 2                     │
  └────┬────────────────────────────────────────────┘
       │
  Mean Sequence Pooling ──► Linear (32 -> 3) ──► Softmax ──► [P(Low), P(Norm), P(High)]
```

1. **Positional Encoding:**
   Self-attention is permutation-invariant—it doesn't naturally know which day came first. We inject time order using sinusoidal frequencies:
   $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
2. **Scaled Dot-Product Attention:**
   Given Queries $Q$, Keys $K$, and Values $V$ derived from linear projections of the input:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   Dividing by $\sqrt{d_k}$ prevents the dot products from growing excessively large, which would push softmax gradients into vanishing regions.
3. **The Embargo Gap Safeguard:**
   Our target is calculated over the *next 10 days* ($t+1$ to $t+10$). 
   If training ends on day $T$, the target for day $T$ requires data up to day $T+10$. If test starts on day $T+1$, the test features would overlap with the training target!
   **Our Solution:** We enforce a strict **$H = 10$-day Embargo Gap**: we discard days $T+1$ through $T+10$ so training targets never overlap with test observations.
4. **The Honest Empirical Finding (Why Persistence Won):**
   On our held-out test set (2022–2026), the Transformer achieved **62.44% raw accuracy**, but had a lower Macro F1 (0.3136) than the simple **Martingale Persistence Baseline** (0.3981), which simply predicts: *"tomorrow's regime will be the same as today's."*
   Because markets can remain in a single regime for 6 to 9 months at a time, naive persistence is a formidable econometric benchmark. Reporting this honestly proves you did genuine scientific experimentation rather than cherry-picking numbers!

---

## 3.3 Spectral Graph Convolutional Network (GCN) for Inter-Market Systemic Risk

### The Intuition (The Financial Spiderweb)
No company exists in isolation. Reliance Industries is deeply impacted by:
- The broad Indian market (**NIFTY 50**)
- The global benchmark (**S&P 500**)
- Its raw material cost (**Brent Crude Oil**)
- Import/export pricing (**USD / INR FX Rate**)
- Domestic market fear (**India VIX**)

If crude oil jumps 10% and the US dollar strengthens against the rupee, a risk shock ripples through the system before hitting Indian equities. 

Standard ML treats each asset as an independent column in a flat table. A **Graph Convolutional Network (GCN)** models the financial market as a **connected relational network**, allowing nodes to communicate across edges via **Message Passing**.

### The Math & Architecture (Kipf & Welling, 2017)

```
        [ S&P 500 ] ────── (Global Sentiment) ────── [ NIFTY 50 ]
             │                                              │
      (Currency Shift)                              (Market Drag)
             │                                              │
             ▼                                              ▼
        [ USD / INR ] ─── (Import Costs) ───► [ RELIANCE.NS ] ◄─── [ Brent Crude ]
             ▲                                      ▲
             │                                      │
             └─────────── [ India VIX ] ────────────┘
```

1. **6-Node Graph Construction:**
   The nodes represent: `[RELIANCE, NIFTY50, SP500, BrentCrude, USDINR, IndiaVIX]`.
2. **Correlation Adjacency Matrix $A$:**
   We compute the pairwise Pearson correlation matrix $\rho_{ij}$ of log-returns **strictly on the training partition**. 
   To eliminate spurious noise, we threshold edges:
   $$A_{ij} = \begin{cases} |\rho_{ij}| & \text{if } |\rho_{ij}| \ge 0.15 \\ 0 & \text{otherwise} \end{cases}$$
3. **Self-Loops & Symmetric Degree Normalization:**
   Each node must retain its own features, so we add an identity matrix: $\tilde{A} = A + I_N$.
   If node $i$ is connected to 5 nodes and node $j$ is connected to 1 node, multiplying by $\tilde{A}$ would scale up node $i$'s features by a factor of 5!
   To fix this, we apply **Kipf & Welling's Symmetric Degree Normalization**:
   $$\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}, \quad \text{where } \tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$$
4. **GCN Layer Update Rule:**
   $$H^{(l+1)} = \text{ReLU}\left(\hat{A} H^{(l)} W^{(l)}\right)$$
   - $H^{(0)} = X \in \mathbb{R}^{6 \times F}$ (initial node feature matrix).
   - $W^{(l)}$ is the learnable weight matrix.
   - Multiplying by $\hat{A}$ aggregates neighbor features smoothly.
5. **Systemic Stress Target:**
   A day is defined as **Systemic Stress (Class 1)** if the primary asset drops $\le -1.5\%$ on the same day that India VIX spikes $\ge 85\text{th}$ percentile. Otherwise, it is Normal (Class 0).
6. **The Result (+20.95% Performance Gain):**
   - **Asset-Only Logistic Regression:** Macro F1 = **0.4425**
   - **Inter-Market GCN:** Macro F1 = **0.6520**
   - **Conclusion:** Graph message passing delivered an undeniable **+20.95% improvement**, proving that macro network topology provides critical risk signals!

---

# 4. What the Backend Actually Does

When you run the project, what code actually executes behind the scenes?

1. **Data Loading (`src/risklens/market_data.py`):**
   - Downloads daily adjusted closing prices from Yahoo Finance for the target ticker and 5 macro assets from 2008 to 2026.
   - Forward-fills minor trading holiday misalignments between the US and India and drops unaligned rows, yielding **4,362 synchronized trading sessions**.
2. **Feature Engineering (`src/risklens/ml_pipeline.py`):**
   - Computes 1-day log returns: $r_t = \ln(P_t / P_{t-1})$.
   - Computes rolling annualized volatility (21-day window): $\sigma_{21, t} = \text{std}(r_{t-20:t}) \times \sqrt{252}$.
   - Computes rolling Parkinson volatility (using High-Low price ranges).
   - Computes rolling MACD (12-day vs 26-day EMA differential) and 14-day RSI.
   - Combines asset features and macroeconomic returns into a 10-dimensional feature matrix.
3. **Zero-Leakage Scaler Execution:**
   - Instantiates `StandardScaler()`.
   - Calls `.fit(X_train)` on the first 3,053 rows only.
   - Stores the learned mean $\mu_{\text{train}}$ and variance $\sigma^2_{\text{train}}$.
   - Calls `.transform(X_test)` using those frozen parameters.
4. **PyTorch Model Execution (`src/risklens/`):**
   - Converts NumPy arrays to `torch.float32` tensors.
   - Runs forward passes through `TimeSeriesVAE`, `VolatilityTransformer`, and `InterMarketGCN`.
   - Computes backpropagation with the `AdamW` optimizer and gradient clipping (`max_norm=1.0`).
   - Uses `EarlyStopping(patience=8)` monitoring validation loss to save the best checkpoint.
5. **Evaluation & Metrics Calculation:**
   - Uses `scikit-learn` to calculate multiclass accuracy, balanced accuracy, Macro F1, and log-loss.
   - Evaluates baseline models (Martingale Persistence and Logistic Regression) on the exact same test splits.
6. **Frontend Presentation (`app/streamlit_app.py`):**
   - Uses Streamlit caching (`@st.cache_data`) so results render instantly.
   - Generates interactive Plotly charts with custom dark-themed styling, interactive sliders, and metric summary badges.

---

# 5. The Outputs & End-Goal Explanation

When you look at the dashboard screens on `http://localhost:8501`, here is what every output means:

### 1. The Anomaly Reconstruction Chart (VAE Tab)
- **What it displays:** A time-series line chart showing the VAE's daily Reconstruction Loss.
- **The Red Dashed Line:** Represents the 95th-percentile threshold calibrated on historical data.
- **What the Spikes Mean:** Whenever the line pierces above the red dashed line, it indicates an acute market anomaly. For example, during March 2020 (COVID-19), the loss spiked over 300% above threshold, correctly flagging an unprecedented regime shift in real time.

### 2. The Volatility Regime Cone & Probabilities (Transformer Tab)
- **What it displays:** Three probability gauges (Low, Normal, High Volatility) for the upcoming 10 trading days.
- **What it means for an investor:** If the model predicts an 85% probability of a **High Volatility Regime**, an institutional fund manager knows to reduce portfolio leverage, tighten stop-losses, or buy protective put options.
- **The Benchmark Comparison Table:** Displays the Transformer's test accuracy (62.44%) alongside the Martingale Persistence baseline, demonstrating academic transparency.

### 3. The Interactive Inter-Market Graph (GCN Tab)
- **What it displays:** A 6-node network graph with weighted connecting edges.
- **What the Edge Thickness means:** Stronger historical correlation. Thick edges connect Reliance with NIFTY 50 and Brent Crude Oil.
- **The Centrality Analysis:** Identifies the top **Risk Transmitters**:
  - **USD / INR (30.51% risk centrality):** Currency devaluation puts immediate pressure on foreign institutional investments.
  - **US S&P 500 (17.61% risk centrality):** US market sentiment drives global institutional liquidity flows.

### The Ultimate End Goal
The end goal of MarketFolio is **Institutional Quantitative Risk Governance**:
It provides portfolio managers with an integrated cockpit that combines classical econometric guarantees (stationarity, cointegration) with advanced deep learning models (generative anomalies, attention regimes, relational network graphs) without falling into the trap of lookahead data leakage.

---

# 6. Obvious & Tough Viva Questions & Bulletproof Answers

### Q1: "Why did you use a VAE instead of a simple Autoencoder or Isolation Forest?"
> **Answer:** "Mam, a standard Autoencoder maps inputs to isolated points in latent space without any continuity constraint, which often leads to overfitting or memorization. A Variational Autoencoder enforces a continuous Gaussian prior $\mathcal{N}(0, I)$ via the KL Divergence term in the loss function. This creates a smooth, probabilistic representation of 'normal' market covariance. An Isolation Forest is a tree-based partitioning heuristic that doesn't learn latent feature correlations. The VAE learns the underlying joint distribution of multivariate market returns, so its reconstruction error reflects true structural breakdown."

### Q2: "What is the Reparameterization Trick, and why can't we just sample randomly?"
> **Answer:** "Mam, in a VAE, we need to sample a latent vector $z$ from $\mathcal{N}(\mu, \sigma^2)$. But random sampling is a non-differentiable operation—meaning gradients cannot be calculated through a random node during backpropagation. The Reparameterization Trick solves this by separating the deterministic parameters from the stochastic noise. We draw $\epsilon \sim \mathcal{N}(0, I)$ independently and compute $z = \mu + \sigma \odot \epsilon$. Because $\mu$ and $\sigma$ are combined with $\epsilon$ using basic arithmetic, the operation is completely differentiable, allowing standard gradient descent to train the encoder."

### Q3: "Did you shuffle your time-series data when splitting into Train, Val, and Test?"
> **Answer:** "No, absolutely not, Mam. In time-series machine learning, **shuffling is strictly forbidden**. If you shuffle, future information leaks into the training set, allowing the model to 'cheat' by memorizing future price movements. We used a strict **chronological partition**: 70% Train (2008–2020), 15% Validation (2020–2022), and 15% Held-Out Test (2022–2026). In addition, our StandardScaler was fit **exclusively on the training partition** and frozen before transforming the validation and test sets."

### Q4: "What is an Embargo Gap and why did you implement it?"
> **Answer:** "Mam, an Embargo Gap prevents forward-looking label leakage across partition boundaries. Our target is the realized volatility over the *next $H=10$ trading days*. For a sample at the very end of the training set, its target label uses prices from days $t+1$ to $t+10$. If the test set starts immediately at $t+1$, the training target overlaps with the test features. To guarantee zero lookahead bias, we enforce an $H=10$ day embargo gap between Train and Validation, and between Validation and Test, completely purging overlapping observations."

### Q5: "Why did your Transformer have a lower Macro F1 than the Persistence baseline on test data?"
> **Answer:** "Mam, this is one of the most important empirical findings of our research. In financial econometrics, volatility exhibits intense **clustering and autoregressive persistence** (Mandelbrot, 1963). If today is in a high-volatility regime, tomorrow has an overwhelming probability of remaining high. Because markets spend extended periods in a single regime, the naive Persistence baseline ('predict tomorrow's regime equals today's regime') is very difficult to beat out-of-sample. Rather than faking synthetic numbers, reporting this real econometric finding demonstrates true academic honesty and respects the Efficient Market Hypothesis."

### Q6: "How did you construct the Adjacency Matrix in the GCN? Why is degree normalization necessary?"
> **Answer:** "Mam, we compute the pairwise Pearson correlation matrix $\rho_{ij}$ of daily log-returns strictly on the training partition. We apply a threshold $\tau = 0.15$ to eliminate noise, and add self-loops $\tilde{A} = A + I_N$ so each asset retains its own features. We then apply Kipf and Welling's Symmetric Degree Normalization: $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$. Without degree normalization, nodes with many connections (like NIFTY 50) would scale feature vectors up by the number of edges during message passing, causing gradient explosion and unstable training."

### Q7: "What loss function did you use for the VAE, and what does $\beta$ do?"
> **Answer:** "Mam, the VAE optimizes the Evidence Lower Bound (ELBO): $\mathcal{L} = \text{MSE}(x, \hat{x}) + \beta \cdot D_{\text{KL}}(q(z|x) \,\|\, \mathcal{N}(0, I))$. The first term is Mean Squared Error, measuring how accurately the input is reconstructed. The second term is KL Divergence, penalizing deviation from a standard Gaussian prior. The hyperparameter $\beta$ controls the trade-off; we use $\beta = 0.05$ to prevent posterior collapse, ensuring the model maintains good reconstruction fidelity while keeping the latent space smooth."

### Q8: "Why did you predict Volatility Regimes instead of tomorrow's exact stock price?"
> **Answer:** "Mam, under Eugene Fama's Weak-Form Efficient Market Hypothesis, stock prices follow a random walk where past prices cannot predict future price direction. Any model claiming 95% accuracy in predicting tomorrow's exact price is either overfitting to lag-1 or suffering from data leakage. However, while return *direction* is unpredictable, **return variance (volatility)** clusters predictably. Institutional risk managers use volatility regime forecasting to adjust portfolio risk, calculate Value at Risk (VaR), and hedge systemic shocks."

### Q9: "What hardware did you train this on?"
> **Answer:** "Mam, all models were trained natively on the host multi-threaded CPU using PyTorch 2.5 (`torch.cuda.is_available() == False`). Because we designed streamlined, parameter-efficient architectures ($d_{\text{model}}=32$ for the Transformer, $d_z=4$ for the VAE), the entire pipeline trains on 4,362 historical trading days in under 5 minutes without needing a dedicated GPU."

---

# 7. SEPARATE SECTION: Research Governance & Trello Tracker

*(Everything about the Trello integration explained in detail, completely decoupled from the ML pipeline)*

```
                               ┌──────────────────────────────────────────────┐
                               │           MarketFolio UI / Terminal          │
                               └──────────────────────┬───────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
     [ Trello API Configured & Online ]                             [ Trello API Offline / Unset ]
       Mode: LIVE_SYNCHRONIZED                                        Mode: LOCAL_ROADMAP_ONLY
       • Fetches live boards via HTTPS REST API                       • Activates offline curated roadmap
       • Authenticated POST / PUT requests                            • Displays 5 audit stages
       • Live card creation and list movement                         • Fully decoupled: Zero ML impact
```

### 7.1 What is the Research Governance Tracker?
In professional quantitative finance and modern MLOps, building machine learning models without a formal experiment tracking and governance system is considered bad engineering practice.

When developing complex models, researchers must track:
- Which models are currently in development?
- Which models are blocked by data issues?
- What data leakage checks have been performed?
- What validation evidence confirms a model is ready for deployment?

To solve this, MarketFolio includes a dedicated **Research Governance Desk** integrated with **Trello**.

### 7.2 The 3-State Tracker Architecture
A common flaw in software architecture is tightly coupling external APIs with core analytical engines: if an external API fails, the entire application crashes.

In MarketFolio, the Trello tracker is **completely decoupled** using an explicit 3-state enum (`TrackerSyncMode`):

| Tracker Mode | When Activated | Behavior |
| :--- | :--- | :--- |
| **`LIVE_SYNCHRONIZED`** | `TRELLO_API_KEY`, `TRELLO_TOKEN`, and `TRELLO_BOARD_ID` are present and valid. | Connects directly to the live Trello REST API. Fetches live cards, allows creating new cards, and moving cards between lists. |
| **`API_CONNECTION_FAILED`** | Credentials are provided, but internet connection is down or API returns 401/404. | Displays an explicit error banner in the UI: *"Trello API connection failed. Reverting to local audit roadmap."* Never pretends live synchronization succeeded! |
| **`LOCAL_ROADMAP_ONLY`** | No credentials configured (default local academic mode). | Displays the curated **Local Research Audit Roadmap** with 5 standardized quantitative stages. |

### 7.3 The 5 Standardized Audit Lists
In the Local Roadmap mode, MarketFolio organizes project milestones into 5 clear research stages:
1. **Planned / Unverified Specification:** Features proposed but not yet implemented.
2. **In Progress:** Features currently under active development.
3. **Blocked / Technical Debt:** Tasks delayed by dependencies or computational constraints.
4. **Validation & Leakage Check:** Critical audit checklist verifying zero-leakage, stationarity checks, and scaler isolation.
5. **Completed & Verified:** Models that have passed the full 23-test suite and real-data training runs.

### 7.4 What the Backend Does for Trello (`src/risklens/trello_tracker.py`)
1. **`fetch_board_data()`:**
   - Checks environment variables.
   - If missing, immediately returns the local roadmap with `sync_mode = LOCAL_ROADMAP_ONLY`.
   - If present, makes an authenticated `GET` request to `https://api.trello.com/1/boards/{board_id}/lists?cards=open`.
   - If the request fails, captures the error message and falls back to local data with `sync_mode = API_CONNECTION_FAILED`.
2. **`create_trello_card(name, description, list_name)`:**
   - In Live Mode: Sends a `POST` request to `https://api.trello.com/1/cards` with the target list ID.
   - In Local Mode: Returns a structured response informing the user: *"Live card creation requires Trello credentials. Card logged locally."*
3. **`update_trello_card(card_id, target_list_name)`:**
   - In Live Mode: Sends a `PUT` request to update the card's parent list.

### 7.5 Obvious Viva Questions on the Trello Tracker

#### Q1: "Why did you include a Trello tracker in a Machine Learning project?"
> **Answer:** "Mam, modern machine learning in quantitative finance requires rigorous **research governance and MLOps tracking**. When building multiple models (VAE, Transformer, GCN), you must track experimental hypotheses, technical debt, and zero-leakage validation checklists. The Trello desk serves as our research audit trail, ensuring every model undergoes documented verification before being deployed."

#### Q2: "What happens if your internet connection fails or Trello credentials are missing?"
> **Answer:** "Mam, our implementation adheres to the **Decoupled Architecture Pattern**. The core ML analytics engine and the Streamlit app will **never crash** if Trello is offline. The tracker checks for credentials: if missing or if the API returns an error, it gracefully falls back to our **Local Research Audit Roadmap**, displaying an explicit banner so the user always knows whether they are looking at live board data or the local audit roadmap."

#### Q3: "Does the Trello tracker interact with or modify the ML model weights?"
> **Answer:** "No, Mam. The Trello tracker is strictly a **governance and task management layer**. It observes and records experiment metadata and audit checklists, but is completely decoupled from the mathematical pipeline (`ml_pipeline.py`). This guarantees that governance tracking never introduces side effects or latency into model training and inference."

---

## 🎯 Final Presentation Quick Checklist

1. **Terminal Command to Run:**
   ```powershell
   $env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
   .\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.port 8501
   ```
2. **Open in Browser:** `http://localhost:8501`
3. **Run Unit Tests:** `pytest tests/ -v` (23 passed)
4. **Key Talking Points:**
   - Don't predict price noise; predict **Anomalies**, **Volatility Regimes**, and **Inter-Market Networks**.
   - Strict **Zero-Leakage** with an $H=10$-day Embargo Gap and train-only scaler.
   - **Academic Honesty:** Transparently acknowledging when econometric baselines (Persistence) dominate vs. when GCN graph topology delivers a +20.95% win.
   - **Decoupled Governance:** Trello tracker provides research MLOps without coupling to the ML pipeline.
