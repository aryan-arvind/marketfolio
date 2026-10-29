# MarketFolio — Master Presentation & Viva Defense Guide
### Advanced Techniques in Machine Learning (ATML) & Quantitative Risk Analytics

---

## 1. Executive Summary & 2-Minute Elevator Pitch

> **What to Say to Mam:**
> *"Good morning, Mam. Today we are presenting **MarketFolio**, an institutional-grade Quantitative Risk Analytics and Macro Terminal designed to evaluate Indian equities and mutual funds.
>
> While classical finance relies on single-variable heuristics like Rolling 3-Sigma bands or linear ARMA models, real market crashes and systemic contagions stem from complex, non-linear multivariate interactions across global asset classes. 
>
> In MarketFolio, we bridge classical econometrics with three distinct Advanced Machine Learning paradigms from our ATML syllabus:
> 1. **Generative Deep Learning:** A Variational Autoencoder (VAE) for multivariate anomaly and structural breakdown detection.
> 2. **Sequence Modeling & Attention:** A Multi-Head Self-Attention Transformer Encoder with learnable temporal attention pooling to predict forward realized volatility regimes.
> 3. **Geometric Deep Learning:** A Spectral Graph Convolutional Network (GCN) that models cross-market systemic risk spillovers across 6 synchronized financial nodes (Equities, S&P 500, Brent Crude, USD/INR, and India VIX).
>
> Above all, our implementation strictly enforces **zero-lookahead chronological embargoes** and **train-only calibration** to satisfy the highest academic integrity standards."*

---

## 2. Step-by-Step Live Demo Sequence

When presenting the live Streamlit dashboard (`http://localhost:8501`):

### Step 1: Sidebar Setup & Standby Screen
- Show the sidebar controls: Select **RELIANCE.NS** (or any NSE ticker/mutual fund), benchmark **^NSEI**, and forecast horizon **10 days**.
- Point out the clean standby container: explains that models are not run continuously in an unbounded loop, but calibrated on demand.
- Click **Run Analysis**.

### Step 2: Tab 1 — Executive Overview
- Show the top KPI strip: Annualized volatility, Sharpe Ratio, Maximum Drawdown, Beta, and Model Statuses.
- Highlight the **Dynamic Narrative**: Point out that the system dynamically synthesizes classical econometrics and deep learning outputs rather than using generic static text.

### Step 3: Tab 2 — Deep Anomaly Engine (VAE)
- Show the **Reconstruction Loss vs. Time Chart**.
- Point out the red dashed horizontal line: **Train P95 Reconstruction Threshold ($1.60267$)**.
- Explain: *"Notice that only points exceeding this threshold are flagged as structural market dislocations. In our test set of 634 days, exactly 5 days (0.81%) were flagged."*
- Show the **2D Latent Manifold Projection ($\mu_{z_1}$ vs $\mu_{z_2}$)**: Normal market windows cluster tightly in the center, while structural anomalies diverge into the periphery.
- Show the **Comparative Scorecard**: Contrast the VAE with Isolation Forest and Rolling 3-Sigma Z-scores. Point out that the VAE flags multivariate covariance breakdowns even on days when univariate daily returns were modest ($0.51\%$).
- Note the **Disclaimer**: Emphasize that historical macro event tags (e.g., COVID-19, Elections) are external calendar overlays for context, NOT claimed causal discoveries by the neural network.

### Step 4: Tab 3 — Volatility Regime Transformer
- Show the **Attention-Pooled Architecture**: Multi-head self-attention mapping 20-day historical feature sequences to 3 discrete forward realized volatility regimes (Low, Normal, Elevated).
- Show the **Empirical Scorecard**: Point out that on the held-out test set, the Transformer achieved $62.44\%$ accuracy and a Macro F1 of $0.3136$.
- Explain the **Honest Scientific Verdict**: *"Mam, notice our system honestly reports 'Martingale Persistence Dominates' because the last-known persistence heuristic achieved a Macro F1 of $0.3981$. Consistent with the Weak-Form Efficient Market Hypothesis, realized volatility has extreme persistence; deep neural networks do not always beat simple heuristics, and our platform never manipulates metrics to fake superiority."*

### Step 5: Tab 4 — Inter-Market Relational Topology (GCN)
- Show the **Normalized Laplacian Adjacency Heatmap ($\mathbf{\tilde{A}}$)**: Illustrates spatial correlation weights between RELIANCE.NS, S&P 500, Brent Crude, USD/INR, India VIX, and NIFTY 50.
- Show the **Out-of-Sample Performance**: GCN achieved a Systemic Stress Macro F1 of **$0.6520$ vs. $0.4425$ for the baseline (+20.95% gain)**, proving that relational graph message passing significantly improves systemic contagion classification.
- Show the **Node Centrality Ranking**: Highlight that USD/INR (30.51%) and S&P 500 (17.61%) are identified as the primary risk transmitters to domestic Indian equities.
- Point out the **Zero-Leakage Guarantee Badge**: Shows the exact 10-day embargo dates separating training from testing.

### Step 6: Tab 5 & 6 — Realized Volatility Cone & Classical Diagnostics
- Show the classical econometric benchmarks: Fan-chart volatility cones, ADF/KPSS stationarity agreement, ARMA conditional mean, and Vector Autoregression (VAR) impulse-response functions.

### Step 7: Tab 7 — Research Governance Tracker (Trello)
- Show the transparent 3-mode tracker: If offline, it clearly displays *"Local Research Roadmap (Offline Specification / API Keys Unset)"* and labels tasks as *"Planned / Unverified Specification"*, demonstrating academic honesty.
- Demonstrate the **Write Operation Expander**: Shows how authenticated POST requests create live cards when credentials are provided.

---

## 3. Deep-Dive Conceptual Reference (ATML Syllabus Alignment)

### 3.1 Model 1: TimeSeriesVAE (Variational Autoencoder)
- **Concept:** Unsupervised generative deep learning for anomaly detection via manifold learning.
- **Why VAE instead of Standard Autoencoder?**
  - Standard Autoencoders lack a continuous, regularized latent space and suffer from overfitting or arbitrary "holes" in representation.
  - VAE imposes a probabilistic prior $\mathcal{N}(0, \mathbf{I})$ over the latent space, forcing the encoder to learn smooth, dense representations of normal market behavior.
- **Core Architecture:**
  - **Encoder Network:** $q_\phi(\mathbf{z}|\mathbf{x})$ parameterized by MLP layers mapping input window ($15 \text{ days} \times 8 \text{ features} = 120\text{D}$) to latent parameters $\boldsymbol{\mu}$ and $\log \boldsymbol{\sigma}^2$ ($\text{latent dim} = 4$).
  - **Reparameterization Trick:** Direct sampling $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ has stochastic nodes with no gradient. We reparameterize as:
    $$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$
    This moves the stochasticity outside the computational graph, allowing backpropagation via standard autograd.
  - **Decoder Network:** $p_\theta(\mathbf{x}|\mathbf{z})$ reconstructs the temporal feature window $\mathbf{\hat{x}}$.
- **Objective Function (Evidence Lower Bound - ELBO):**
  $$\mathcal{L}_{\text{ELBO}}(\theta, \phi; \mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{Reconstruction Loss (MSE)}} - \beta \cdot \underbrace{\mathcal{D}_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \,||\, p(\mathbf{z}))}_{\text{Latent Regularization}}$$
  $$\mathcal{D}_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$
- **Anomaly Scoring & Calibration:**
  - Anomaly score = Mean Squared Reconstruction Error $\|\mathbf{x} - \mathbf{\hat{x}}\|^2$.
  - Threshold is calibrated strictly as the **95th percentile ($P_{95}$) of reconstruction errors on the training set**. Any observation in the test set with error $> P_{95}$ is flagged.

---

### 3.2 Model 2: Volatility Transformer Encoder
- **Concept:** Sequence-to-label classification using multi-head self-attention with temporal pooling.
- **Why Transformer for Volatility Regimes?**
  - Recurrent neural networks (LSTMs/GRUs) suffer from vanishing gradients and sequential computational bottlenecks.
  - Transformers capture long-range temporal dependencies and non-linear volatility clustering directly through pairwise attention across the 20-day lookback window.
- **Core Components:**
  1. **Sinusoidal Positional Encoding:** Injects sequence order since self-attention is permutation-invariant:
     $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
  2. **Multi-Head Scaled Dot-Product Attention:**
     $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
  3. **Learned Attention Pooling:** Rather than taking arbitrary mean pooling or just the last token, we parameterize a learnable query vector $\mathbf{w}_q$:
     $$\alpha_t = \frac{\exp(\mathbf{h}_t^T \mathbf{w}_q)}{\sum_{s=1}^L \exp(\mathbf{h}_s^T \mathbf{w}_q)}, \quad \mathbf{c} = \sum_{t=1}^L \alpha_t \mathbf{h}_t$$
     This allows extracting exact temporal attention weights to see which trading days influenced the prediction most.
  4. **Classification Head:** Linear projection with GELU activation and balanced cross-entropy loss over 3 discrete volatility regimes:
     $$\text{Class 0: Low Vol } (\le P_{33.3}), \quad \text{Class 1: Normal Vol}, \quad \text{Class 2: High Vol } (> P_{66.7})$$
- **Baseline Benchmarking:**
  - Evaluated against **Majority Class**, **Last-Known Martingale Persistence**, and **L2 Logistic Regression** on identical held-out test dates.

---

### 3.3 Model 3: InterMarketGCN (Graph Convolutional Network)
- **Concept:** Geometric deep learning over non-Euclidean cross-asset topology for systemic stress prediction.
- **Why GCN for Multi-Asset Risk?**
  - Financial markets are deeply coupled networks. Asset volatility does not occur in a silo; shocks propagate across currency markets (USD/INR), commodities (Brent Crude), and global indices (S&P 500).
- **Core Formulation (Kipf & Welling Spectral Graph Convolution):**
  - Graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with $N=6$ nodes: $\text{Primary Asset}, \text{S&P 500}, \text{Brent}, \text{USD/INR}, \text{India VIX}, \text{NIFTY 50}$.
  - Adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$ constructed from training return correlation matrix filtered by threshold $\tau=0.20$.
  - Renormalization trick (adding self-loops): $\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{I}_N$.
  - Symmetric degree normalization: $\mathbf{\tilde{D}}_{ii} = \sum_j \mathbf{\tilde{A}}_{ij}$, yielding normalized Laplacian:
    $$\mathbf{\hat{A}} = \mathbf{\tilde{D}}^{-1/2} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-1/2}$$
  - Layer-wise propagation rule:
    $$\mathbf{H}^{(l+1)} = \text{ReLU}\left( \mathbf{\hat{A}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)$$
- **Node Centrality / Vulnerability Metric:**
  - Evaluates the $L_2$ norm of learned node representations under market tension to measure which asset serves as the primary transmission channel of systemic risk.

---

## 4. Academic Integrity & Zero-Leakage Architecture

This is where professors probe the deepest. Here is how our architecture prevents all lookahead bias:

```
Timeline: ---------------------------------------------------------------------------------------->
[=================== TRAIN SET ===================] [EMBARGO 1] [=== VAL SET ===] [EMBARGO 2] [=== TEST SET ===]
(Samples: 3,047 | 2008-03-04 to 2021-04-16)        (10 Days)    (652 Samples)    (10 Days)    (634 Samples)
- StandardScaler fitted on Train ONLY                            - Validation     - Purge      - Strictly held-out
- Volatility Quantile Thresholds (P33, P66) fit on Train ONLY    Early Stopping   Window       - No parameter updates
- VAE Anomaly Threshold (P95) fit on Train ONLY                  Tuning           Boundary     - Unseen test metrics
```

### The 4 Safeguards:
1. **Target Embargo Window ($H = 10$ Days):** When predicting volatility over the next $H$ days $[t+1, t+H]$, training sample targets near the end of the partition would reach into the validation/test period. We strictly terminate training samples at $n_{\text{train}} - H$, and start test samples at $n_{\text{train}} + H$.
2. **Train-Only Preprocessing:** All `StandardScaler` scalers, correlation matrices, and quantile boundaries are fitted strictly on training data (`train_df`), then used to transform validation and test sets.
3. **Strict Parameter Boundary Validation:** `construct_volatility_regime_targets` throws a `ValueError` if `train_end_idx <= 0` or is missing. There is zero silent fallback to the full series.
4. **No Metric Fabrication:** If multiclass `log_loss` cannot be computed, it records `None` with the exact string error. It never fabricates a default like `1.0`.

---

## 5. Most Probable Questions Mam Will Ask & How to Answer

### Q1: "Why did you use a VAE instead of a standard Autoencoder or simple Isolation Forest?"
> **Answer:**
> *"Mam, a standard Autoencoder only maps inputs to discrete deterministic points in latent space without regularization, making the latent space discontinuous and prone to overfitting. A VAE forces the latent representation to follow a standard Gaussian prior $\mathcal{N}(0, \mathbf{I})$ via the KL divergence term. This ensures that normal market conditions form a smooth, continuous manifold.
>
> Compared to Isolation Forest, which relies on random orthogonal splits on tabular values, the VAE evaluates temporal sequences of multivariate features (momentum, rolling volatility, return skewness). In our real-data experiment on RELIANCE.NS, the VAE identified 5 structural market dislocations where multivariate relationships decoupled, even when univariate daily returns were modest ($0.51\%$), proving that it detects covariance breakdowns that tree-based isolation methods miss."*

---

### Q2: "Why does the persistence baseline beat your Transformer on volatility regime classification?"
> **Answer:**
> *"Mam, this is one of our most important scientific findings, and it directly supports the Weak-Form Efficient Market Hypothesis (Fama, 1970). In financial econometrics, realized volatility exhibits extreme autoregressive persistence (volatility clustering). 
> 
> When we benchmarked our Volatility Transformer on the held-out test set (634 days from 2024 to 2026), it achieved an accuracy of $62.44\%$ and a Macro F1 of $0.3136$. However, the Last-Known Persistence baseline (predicting tomorrow's regime to match the prior 10 days) achieved a Macro F1 of $0.3981$. 
>
> In machine learning research, especially finance, reporting that a complex model does not beat a strong persistence martingale is an honest and academically rigorous result. Rather than faking metrics or manipulating thresholds, our system honestly declares 'Martingale Persistence Dominates'."*

---

### Q3: "What is the reparameterization trick in your VAE, and why is it necessary?"
> **Answer:**
> *"Mam, in a VAE, the encoder outputs the parameters of a distribution: mean $\boldsymbol{\mu}$ and log-variance $\log \boldsymbol{\sigma}^2$. If we directly draw a random sample $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$, the sampling operation is a stochastic node. You cannot differentiate through a random sampling process, which breaks backpropagation.
>
> Kingma and Welling introduced the reparameterization trick: we generate an independent random noise vector $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$, and express the latent vector deterministically as:
> $$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$$
> Now, the stochasticity is isolated in $\boldsymbol{\epsilon}$, and the parameters $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ can receive continuous gradients from the reconstruction loss via standard backpropagation."*

---

### Q4: "How does the GCN build its adjacency matrix? Isn't financial data tabular rather than a graph?"
> **Answer:**
> *"Mam, while financial returns are time series, the relationship between assets forms a dynamic network topology. We construct a 6-node graph consisting of our primary asset plus 5 macro drivers: S&P 500, Brent Crude, USD/INR, India VIX, and NIFTY 50.
>
> To define edges, we compute the Pearson correlation matrix of daily returns strictly over the training slice and apply a threshold $\tau = 0.20$. To ensure numerical stability and allow self-feature propagation, we apply Kipf and Welling's renormalization trick:
> $$\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{I}_N, \quad \mathbf{\hat{A}} = \mathbf{\tilde{D}}^{-1/2} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-1/2}$$
> In our held-out test set, the GCN achieved a Macro F1 of **$0.6520$ compared to $0.4425$ for the baseline**, an improvement of over 20%, proving that spatial graph message passing effectively captures systemic contagion."*

---

### Q5: "How do you guarantee that your model did not suffer from data leakage (lookahead bias)?"
> **Answer:**
> *"Mam, data leakage is the most common flaw in financial ML. We implemented three strict architectural safeguards:
> 1. **Chronological Splitting:** We never use random k-fold cross-validation or shuffle time series. Splits are strictly ordered in time (Train: 2008–2021, Val: 2021–2024, Test: 2024–2026).
> 2. **Purge and Embargo Windows:** Because our prediction horizon is $H=10$ days, a label at day $t$ uses returns from $t+1$ to $t+10$. We apply a 10-day embargo gap after the train split and after the validation split. No training label ever consumes a price from the validation or test splits.
> 3. **Train-Only Parameter Fitting:** The `StandardScaler` means and standard deviations, the GCN adjacency matrix, the VAE P95 anomaly threshold, and the Transformer volatility regime cutoffs ($P_{33}, P_{66}$) are calibrated strictly on training observations. If `train_end_idx` is missing or zero, our code throws an explicit `ValueError` rather than falling back to the full dataset."*

---

### Q6: "Does your VAE discover the cause of financial crashes?"
> **Answer:**
> *"No, Mam. We explicitly clarify in our dashboard and report that the VAE is an unsupervised density estimator; it detects mathematical deviations from the learned training manifold, not causal explanations. 
> 
> The event names shown in the anomaly table (such as COVID-19 or General Elections) are external calendar overlays mapped post-hoc for market context. We do not claim that the neural network inferred causality."*

---

### Q7: "What hardware did you train on, and what are the runtimes?"
> **Answer:**
> *"All models were trained on CPU using PyTorch 2.15.0+cpu. Our real experiment on 4,362 historical trading days of RELIANCE.NS and the 5 macro series took:
> - TimeSeriesVAE: 117.62 seconds (35 epochs)
> - Volatility Transformer: 116.33 seconds (30 epochs)
> - InterMarketGCN: 31.70 seconds (35 epochs)
> Total training runtime was approximately 4.5 minutes on CPU. The results, weights, and loss histories are persisted in `reports/real_experiment_evidence_RELIANCE_NS.json` so the UI reuses verified results without needing to retrain on every user click."*

---

## 6. Exact Commands to Launch and Test

### 1. Launch Streamlit Application (Primary Faculty Interface):
```powershell
powershell -Command "$env:PYTHONIOENCODING='utf-8'; .\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.port 8501"
```
*Open browser at `http://localhost:8501`*

### 2. Launch FastAPI + React TradingView Terminal (Alternative Interface):
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```
*Open browser at `http://localhost:8000` (serves the React single-page app and backend API)*

### 3. Run the Full Automated Test Suite (23 Tests):
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### 4. Re-run Real Financial Data Experiment Script:
```powershell
.\.venv\Scripts\python.exe scripts/run_real_experiments.py
```
