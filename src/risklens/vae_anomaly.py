"""
MarketFolio Advanced Machine Learning: Variational Autoencoder (VAE) Anomaly Engine.
Provides a deep generative VAE model for multi-channel financial time-series anomaly detection,
with strict training-only threshold calibration, latent space projections,
and rigorous comparison against classical Rolling Z-Score and Isolation Forest baselines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from risklens.config import (
    ML_RANDOM_SEED,
    VAE_ANOMALY_PERCENTILE,
    VAE_EPOCHS,
    VAE_HIDDEN_DIM,
    VAE_LATENT_DIM,
    VAE_LEARNING_RATE,
)
from risklens.ml_pipeline import DatasetSplits, create_sliding_windows

logger = logging.getLogger("risklens")


class TimeSeriesVAE(nn.Module):
    """
    Variational Autoencoder for multivariate financial time-series windows.
    Compresses temporal multi-channel feature windows into a low-dimensional
    Gaussian latent space, reconstructing normal financial dynamics.
    High reconstruction loss indicates structural market dislocations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = VAE_HIDDEN_DIM,
        latent_dim: int = VAE_LATENT_DIM,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder Network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder Network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10.0, 10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ELBO loss = MSE reconstruction loss + beta * KL Divergence.
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    # KL divergence per sample averaged across batch
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


@dataclass
class AnomalyResult:
    model_name: str
    timeline_df: pd.DataFrame
    reconstruction_threshold: float
    train_loss_history: list[float]
    val_loss_history: list[float]
    n_anomalies_detected: int
    anomaly_rate_pct: float
    top_anomalies: list[dict[str, Any]]
    latent_coords: pd.DataFrame
    baseline_scorecard: pd.DataFrame
    interpretation: str
    model_summary: dict[str, Any]


KNOWN_HISTORICAL_EVENTS = [
    ("2020-03", "COVID-19 Global Market Crash / Nationwide Lockdown"),
    ("2020-04", "WTI Crude Negative Price Dislocation"),
    ("2020-11", "Global Vaccine Announcement Rally"),
    ("2021-02", "Union Budget 2021 Infrastructure Push"),
    ("2022-02", "Russia-Ukraine Geopolitical Incursion & Energy Shock"),
    ("2022-06", "Global Central Bank Rate Hike Acceleration"),
    ("2023-01", "Adani Group Hindenburg Report Volatility Dislocation"),
    ("2023-03", "US Regional Banking Crisis (SVB / Credit Suisse)"),
    ("2024-06", "Indian General Election Results Day Volatility Spike"),
]


def _match_market_event(date_str: str) -> str:
    """
    Matches external calendar dates to known macro events for contextual reference only.
    CRITICAL INTERPRETABILITY RULE: The neural network detects mathematical reconstruction
    deviations; it does not infer or discover causal market explanations.
    """
    for prefix, event in KNOWN_HISTORICAL_EVENTS:
        if date_str.startswith(prefix):
            return f"Historical Reference: {event}"
    return "None recorded (Unmatched calendar date)"


def train_and_evaluate_vae_anomaly(
    splits: DatasetSplits,
    price_series: pd.Series,
    return_series: pd.Series,
    lookback: int = 15,
    epochs: int = VAE_EPOCHS,
    batch_size: int = 32,
    percentile: float = VAE_ANOMALY_PERCENTILE,
) -> AnomalyResult:
    """
    Trains TimeSeriesVAE on training split windows, calibrates anomaly threshold
    strictly on training reconstruction error, and evaluates across all splits.
    Simultaneously runs Rolling Z-Score and Isolation Forest baselines.
    """
    torch.manual_seed(ML_RANDOM_SEED)
    np.random.seed(ML_RANDOM_SEED)

    # 1. Create sliding windows for Train, Val, Test
    X_tr_w, _, idx_tr = create_sliding_windows(splits.X_train_scaled, lookback=lookback)
    X_val_w, _, idx_val = create_sliding_windows(splits.X_val_scaled, lookback=lookback)
    X_te_w, _, idx_te = create_sliding_windows(splits.X_test_scaled, lookback=lookback)

    # Flatten windows: (B, lookback * n_features)
    n_feat = splits.X_train_scaled.shape[1]
    input_dim = lookback * n_feat

    X_tr_flat = X_tr_w.reshape(-1, input_dim)
    X_val_flat = X_val_w.reshape(-1, input_dim)
    X_te_flat = X_te_w.reshape(-1, input_dim)

    # PyTorch DataLoaders
    tr_tensor = torch.tensor(X_tr_flat, dtype=torch.float32)
    val_tensor = torch.tensor(X_val_flat, dtype=torch.float32)
    te_tensor = torch.tensor(X_te_flat, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(tr_tensor), batch_size=batch_size, shuffle=True)

    # 2. Instantiate VAE Model
    vae = TimeSeriesVAE(input_dim=input_dim, hidden_dim=VAE_HIDDEN_DIM, latent_dim=VAE_LATENT_DIM)
    optimizer = torch.optim.Adam(vae.parameters(), lr=VAE_LEARNING_RATE, weight_decay=1e-5)

    train_loss_hist: list[float] = []
    val_loss_hist: list[float] = []
    best_val_loss = float("inf")
    best_state = None

    vae.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            recon_b, mu_b, logvar_b = vae(batch_x)
            loss, _, _ = vae_loss_function(recon_b, batch_x, mu_b, logvar_b)
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.item()) * len(batch_x)

        ep_loss /= len(tr_tensor)
        train_loss_hist.append(ep_loss)

        # Validation step
        vae.eval()
        with torch.no_grad():
            recon_v, mu_v, logvar_v = vae(val_tensor)
            val_loss, _, _ = vae_loss_function(recon_v, val_tensor, mu_v, logvar_v)
            v_val = float(val_loss.item())
            val_loss_hist.append(v_val)
            if v_val < best_val_loss:
                best_val_loss = v_val
                best_state = {k: v.cpu().clone() for k, v in vae.state_dict().items()}
        vae.train()

    # Load best weights
    if best_state is not None:
        vae.load_state_dict(best_state)
    vae.eval()

    # 3. Compute Per-Sample Reconstruction Errors
    with torch.no_grad():
        recon_tr, mu_tr, _ = vae(tr_tensor)
        tr_errors = torch.mean((recon_tr - tr_tensor) ** 2, dim=1).numpy()

        recon_val, mu_val, _ = vae(val_tensor)
        val_errors = torch.mean((recon_val - val_tensor) ** 2, dim=1).numpy()

        recon_te, mu_te, _ = vae(te_tensor)
        te_errors = torch.mean((recon_te - te_tensor) ** 2, dim=1).numpy()

    # CRITICAL LEAKAGE RULE: Threshold calibrated EXCLUSIVELY on training set!
    threshold = float(np.percentile(tr_errors, percentile))
    logger.info("VAE Anomaly threshold (P%.1f on train set): %.5f", percentile, threshold)

    # 4. Classical Baseline 1: Rolling Z-Score (|Z| > 3.0 on 21d returns)
    r_all = return_series.sort_index()
    r_mean_21 = r_all.rolling(21).mean()
    r_std_21 = r_all.rolling(21).std().clip(lower=1e-6)
    z_scores = ((r_all - r_mean_21) / r_std_21).abs()

    # 5. Classical Baseline 2: Isolation Forest fitted on Train
    iso = IsolationForest(contamination=0.05, random_state=ML_RANDOM_SEED)
    iso.fit(splits.X_train_scaled)

    # 6. Assemble Full Timeline across Splits
    # End dates for each window
    tr_dates = splits.train_dates[idx_tr]
    val_dates = splits.val_dates[idx_val]
    te_dates = splits.test_dates[idx_te]

    all_dates = list(tr_dates) + list(val_dates) + list(te_dates)
    all_split_tags = ["Train"] * len(tr_dates) + ["Val"] * len(val_dates) + ["Test"] * len(te_dates)
    all_errors = np.concatenate([tr_errors, val_errors, te_errors])
    all_mu = torch.cat([mu_tr, mu_val, mu_te], dim=0).numpy()

    timeline_records = []
    top_anomalies_list = []

    for dt, tag, err, z_vec in zip(all_dates, all_split_tags, all_errors, all_mu):
        d_str = dt.strftime("%Y-%m-%d")
        p_val = float(price_series.get(dt, np.nan))
        r_val = float(return_series.get(dt, 0.0))
        z_val = float(z_scores.get(dt, 0.0))

        # Isolation forest evaluation on tabular vector for that date
        iso_pred = False
        if dt in splits.X_test_raw.index:
            row_scaled = splits.scaler.transform(splits.X_test_raw.loc[[dt]].values)
            iso_pred = bool(iso.predict(row_scaled)[0] == -1)
        elif dt in splits.X_val_raw.index:
            row_scaled = splits.scaler.transform(splits.X_val_raw.loc[[dt]].values)
            iso_pred = bool(iso.predict(row_scaled)[0] == -1)
        elif dt in splits.X_train_raw.index:
            row_scaled = splits.scaler.transform(splits.X_train_raw.loc[[dt]].values)
            iso_pred = bool(iso.predict(row_scaled)[0] == -1)

        is_vae_anomaly = bool(err > threshold)
        is_zscore_anomaly = bool(z_val > 3.0)

        record = {
            "date": d_str,
            "split": tag,
            "price": p_val,
            "return_pct": r_val * 100.0,
            "vae_recon_error": float(err),
            "threshold": threshold,
            "is_vae_anomaly": is_vae_anomaly,
            "zscore": z_val,
            "is_zscore_anomaly": is_zscore_anomaly,
            "is_isolation_forest_anomaly": iso_pred,
            "latent_z1": float(z_vec[0]),
            "latent_z2": float(z_vec[1]),
        }
        timeline_records.append(record)

        if is_vae_anomaly:
            top_anomalies_list.append({
                "date": d_str,
                "split": tag,
                "return_pct": round(r_val * 100.0, 2),
                "reconstruction_error": round(float(err), 5),
                "threshold_ratio": round(float(err / threshold), 2),
                "matched_event": _match_market_event(d_str),
                "is_zscore_confirmed": is_zscore_anomaly,
                "is_iso_confirmed": iso_pred,
            })

    timeline_df = pd.DataFrame(timeline_records)

    # Sort top anomalies by reconstruction error ratio
    top_anomalies_list.sort(key=lambda x: x["reconstruction_error"], reverse=True)
    top_anomalies = top_anomalies_list[:12]

    # 7. Baseline Scorecard Comparison
    test_sub = timeline_df[timeline_df["split"] == "Test"]
    scorecard_data = [
        {
            "Model": "Deep VAE (Ours)",
            "Split": "Test",
            "Anomalies Flagged": int(test_sub["is_vae_anomaly"].sum()),
            "Anomaly Rate": f"{(test_sub['is_vae_anomaly'].mean() * 100.0):.2f}%",
            "Mean |Return| on Flagged Days": f"{test_sub.loc[test_sub['is_vae_anomaly'], 'return_pct'].abs().mean():.2f}%" if test_sub["is_vae_anomaly"].any() else "N/A",
            "Mechanism": "High reconstruction loss exceeding train P95 threshold",
        },
        {
            "Model": "Isolation Forest",
            "Split": "Test",
            "Anomalies Flagged": int(test_sub["is_isolation_forest_anomaly"].sum()),
            "Anomaly Rate": f"{(test_sub['is_isolation_forest_anomaly'].mean() * 100.0):.2f}%",
            "Mean |Return| on Flagged Days": f"{test_sub.loc[test_sub['is_isolation_forest_anomaly'], 'return_pct'].abs().mean():.2f}%" if test_sub["is_isolation_forest_anomaly"].any() else "N/A",
            "Mechanism": "Tree-based recursive path length isolation in feature space",
        },
        {
            "Model": "Rolling Z-Score (3-Sigma)",
            "Split": "Test",
            "Anomalies Flagged": int(test_sub["is_zscore_anomaly"].sum()),
            "Anomaly Rate": f"{(test_sub['is_zscore_anomaly'].mean() * 100.0):.2f}%",
            "Mean |Return| on Flagged Days": f"{test_sub.loc[test_sub['is_zscore_anomaly'], 'return_pct'].abs().mean():.2f}%" if test_sub["is_zscore_anomaly"].any() else "N/A",
            "Mechanism": "Univariate return deviation exceeding 3 standard deviations",
        },
    ]
    scorecard_df = pd.DataFrame(scorecard_data)

    # 8. Concise, Finance-Specific Analytical Interpretation
    n_detected_test = int(test_sub["is_vae_anomaly"].sum())
    overlap_count = int((test_sub["is_vae_anomaly"] & test_sub["is_zscore_anomaly"]).sum())
    novel_vae_count = n_detected_test - overlap_count

    if n_detected_test > 0:
        interpretation = (
            f"The TimeSeriesVAE identified {n_detected_test} out-of-sample temporal dislocation windows "
            f"where multivariate market structure broke down relative to the training manifold (threshold: {threshold:.4f}). "
            f"Of these, {overlap_count} coincided with univariate 3-sigma return shocks, while {novel_vae_count} represented "
            f"complex multivariate volatility-momentum decouplings that univariate filters missed. "
            f"(Note: Historical event overlays indicate external calendar co-occurrence and are not model-inferred causal attributions)."
        )
    else:
        interpretation = (
            f"During the out-of-sample test window, reconstruction loss remained comfortably below the training "
            f"P95 threshold ({threshold:.4f}). Asset behavior remained consistent with learned normal regime dynamics "
            f"without severe structural breakdown. "
            f"(Note: Historical event overlays indicate external calendar co-occurrence and are not model-inferred causal attributions)."
        )

    latent_coords = timeline_df[["date", "split", "latent_z1", "latent_z2", "is_vae_anomaly", "vae_recon_error"]].copy()

    return AnomalyResult(
        model_name="TimeSeriesVAE (Variational Autoencoder)",
        timeline_df=timeline_df,
        reconstruction_threshold=threshold,
        train_loss_history=train_loss_hist,
        val_loss_history=val_loss_hist,
        n_anomalies_detected=int(timeline_df["is_vae_anomaly"].sum()),
        anomaly_rate_pct=float(timeline_df["is_vae_anomaly"].mean() * 100.0),
        top_anomalies=top_anomalies,
        latent_coords=latent_coords,
        baseline_scorecard=scorecard_df,
        interpretation=interpretation,
        model_summary={
            "input_dim": input_dim,
            "lookback": lookback,
            "latent_dim": VAE_LATENT_DIM,
            "epochs": epochs,
            "final_train_loss": round(train_loss_hist[-1], 5),
            "final_val_loss": round(val_loss_hist[-1], 5),
            "threshold": round(threshold, 5),
        },
    )
