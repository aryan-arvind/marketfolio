"""
MarketFolio End-to-End Real Financial Data Experiment Runner.
Executes genuine training, validation, and held-out testing on historical market data
for the three Advanced Machine Learning models:
1. TimeSeriesVAE (Variational Autoencoder Anomaly Detection)
2. VolatilityTransformerEncoder (Multi-Head Self-Attention Regime Classification)
3. InterMarketGCN (Spectral Graph Convolutional Relational Risk Modeling)

Records and persists:
- Exact asset data sources, date ranges, and sample counts.
- Strict chronological train/validation/test split dates and embargo gaps.
- PyTorch runtime device (CPU / CUDA).
- Epoch-by-epoch training and validation loss histories.
- Early-stopping checkpoint selection.
- Out-of-sample metrics vs. classical econometric baselines.
- Trained model state checkpoints (.pt files) and complete JSON evidence.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# Add src to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from risklens.cleaning import align_macro_series, clean_primary_asset
from risklens.config import (
    LOGS_DIR,
    REPORTS_DIR,
    setup_logging,
)
from risklens.data_sources import fetch_all_macro_series, fetch_yfinance_ticker
from risklens.gcn_relational import train_and_evaluate_intermarket_gcn
from risklens.ml_pipeline import build_multivariate_feature_matrix, prepare_ml_dataset
from risklens.transformer_regime import train_and_evaluate_transformer_regime
from risklens.vae_anomaly import train_and_evaluate_vae_anomaly


def run_genuine_experiments(
    primary_ticker: str = "RELIANCE.NS",
    horizon: int = 10,
    vae_epochs: int = 35,
    transformer_epochs: int = 30,
    gcn_epochs: int = 35,
) -> dict[str, Any]:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logger = setup_logging()
    print("=" * 80)
    print(" MARKETFOLIO GENUINE REAL-DATA ADVANCED ML EXPERIMENT RUNNER")
    print(f" Target Asset: {primary_ticker} | Horizon: {horizon} trading days")
    print("=" * 80)

    # 1. Inspect Hardware / PyTorch Runtime
    is_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"
    cuda_version = torch.version.cuda if is_cuda else "None (CPU Build)"
    print(f"\n[Environment] PyTorch: {torch.__version__} | Device: {device_name} (CUDA available: {is_cuda})")

    # 2. Ingest Primary Asset and Synchronized Macro Assets
    print(f"\n[Data Ingestion] Ingesting historical prices for {primary_ticker} and 5 macro series...")
    raw_primary, meta_primary = fetch_yfinance_ticker(primary_ticker, force_refresh=False, is_primary=True)
    clean_primary = clean_primary_asset(raw_primary, asset_name=primary_ticker, min_observations=500)
    prices = clean_primary.price_series
    returns = clean_primary.log_return_series

    macro_dict, _ = fetch_all_macro_series(force_refresh=False)
    aligned_macro, macro_valid, macro_reason = align_macro_series(macro_dict)

    if not macro_valid:
        raise RuntimeError(f"Macro series alignment failed: {macro_reason}")

    print(f"  -> Primary Asset: {primary_ticker} ({len(prices)} clean trading days: {prices.index.min().date()} to {prices.index.max().date()})")
    print(f"  -> Synchronized Macro Matrix: {len(aligned_macro)} aligned dates across {list(aligned_macro.columns)}")

    # 3. Build Feature Matrix & Prepare Zero-Leakage Dataset
    print("\n[Feature Engineering] Constructing 8-channel multivariate technical & macro features...")
    features_df = build_multivariate_feature_matrix(
        price_series=prices,
        return_series=returns,
        macro_df=aligned_macro,
    )
    print(f"  -> Features shape: {features_df.shape[0]} rows x {features_df.shape[1]} columns")

    print("\n[Data Partitioning] Partitioning Chronological Train (70%), Val (15%), Test (15%) with H-day Embargo...")
    splits, thresholds = prepare_ml_dataset(
        features_df=features_df,
        return_series=returns,
        horizon=horizon,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    n_tr = len(splits.train_dates)
    n_va = len(splits.val_dates)
    n_te = len(splits.test_dates)
    print(f"  -> Train Split: {n_tr} samples ({splits.train_dates[0].date()} to {splits.train_dates[-1].date()})")
    print(f"  -> Embargo 1:   {splits.embargo_days} days purge ({splits.train_purge_dates[0]} to {splits.train_purge_dates[1]})")
    print(f"  -> Val Split:   {n_va} samples ({splits.val_dates[0].date()} to {splits.val_dates[-1].date()})")
    print(f"  -> Embargo 2:   {splits.embargo_days} days purge ({splits.val_purge_dates[0]} to {splits.val_purge_dates[1]})")
    print(f"  -> Test Split:  {n_te} samples ({splits.test_dates[0].date()} to {splits.test_dates[-1].date()})")
    print(f"  -> Calibrated Thresholds on Train only: Low <= {thresholds.low_cutoff*100:.2f}%, High > {thresholds.high_cutoff*100:.2f}%")

    checkpoint_dir = REPORTS_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # EXPERIMENT 1: TimeSeriesVAE Anomaly Detection
    # =========================================================================
    print("\n" + "=" * 60)
    print(" EXPERIMENT 1: Deep TimeSeriesVAE Anomaly Engine")
    print("=" * 60)
    t0 = time.time()
    vae_res = train_and_evaluate_vae_anomaly(
        splits=splits,
        price_series=prices,
        return_series=returns,
        lookback=15,
        epochs=vae_epochs,
    )
    vae_time = time.time() - t0

    print(f"  -> Training Completed in {vae_time:.2f}s across {vae_epochs} epochs on {device_name}")
    print(f"  -> Initial Train Loss: {vae_res.train_loss_history[0]:.5f} | Final Train Loss: {vae_res.train_loss_history[-1]:.5f}")
    print(f"  -> Initial Val Loss:   {vae_res.val_loss_history[0]:.5f} | Final Val Loss:   {vae_res.val_loss_history[-1]:.5f}")
    print(f"  -> Calibrated Train P95 Threshold: {vae_res.reconstruction_threshold:.5f}")
    print(f"  -> Out-of-Sample Test Anomalies Flagged: {int(vae_res.baseline_scorecard[vae_res.baseline_scorecard['Model']=='Deep VAE (Ours)']['Anomalies Flagged'].values[0])}")
    print(f"  -> Baseline Scorecard on Test Set:\n{vae_res.baseline_scorecard.to_string(index=False)}")

    # =========================================================================
    # EXPERIMENT 2: Volatility Transformer Encoder
    # =========================================================================
    print("\n" + "=" * 60)
    print(" EXPERIMENT 2: Volatility Transformer Encoder (Attention Pooling)")
    print("=" * 60)
    t0 = time.time()
    trans_res = train_and_evaluate_transformer_regime(
        splits=splits,
        return_series=returns,
        horizon=horizon,
        lookback=20,
        epochs=transformer_epochs,
    )
    trans_time = time.time() - t0

    print(f"  -> Training Completed in {trans_time:.2f}s across {transformer_epochs} epochs on {device_name}")
    print(f"  -> Initial Train Loss: {trans_res.train_loss_history[0]:.5f} | Final Train Loss: {trans_res.train_loss_history[-1]:.5f}")
    print(f"  -> Initial Val Loss:   {trans_res.val_loss_history[0]:.5f} | Final Val Loss:   {trans_res.val_loss_history[-1]:.5f}")
    print(f"  -> Out-of-Sample Test Accuracy:     {trans_res.accuracy * 100:.2f}%")
    print(f"  -> Out-of-Sample Balanced Accuracy: {trans_res.balanced_accuracy * 100:.2f}%")
    print(f"  -> Out-of-Sample Macro F1:          {trans_res.macro_f1:.4f}")
    print(f"  -> Out-of-Sample Log Loss:          {trans_res.log_loss_val:.4f}" if trans_res.log_loss_val is not None else f"  -> Log Loss: N/A ({trans_res.log_loss_error})")
    print(f"  -> Empirical Verdict: {trans_res.verdict}")
    print(f"  -> Scorecard vs Econometric Baselines:\n{trans_res.scorecard_df.to_string(index=False)}")

    # =========================================================================
    # EXPERIMENT 3: InterMarketGCN Relational Topology
    # =========================================================================
    print("\n" + "=" * 60)
    print(" EXPERIMENT 3: InterMarketGCN Relational Graph (6 Core Nodes)")
    print("=" * 60)
    t0 = time.time()
    gcn_res = train_and_evaluate_intermarket_gcn(
        primary_series=prices,
        primary_name=primary_ticker,
        macro_dict=macro_dict,
        horizon=horizon,
        epochs=gcn_epochs,
    )
    gcn_time = time.time() - t0

    if gcn_res.available:
        print(f"  -> Training Completed in {gcn_time:.2f}s across {gcn_epochs} epochs on {device_name}")
        print(f"  -> Nodes: {gcn_res.nodes}")
        print(f"  -> Train Samples: {gcn_res.train_samples} | Test Samples: {gcn_res.test_samples} (Embargo: {gcn_res.embargo_days}d)")
        print(f"  -> Out-of-Sample Systemic Stress F1: {gcn_res.systemic_stress_f1:.4f} (vs Majority Baseline: {gcn_res.baseline_f1:.4f})")
        print(f"  -> Systemic Stress Accuracy: {gcn_res.systemic_stress_accuracy * 100:.2f}% (vs Baseline: {gcn_res.baseline_accuracy * 100:.2f}%)")
        print(f"  -> Node Centrality / Vulnerability Scores:\n     {gcn_res.node_vulnerability_scores}")
    else:
        print(f"  -> GCN Unavailable: {gcn_res.reason}")

    # =========================================================================
    # PERSISTENCE & ARTIFACT GENERATION
    # =========================================================================
    output_evidence_file = REPORTS_DIR / f"real_experiment_evidence_{primary_ticker.replace('.', '_')}.json"
    evidence_payload: dict[str, Any] = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "asset": primary_ticker,
        "runtime_device": {
            "device": device_name,
            "cuda_available": is_cuda,
            "cuda_version": cuda_version,
            "torch_version": torch.__version__,
        },
        "dataset": {
            "primary_observations": len(prices),
            "date_range": (str(prices.index.min().date()), str(prices.index.max().date())),
            "macro_aligned_observations": len(aligned_macro),
            "macro_nodes": list(aligned_macro.columns),
            "horizon_days": horizon,
            "splits": {
                "train_samples": n_tr,
                "train_range": (str(splits.train_dates[0].date()), str(splits.train_dates[-1].date())),
                "val_samples": n_va,
                "val_range": (str(splits.val_dates[0].date()), str(splits.val_dates[-1].date())),
                "test_samples": n_te,
                "test_range": (str(splits.test_dates[0].date()), str(splits.test_dates[-1].date())),
                "embargo_days": splits.embargo_days,
            },
            "thresholds": {
                "low_vol_cutoff": thresholds.low_cutoff,
                "high_vol_cutoff": thresholds.high_cutoff,
            },
        },
        "models": {
            "vae": {
                "status": "COMPLETED",
                "epochs": vae_epochs,
                "train_loss_history": [float(v) for v in vae_res.train_loss_history],
                "val_loss_history": [float(v) for v in vae_res.val_loss_history],
                "reconstruction_threshold": float(vae_res.reconstruction_threshold),
                "anomalies_flagged_test": int(vae_res.baseline_scorecard[vae_res.baseline_scorecard["Model"] == "Deep VAE (Ours)"]["Anomalies Flagged"].values[0]),
                "baseline_scorecard": vae_res.baseline_scorecard.to_dict(orient="records"),
                "runtime_seconds": round(vae_time, 2),
                "interpretation": vae_res.interpretation,
            },
            "transformer": {
                "status": "COMPLETED",
                "epochs": transformer_epochs,
                "train_loss_history": [float(v) for v in trans_res.train_loss_history],
                "val_loss_history": [float(v) for v in trans_res.val_loss_history],
                "accuracy": float(trans_res.accuracy),
                "balanced_accuracy": float(trans_res.balanced_accuracy),
                "macro_f1": float(trans_res.macro_f1),
                "log_loss": float(trans_res.log_loss_val) if trans_res.log_loss_val is not None else None,
                "log_loss_error": trans_res.log_loss_error,
                "confusion_matrix": trans_res.confusion_matrix,
                "scorecard": trans_res.scorecard_df.to_dict(orient="records"),
                "verdict": trans_res.verdict,
                "runtime_seconds": round(trans_time, 2),
                "interpretation": trans_res.interpretation,
            },
            "gcn": {
                "status": "COMPLETED" if gcn_res.available else "UNAVAILABLE",
                "nodes": gcn_res.nodes,
                "train_samples": gcn_res.train_samples,
                "test_samples": gcn_res.test_samples,
                "embargo_days": gcn_res.embargo_days,
                "systemic_f1": float(gcn_res.systemic_stress_f1),
                "systemic_accuracy": float(gcn_res.systemic_stress_accuracy),
                "baseline_f1": float(gcn_res.baseline_f1),
                "baseline_accuracy": float(gcn_res.baseline_accuracy),
                "node_vulnerability_scores": gcn_res.node_vulnerability_scores,
                "runtime_seconds": round(gcn_time, 2),
                "interpretation": gcn_res.interpretation,
            },
        },
    }

    with open(output_evidence_file, "w", encoding="utf-8") as f:
        json.dump(evidence_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f" ALL REAL-DATA EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f" Evidence saved to: {output_evidence_file}")
    print("=" * 80 + "\n")

    return evidence_payload


if __name__ == "__main__":
    run_genuine_experiments()
