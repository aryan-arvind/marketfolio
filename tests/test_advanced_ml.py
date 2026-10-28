"""
Tests for MarketFolio Advanced Machine Learning modules:
- Shared feature pipeline & leak-free chronological splits with embargo purging
- TimeSeriesVAE anomaly detection & baseline comparisons
- VolatilityTransformerEncoder regime classification & attention pooling
- InterMarketGCN relational risk modeling
- Trello research tracker graceful fallback
"""

import numpy as np
import pandas as pd
import pytest
import torch

from risklens.gcn_relational import (
    GraphConvolutionLayer,
    InterMarketGCN,
    compute_normalized_adjacency,
    train_and_evaluate_intermarket_gcn,
)
from risklens.ml_pipeline import (
    build_multivariate_feature_matrix,
    construct_volatility_regime_targets,
    create_chronological_splits,
    create_sliding_windows,
    evaluate_majority_baseline,
    evaluate_persistence_baseline,
    prepare_ml_dataset,
)
from risklens.transformer_regime import (
    VolatilityTransformerEncoder,
    train_and_evaluate_transformer_regime,
)
from risklens.trello_tracker import fetch_trello_research_board
from risklens.vae_anomaly import (
    TimeSeriesVAE,
    train_and_evaluate_vae_anomaly,
    vae_loss_function,
)


@pytest.fixture
def synthetic_market_data():
    """Generates synthetic daily prices and returns for 800 business days."""
    np.random.seed(42)
    n = 800
    dates = pd.bdate_range(start="2021-01-01", periods=n)
    # Geometric Brownian motion with volatility clustering
    rets = np.random.normal(0.0005, 0.012, size=n)
    # Inject volatility cluster
    rets[300:350] *= 3.5
    # Inject large discrete jump
    rets[500] = -0.09

    prices = 100.0 * np.exp(np.cumsum(rets))
    price_series = pd.Series(prices, index=dates)
    return_series = pd.Series(rets, index=dates)
    return price_series, return_series


def test_feature_pipeline_and_chronological_splits(synthetic_market_data):
    prices, rets = synthetic_market_data
    features_df = build_multivariate_feature_matrix(prices, rets)

    # 1. Verify feature generation integrity
    assert "log_return" in features_df.columns
    assert "vol_21d" in features_df.columns
    assert "price_zscore_21d" in features_df.columns
    assert not features_df.isna().any().any(), "Features must contain zero NaN values"

    # 2. Verify regime target construction on training slice
    _, regimes, thresholds = construct_volatility_regime_targets(
        rets, horizon=10, train_end_idx=int(len(features_df) * 0.70)
    )
    assert thresholds.low_cutoff < thresholds.high_cutoff
    assert set(regimes.unique()).issubset({0, 1, 2})

    # 3. Verify chronological splitting with embargo purging
    splits = create_chronological_splits(features_df, regimes, horizon=10)
    assert max(splits.train_dates) < min(splits.val_dates)
    assert max(splits.val_dates) < min(splits.test_dates)

    # Verify embargo gap
    train_val_gap = (min(splits.val_dates) - max(splits.train_dates)).days
    assert train_val_gap >= 10, f"Embargo gap between train and val must be >= 10 days, got {train_val_gap}"

    # Verify scaler fitted on train only
    assert splits.X_train_scaled.shape[0] == len(splits.train_dates)
    assert np.allclose(splits.X_train_scaled.mean(axis=0), 0.0, atol=1e-5)


def test_sliding_window_dimensions():
    X = np.random.randn(100, 8).astype(np.float32)
    y = np.random.randint(0, 3, size=100)
    X_w, y_w, indices = create_sliding_windows(X, y, lookback=15)

    assert X_w.shape == (86, 15, 8)
    assert y_w.shape == (86,)
    assert indices[-1] == 99


def test_vae_architecture_and_loss():
    input_dim = 15 * 8
    vae = TimeSeriesVAE(input_dim=input_dim, hidden_dim=32, latent_dim=4)
    x = torch.randn(16, input_dim)
    recon, mu, logvar = vae(x)

    assert recon.shape == x.shape
    assert mu.shape == (16, 4)
    assert logvar.shape == (16, 4)

    total_loss, recon_l, kl_l = vae_loss_function(recon, x, mu, logvar)
    assert total_loss.item() > 0.0
    assert recon_l.item() >= 0.0
    assert kl_l.item() >= 0.0


def test_vae_anomaly_training(synthetic_market_data):
    prices, rets = synthetic_market_data
    features_df = build_multivariate_feature_matrix(prices, rets)
    splits = create_chronological_splits(features_df, horizon=10)

    res = train_and_evaluate_vae_anomaly(
        splits=splits,
        price_series=prices,
        return_series=rets,
        lookback=10,
        epochs=5,  # fast test run
    )

    assert res.reconstruction_threshold > 0.0
    assert len(res.timeline_df) > 0
    assert "is_vae_anomaly" in res.timeline_df.columns
    assert len(res.baseline_scorecard) == 3
    assert "Deep VAE (Ours)" in res.baseline_scorecard["Model"].values


def test_transformer_architecture_and_attention():
    lookback = 20
    input_dim = 6
    model = VolatilityTransformerEncoder(
        input_dim=input_dim,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        num_classes=3,
    )
    x = torch.randn(8, lookback, input_dim)
    logits, attn_weights = model(x)

    assert logits.shape == (8, 3)
    assert attn_weights.shape == (8, lookback)
    # Check that attention weights sum to 1.0 along the temporal dimension
    attn_sums = torch.sum(attn_weights, dim=1).detach().numpy()
    assert np.allclose(attn_sums, 1.0, atol=1e-5)


def test_transformer_training_and_baselines(synthetic_market_data):
    prices, rets = synthetic_market_data
    features_df = build_multivariate_feature_matrix(prices, rets)
    train_end = int(len(features_df) * 0.70)
    _, regimes, thresholds = construct_volatility_regime_targets(rets, train_end_idx=train_end, horizon=10)
    splits = create_chronological_splits(features_df, regimes, horizon=10)
    splits.regime_thresholds = thresholds

    res = train_and_evaluate_transformer_regime(
        splits=splits,
        return_series=rets,
        horizon=10,
        lookback=15,
        epochs=4,  # fast test run
    )

    assert res.accuracy >= 0.0
    assert res.macro_f1 >= 0.0
    assert len(res.confusion_matrix) == 3
    assert len(res.baselines) == 3
    assert "Majority Class Baseline" in [b.name for b in res.baselines]
    assert "Last-Known Regime Persistence" in [b.name for b in res.baselines]
    assert len(res.mean_attention_weights) == 15
    assert res.log_loss_val is not None, "Log loss should be a valid float on properly shaped test output"
    assert res.log_loss_error is None


def test_gcn_architecture_and_training():
    # Test GraphConvolutionLayer
    in_feat, out_feat, N = 4, 16, 5
    gcn_layer = GraphConvolutionLayer(in_feat, out_feat)
    x = torch.randn(4, N, in_feat)
    adj_norm = torch.eye(N)
    out = gcn_layer(x, adj_norm)
    assert out.shape == (4, N, out_feat)

    # Test normalized adjacency
    corr = np.array([
        [1.0, 0.6, 0.1],
        [0.6, 1.0, -0.4],
        [0.1, -0.4, 1.0],
    ])
    adj_norm_np = compute_normalized_adjacency(corr, threshold=0.2)
    assert adj_norm_np.shape == (3, 3)
    assert np.allclose(adj_norm_np, adj_norm_np.T)  # Symmetric


def test_regime_threshold_requires_explicit_train_boundary(synthetic_market_data):
    """Regression Test: Ensures regime threshold calibration requires an explicit valid training boundary."""
    _, rets = synthetic_market_data

    # 1. train_end_idx = None must fail
    with pytest.raises(ValueError, match="train_end_idx must be an explicit positive integer"):
        construct_volatility_regime_targets(rets, train_end_idx=None, horizon=10)  # type: ignore

    # 2. train_end_idx = 0 must fail (never silently fall back to full series)
    with pytest.raises(ValueError, match="train_end_idx must be an explicit positive integer"):
        construct_volatility_regime_targets(rets, train_end_idx=0, horizon=10)

    # 3. train_end_idx exceeding series length must fail
    with pytest.raises(ValueError, match="strictly less than valid target length"):
        construct_volatility_regime_targets(rets, train_end_idx=len(rets) + 10, horizon=10)

    # 4. Valid positive boundary succeeds and calibrates solely on training portion
    valid_vol, regimes, th = construct_volatility_regime_targets(rets, train_end_idx=300, horizon=10)
    expected_low = float(valid_vol.iloc[:300].quantile(0.333))
    expected_high = float(valid_vol.iloc[:300].quantile(0.667))
    assert np.isclose(th.low_cutoff, expected_low, atol=1e-5)
    assert np.isclose(th.high_cutoff, expected_high, atol=1e-5)


def test_gcn_target_embargo_zero_leakage():
    """Regression Test: Ensures GCN requires 6 nodes and enforces forward return embargo with zero leakage."""
    np.random.seed(42)
    n = 450
    dates = pd.bdate_range(start="2022-01-01", periods=n)
    p_primary = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, size=n))), index=dates)

    macro_dict = {
        "SP500": pd.Series(4000.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.012, size=n))), index=dates),
        "Brent": pd.Series(80.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.02, size=n))), index=dates),
        "USDINR": pd.Series(82.0 * np.exp(np.cumsum(np.random.normal(0.0001, 0.003, size=n))), index=dates),
        "IndiaVIX": pd.Series(15.0 + np.abs(np.cumsum(np.random.normal(0, 0.5, size=n))), index=dates),
        "NIFTY50": pd.Series(18000.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.013, size=n))), index=dates),
    }

    # 1. Missing macro key must return available=False with explicit explanation
    incomplete_macro = {k: v for k, v in macro_dict.items() if k != "IndiaVIX"}
    res_incomplete = train_and_evaluate_intermarket_gcn(
        primary_series=p_primary,
        primary_name="RELIANCE.NS",
        macro_dict=incomplete_macro,
        horizon=10,
        epochs=2,
    )
    assert not res_incomplete.available
    assert "IndiaVIX" in res_incomplete.reason

    # 2. Complete 6-node system succeeds with strict embargo
    horizon = 10
    res_full = train_and_evaluate_intermarket_gcn(
        primary_series=p_primary,
        primary_name="RELIANCE.NS",
        macro_dict=macro_dict,
        horizon=horizon,
        epochs=3,
    )
    assert res_full.available
    assert len(res_full.nodes) == 6
    assert res_full.embargo_days == horizon
    assert res_full.train_samples > 0
    assert res_full.test_samples > 0

    # Verify chronological embargo: train end must precede test start
    train_end = pd.to_datetime(res_full.train_date_range[1])
    test_start = pd.to_datetime(res_full.test_date_range[0])
    gap_days = (test_start - train_end).days
    assert gap_days >= horizon, f"Embargo gap ({gap_days} days) must be >= horizon ({horizon} days)"


def test_transformer_log_loss_no_silent_fallback(synthetic_market_data):
    """Regression Test: Ensures log_loss never substitutes fabricated constants on calculation exceptions."""
    from unittest.mock import patch
    prices, rets = synthetic_market_data
    features_df = build_multivariate_feature_matrix(prices, rets)
    splits, _ = prepare_ml_dataset(features_df, rets, horizon=10)

    # Simulate an exception inside log_loss calculation
    with patch("risklens.transformer_regime.log_loss", side_effect=ValueError("Simulated log_loss failure")):
        res = train_and_evaluate_transformer_regime(
            splits=splits,
            return_series=rets,
            horizon=10,
            lookback=15,
            epochs=2,
        )

    # Must NOT substitute 1.0; must record None and explicit error reason
    assert res.log_loss_val is None, "Failed log_loss must be None, never substituted with 1.0"
    assert res.log_loss_error is not None
    assert "Simulated log_loss failure" in res.log_loss_error


def test_trello_tracker_state_distinction_and_no_fake_completion():
    """Regression Test: Ensures Trello tracker never reports fake completion and distinguishes API states."""
    from unittest.mock import patch, MagicMock
    from risklens.trello_tracker import (
        DEFAULT_WORKFLOW_LISTS,
        TrackerSyncMode,
        create_trello_card,
        fetch_trello_research_board,
        update_trello_card,
    )

    # 1. No card in DEFAULT_WORKFLOW_LISTS may be hardcoded as "Completed"
    for lst in DEFAULT_WORKFLOW_LISTS:
        for card in lst["cards"]:
            assert card["status"] != "Completed", f"Card {card['id']} must not claim unverified 'Completed' status"
            assert "Planned" in card["status"] or "Specification" in card["status"]

    # 2. Local mode when env vars are missing
    with patch.dict("os.environ", {}, clear=True):
        res_local = fetch_trello_research_board()
        assert res_local["mode"] == TrackerSyncMode.LOCAL_ROADMAP_ONLY
        assert not res_local["is_live"]
        assert not res_local["write_supported"]
        assert "Offline Specification" in res_local["status_label"]

        # Write operations in local mode must return explicit unsupported payload without raising
        write_res = create_trello_card(list_id="test-list", name="New Task")
        assert not write_res["success"]
        assert "unsupported" in write_res["error"]

        upd_res = update_trello_card(card_id="test-card", name="Updated Task")
        assert not upd_res["success"]
        assert "unsupported" in upd_res["error"]

    # 3. API Failure mode when credentials exist but request fails
    mock_env = {
        "TRELLO_API_KEY": "fake_key",
        "TRELLO_TOKEN": "fake_token",
        "TRELLO_BOARD_ID": "fake_board",
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized: Invalid key"

    with patch.dict("os.environ", mock_env, clear=True):
        with patch("requests.get", return_value=mock_resp):
            res_fail = fetch_trello_research_board()
            assert res_fail["mode"] == TrackerSyncMode.API_CONNECTION_FAILED
            assert not res_fail["is_live"]
            assert res_fail["error_detail"] is not None
            assert "HTTP 401" in res_fail["error_detail"]
            assert "API Connection Failed" in res_fail["status_label"]


def test_vae_historical_event_non_causal():
    """Regression Test: Ensures historical event annotations are explicitly framed as non-causal calendar references."""
    from risklens.vae_anomaly import _match_market_event

    # 1. Unmatched date must return explicit 'None recorded' rather than causal diagnoses
    res_unmatched = _match_market_event("2021-08-15")
    assert res_unmatched == "None recorded (Unmatched calendar date)"

    # 2. Matched dates must be prefixed with 'Historical Reference: '
    res_covid = _match_market_event("2020-03-23")
    assert res_covid.startswith("Historical Reference: ")
    assert "COVID-19" in res_covid

