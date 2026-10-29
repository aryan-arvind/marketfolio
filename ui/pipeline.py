"""
MarketFolio Pipeline Runner.
Coordinates data ingestion, cleaning, stationarity tests, volatility forecasting,
and Advanced Deep Learning models:
- Variational Autoencoder (VAE) Anomaly Engine
- Transformer Volatility Regime Classifier
- Inter-Market Graph Convolutional Network (GCN)
- Decoupled Trello Research Workflow Tracker
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from risklens.arma_models import grid_search_arma
from risklens.cleaning import align_macro_series, clean_primary_asset
from risklens.data_sources import (
    fetch_all_macro_series,
    fetch_mutual_fund_nav,
    fetch_yfinance_ticker,
    resolve_mutual_fund,
)
from risklens.diagnostics import (
    compute_ljung_box,
    run_stationarity_suite,
)
from risklens.evaluation import run_rolling_origin_evaluation
from risklens.gcn_relational import train_and_evaluate_intermarket_gcn
from risklens.ml_pipeline import (
    build_multivariate_feature_matrix,
    construct_volatility_regime_targets,
    create_chronological_splits,
    prepare_ml_dataset,
)
from risklens.narrator import generate_narrative
from risklens.regression_arma import fit_regression_arma
from risklens.seasonality import (
    analyze_calendar_effects,
    run_stl_decomposition,
)
from risklens.transformer_regime import train_and_evaluate_transformer_regime
from risklens.trello_tracker import fetch_trello_research_board
from risklens.vae_anomaly import train_and_evaluate_vae_anomaly
from risklens.var_analysis import run_var_analysis
from risklens.volatility import forecast_volatility

logger = logging.getLogger("risklens")


def execute_analysis_pipeline(
    asset_mode: str,
    asset_input: str,
    benchmark_symbol: str,
    horizon: int,
    force_refresh: bool,
    max_origins: int = 12,
    status_box: Any = None,
) -> dict[str, Any]:
    """
    Executes the full MarketFolio quantitative and machine learning analytics pipeline.
    Combines classical econometrics with verified deep neural models (VAE, Transformer, GCN).
    """
    warnings_list: list[str] = []

    if status_box:
        status_box.write("Step 1/5: Ingesting & cleaning historical price series...")

    if asset_mode == "Indian Mutual Fund":
        scheme_code, scheme_name = resolve_mutual_fund(asset_input)
        raw_price, meta = fetch_mutual_fund_nav(scheme_code, scheme_name=scheme_name, force_refresh=force_refresh)
        display_name = scheme_name
        symbol_code = str(scheme_code)
        if "dividend" in scheme_name.lower() or "idcw" in scheme_name.lower():
            warnings_list.append(
                f"Selected fund '{scheme_name}' is a Dividend/IDCW plan. "
                "Dividend payouts distort NAV returns; Growth plans provide clean return series."
            )
    else:
        raw_price, meta = fetch_yfinance_ticker(asset_input, force_refresh=force_refresh, is_primary=True)
        display_name = asset_input
        symbol_code = asset_input

    if meta.get("is_stale"):
        warnings_list.append(f"Network interruption: Using cached data as of {meta.get('last_date')}.")

    # 1. Clean primary series
    clean_res = clean_primary_asset(raw_price, asset_name=display_name)
    prices = clean_res.price_series
    returns = clean_res.log_return_series

    # 2. Stationarity & Autocorrelation
    stat_suite = run_stationarity_suite(prices, returns, display_name)
    lb_returns = compute_ljung_box(returns, lags=[10, 20])
    lb_sq_returns = compute_ljung_box(returns ** 2, lags=[10, 20])

    if status_box:
        status_box.write("Step 2/5: Fitting conditional mean ARMA & realized volatility cones...")

    # 3. Calendar Tendencies & Decomposition
    cal_res = analyze_calendar_effects(returns, display_name)
    stl_res = run_stl_decomposition(prices, display_name, period=21)

    # 4. Conditional Mean ARMA (max_p=2, max_q=2 for rapid convergence)
    arma_search = grid_search_arma(returns, display_name, d=0, max_p=2, max_q=2)

    # 5. Volatility Modeling & Bootstrap Forecast
    vol_res = forecast_volatility(returns, display_name, horizon=horizon)

    # 6. Benchmark Sensitivity Regression
    bench_clean = None
    market_res = None
    try:
        bench_raw, _ = fetch_yfinance_ticker(benchmark_symbol, force_refresh=force_refresh, is_primary=False)
        bench_clean = clean_primary_asset(bench_raw, asset_name=benchmark_symbol, min_observations=250)
        market_res = fit_regression_arma(returns, bench_clean.log_return_series, asset_name=display_name, benchmark_name=benchmark_symbol)
    except Exception as exc:
        warnings_list.append(f"Benchmark regression vs {benchmark_symbol} skipped: {exc}")

    # 7. Global Macro Series Ingestion
    macro_dict, _ = fetch_all_macro_series(force_refresh=force_refresh)
    aligned_macro, macro_valid, macro_reason = align_macro_series(macro_dict)
    var_res = None
    if macro_valid:
        v_res, v_ok, _ = run_var_analysis(aligned_macro)
        if v_ok:
            var_res = v_res

    if status_box:
        status_box.write("Step 3/5: Calibrating Deep VAE Anomaly Engine...")

    # 8. Advanced ML Pipeline: Features & Leakage-Free Chronological Splits
    features_df = build_multivariate_feature_matrix(
        price_series=prices,
        return_series=returns,
        macro_df=aligned_macro if macro_valid else None,
    )

    # Pre-aligns features and forward targets, calibrates thresholds on train only, enforces H-day embargo
    splits, thresholds = prepare_ml_dataset(
        features_df=features_df,
        return_series=returns,
        horizon=horizon,
    )

    # 9. Train & Evaluate Deep VAE Anomaly Engine
    vae_res = None
    try:
        vae_res = train_and_evaluate_vae_anomaly(
            splits=splits,
            price_series=prices,
            return_series=returns,
            lookback=15,
            epochs=3,
        )
    except Exception as exc:
        logger.error("VAE training error: %s", exc)
        warnings_list.append(f"VAE Anomaly Engine skipped: {exc}")

    if status_box:
        status_box.write("Step 4/5: Calibrating Transformer Volatility Regime Classifier...")

    # 10. Train & Evaluate Volatility Transformer Encoder
    transformer_res = None
    try:
        transformer_res = train_and_evaluate_transformer_regime(
            splits=splits,
            return_series=returns,
            horizon=horizon,
            lookback=20,
            epochs=3,
        )
    except Exception as exc:
        logger.error("Transformer training error: %s", exc)
        warnings_list.append(f"Transformer Volatility Classifier skipped: {exc}")

    # 11. Train & Evaluate Inter-Market GCN
    gcn_res = None
    try:
        if macro_valid and macro_dict:
            gcn_res = train_and_evaluate_intermarket_gcn(
                primary_series=prices,
                primary_name=display_name,
                macro_dict=macro_dict,
                horizon=horizon,
                epochs=3,
            )
    except Exception as exc:
        logger.error("GCN training error: %s", exc)
        warnings_list.append(f"Inter-Market GCN skipped: {exc}")

    if status_box:
        status_box.write("Step 5/5: Running rolling-origin out-of-sample backtest & assembling scorecard...")

    # 12. Out-of-Sample Rolling-Origin Backtesting (ARIMA/SARIMAX Volatility)
    eval_res = run_rolling_origin_evaluation(
        series=vol_res.log_vol_series,
        target_name=f"{display_name} Volatility",
        model_order=vol_res.best_order,
        horizon=horizon,
        max_origins=min(max_origins, 3),
    )

    # 13. Trello Research Workflow Tracker
    trello_res = fetch_trello_research_board()

    # 14. Assembled Results Structure
    first_lb_p = (
        float(arma_search.best_model.lb_df.iloc[0]["lb_pvalue"])
        if len(arma_search.best_model.lb_df) > 0
        else 0.0
    )

    assembled_results = {
        "asset_metadata": {
            "name": display_name,
            "symbol_or_code": symbol_code,
            "source": meta.get("source", "Public Data"),
            "last_available_date": meta.get("last_date", str(prices.index.max().date())),
            "first_available_date": meta.get("first_date", str(prices.index.min().date())),
            "fetch_timestamp": meta.get("fetch_timestamp", ""),
            "n_observations": len(returns),
            "is_stale": meta.get("is_stale", False),
        },
        "warnings": warnings_list,
        "stationarity": {
            "log_price_verdict": stat_suite["log_price"].verdict,
            "log_returns_verdict": stat_suite["log_returns"].verdict,
            "returns_adf_pvalue": stat_suite["log_returns"].adf_pvalue,
            "returns_kpss_pvalue": stat_suite["log_returns"].kpss_pvalue,
            "price_adf_stat": stat_suite["log_price"].adf_stat,
            "price_adf_pvalue": stat_suite["log_price"].adf_pvalue,
            "price_kpss_stat": stat_suite["log_price"].kpss_stat,
            "price_kpss_pvalue": stat_suite["log_price"].kpss_pvalue,
            "returns_adf_stat": stat_suite["log_returns"].adf_stat,
            "returns_kpss_stat": stat_suite["log_returns"].kpss_stat,
            "lb_returns_pvalue_lag10": float(lb_returns.loc[10, "lb_pvalue"]),
            "lb_sq_returns_pvalue_lag10": float(lb_sq_returns.loc[10, "lb_pvalue"]),
        },
        "seasonality": {
            "day_of_week_significant": cal_res.is_day_of_week_significant,
            "ols_f_stat": cal_res.ols_f_stat,
            "ols_f_pvalue": cal_res.ols_f_pvalue,
            "kruskal_stat": cal_res.kruskal_stat,
            "kruskal_pvalue": cal_res.kruskal_pvalue,
            "trend_strength_f_t": stl_res.trend_strength_f_t,
            "seasonal_strength_f_s": stl_res.seasonal_strength_f_s,
            "summary_verdict": cal_res.summary_verdict,
        },
        "arma_predictability": {
            "best_order": str(arma_search.best_order),
            "best_aicc": arma_search.best_model.aicc,
            "best_bic": arma_search.best_model.bic,
            "constant_aicc": arma_search.constant_model.aicc,
            "delta_aicc_vs_constant": arma_search.best_model.aicc - arma_search.constant_model.aicc,
            "is_predictable": arma_search.is_predictable,
            "residual_lb_pvalue": first_lb_p,
            "residual_jb_pvalue": arma_search.best_model.jb_pvalue,
            "verdict": arma_search.predictability_verdict,
        },
        "volatility_forecast": {
            "last_observed_vol_pct": vol_res.last_observed_vol_pct,
            "point_forecast_pct": vol_res.point_forecast_pct,
            "expected_risk_range_lower_pct": vol_res.expected_risk_range_pct[0],
            "expected_risk_range_upper_pct": vol_res.expected_risk_range_pct[1],
            "naive_forecast_pct": vol_res.naive_forecast_pct,
            "rolling_mean_forecast_pct": vol_res.rolling_mean_forecast_pct,
            "bootstrap_fallback_used": vol_res.boot_failed_or_fallback,
        },
        "vae_anomaly": (
            {
                "available": True,
                "reconstruction_threshold": vae_res.reconstruction_threshold,
                "n_anomalies_detected": vae_res.n_anomalies_detected,
                "anomaly_rate_pct": vae_res.anomaly_rate_pct,
                "top_anomalies": vae_res.top_anomalies,
                "interpretation": vae_res.interpretation,
                "baseline_scorecard": vae_res.baseline_scorecard.to_dict(orient="records"),
                "model_summary": vae_res.model_summary,
            }
            if vae_res
            else {"available": False}
        ),
        "transformer_regime": (
            {
                "available": True,
                "accuracy": transformer_res.accuracy,
                "balanced_accuracy": transformer_res.balanced_accuracy,
                "macro_f1": transformer_res.macro_f1,
                "log_loss": transformer_res.log_loss_val,
                "confusion_matrix": transformer_res.confusion_matrix,
                "verdict": transformer_res.verdict,
                "interpretation": transformer_res.interpretation,
                "scorecard": transformer_res.scorecard_df.to_dict(orient="records"),
                "thresholds": {
                    "low_cutoff": transformer_res.thresholds.low_cutoff,
                    "high_cutoff": transformer_res.thresholds.high_cutoff,
                },
                "mean_attention_weights": transformer_res.mean_attention_weights,
            }
            if transformer_res
            else {"available": False}
        ),
        "gcn_relational": (
            {
                "available": gcn_res.available,
                "nodes": gcn_res.nodes,
                "node_vulnerability_scores": gcn_res.node_vulnerability_scores,
                "systemic_stress_accuracy": gcn_res.systemic_stress_accuracy,
                "systemic_stress_f1": gcn_res.systemic_stress_f1,
                "baseline_accuracy": gcn_res.baseline_accuracy,
                "baseline_f1": gcn_res.baseline_f1,
                "interpretation": gcn_res.interpretation,
            }
            if gcn_res and gcn_res.available
            else {"available": False, "reason": getattr(gcn_res, "reason", "Macro data not aligned")}
        ),
        "market_model": (
            {
                "benchmark_name": market_res.benchmark_name,
                "n_aligned_obs": market_res.n_aligned_obs,
                "start_date": market_res.start_date,
                "end_date": market_res.end_date,
                "beta": market_res.arma_beta,
                "beta_se": market_res.arma_beta_se,
                "beta_ci_lower": market_res.arma_beta_ci[0],
                "beta_ci_upper": market_res.arma_beta_ci[1],
                "alpha_annualized_pct": market_res.arma_alpha_annualized * 100.0,
                "alpha_pvalue": market_res.arma_alpha_pvalue,
                "r_squared": market_res.ols_r_squared,
                "durbin_watson": market_res.ols_dw_stat,
                "ols_lb_pvalue": market_res.ols_lb_pvalue,
                "interpretation": market_res.interpretation,
            }
            if market_res
            else {}
        ),
        "var_macro": (
            {
                "available": True,
                "aligned_obs": var_res.aligned_obs,
                "selected_lag": var_res.selected_lag,
                "cholesky_order": var_res.cholesky_order,
                "granger_causal_factors": [g.causing_variable for g in var_res.granger_tests if g.is_causal_05],
                "granger_details": [
                    {
                        "var": g.causing_variable,
                        "f_stat": g.f_stat,
                        "p_val": g.p_value,
                        "conclusion": g.conclusion,
                    }
                    for g in var_res.granger_tests
                ],
                "var_beats_baseline": var_res.var_beats_baseline,
                "oos_verdict": var_res.out_of_sample_verdict,
            }
            if var_res
            else {"available": False, "reason": macro_reason}
        ),
        "evaluation": {
            "n_origins": eval_res.n_origins_evaluated,
            "horizon": eval_res.horizon,
            "model_mase_improvement_pct": eval_res.model_metrics.mase_improvement_pct,
            "table_rows": eval_res.metrics_table.to_dict(orient="records"),
            "verdict": eval_res.summary_verdict,
        },
        "flagged_anomalies": clean_res.flagged_anomalies.to_dict(orient="records"),
        "trello": trello_res,
    }

    narrative = generate_narrative(assembled_results, use_llm=False)

    return {
        "display_name": display_name,
        "symbol_code": symbol_code,
        "meta": meta,
        "warnings": warnings_list,
        "clean_res": clean_res,
        "stat_suite": stat_suite,
        "lb_returns": lb_returns,
        "lb_sq_returns": lb_sq_returns,
        "cal_res": cal_res,
        "stl_res": stl_res,
        "arma_search": arma_search,
        "vol_res": vol_res,
        "market_res": market_res,
        "bench_returns": bench_clean.log_return_series if bench_clean else None,
        "var_res": var_res,
        "eval_res": eval_res,
        "splits": splits,
        "vae_res": vae_res,
        "transformer_res": transformer_res,
        "gcn_res": gcn_res,
        "trello_res": trello_res,
        "narrative": narrative,
        "assembled_results": assembled_results,
    }
