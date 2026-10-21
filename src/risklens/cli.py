"""
Command Line Interface (CLI) runner for RiskLens.
Entry point: python -m risklens.cli
Orchestrates the complete quantitative analysis pipeline, logs progress,
saves report artifacts, and outputs a terminal executive summary table.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from risklens.arma_models import grid_search_arma, plot_arma_residual_diagnostics
from risklens.cleaning import align_macro_series, clean_primary_asset
from risklens.config import (
    DEFAULT_BENCHMARK_SYMBOL,
    DEFAULT_FORECAST_HORIZON,
    REPORTS_DIR,
    setup_logging,
)
from risklens.data_sources import (
    fetch_all_macro_series,
    fetch_mutual_fund_nav,
    fetch_yfinance_ticker,
    resolve_mutual_fund,
)
from risklens.diagnostics import (
    compute_ljung_box,
    plot_acf_pacf_diagnostics,
    run_stationarity_suite,
)
from risklens.evaluation import run_rolling_origin_evaluation
from risklens.narrator import generate_narrative
from risklens.regression_arma import fit_regression_arma, plot_market_regression
from risklens.report import create_report_directory, export_analysis_bundle
from risklens.seasonality import (
    analyze_calendar_effects,
    plot_seasonality_and_subseries,
    plot_stl_components,
    run_stl_decomposition,
)
from risklens.var_analysis import plot_var_diagnostics_and_irf, run_var_analysis
from risklens.volatility import forecast_volatility, plot_volatility_forecast


def build_arg_parser() -> argparse.ArgumentParser:
    """Configures the command-line arguments according to the ATSA specification."""
    parser = argparse.ArgumentParser(
        prog="risklens",
        description="RiskLens: Real-Data Risk and Macro-Linkage Monitor for Indian Mutual Funds and Equities.",
    )

    asset_group = parser.add_mutually_exclusive_group(required=True)
    asset_group.add_argument(
        "--fund-query",
        type=str,
        help="Search query text to resolve Indian mutual fund from AMFI via mfapi.in (e.g. 'Nifty 50 Index Fund Direct Growth')",
    )
    asset_group.add_argument(
        "--ticker",
        type=str,
        help="Stock or index ticker symbol via Yahoo Finance (e.g. 'RELIANCE.NS', '^NSEI')",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default=DEFAULT_BENCHMARK_SYMBOL,
        help=f"Benchmark ticker symbol for alpha/beta regression (default: {DEFAULT_BENCHMARK_SYMBOL})",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_FORECAST_HORIZON,
        help=f"Volatility forecast horizon in trading days (default: {DEFAULT_FORECAST_HORIZON})",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download of public data, bypassing local 6-hour cache",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM narrator and use deterministic econometric template",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Custom output directory for generated reports (default: reports/<asset>_<date>/)",
    )

    return parser


def run_pipeline(
    fund_query: str | None = None,
    ticker: str | None = None,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    horizon: int = DEFAULT_FORECAST_HORIZON,
    force_refresh: bool = False,
    use_llm: bool = True,
    custom_out_dir: str | None = None,
) -> Path:
    """
    Executes the entire RiskLens ATSA pipeline on real public data.
    Returns the path to the generated markdown report.
    """
    logger = setup_logging()
    logger.info("==================================================")
    logger.info("Initializing RiskLens ATSA Quantitative Pipeline")
    logger.info("==================================================")

    # 1. Ingestion: Resolve Primary Asset
    warnings_list: list[str] = []
    if fund_query:
        logger.info("Resolving mutual fund from query: '%s'", fund_query)
        scheme_code, scheme_name = resolve_mutual_fund(fund_query)
        raw_price_series, asset_metadata = fetch_mutual_fund_nav(
            scheme_code, scheme_name=scheme_name, force_refresh=force_refresh
        )
        asset_id = f"mf_{scheme_code}"
        display_name = scheme_name
        symbol_code = str(scheme_code)
        if "dividend" in scheme_name.lower() or "idcw" in scheme_name.lower():
            warnings_list.append(
                f"Selected fund '{scheme_name}' is a Dividend/IDCW plan. "
                "Dividend payouts distort NAV time series returns. Growth plans are advised."
            )
    elif ticker:
        logger.info("Fetching equity/index series for ticker: '%s'", ticker)
        raw_price_series, asset_metadata = fetch_yfinance_ticker(
            ticker, force_refresh=force_refresh, is_primary=True
        )
        asset_id = ticker
        display_name = ticker
        symbol_code = ticker
    else:
        raise ValueError("Must provide either --fund-query or --ticker.")

    if asset_metadata.get("is_stale"):
        warnings_list.append(
            f"Network interruption encountered; operating on STALE CACHE as of {asset_metadata.get('last_date')}."
        )

    # 2. Data Cleaning & Return Calculation
    logger.info("Cleaning primary asset time series...")
    clean_res = clean_primary_asset(raw_price_series, asset_name=display_name)
    returns = clean_res.log_return_series
    prices = clean_res.price_series

    # Setup report directory
    base_out = Path(custom_out_dir) if custom_out_dir else REPORTS_DIR
    report_dir = create_report_directory(asset_id, base_out_dir=base_out)
    logger.info("Report artifacts will be saved to: %s", report_dir)

    # 3. Diagnostics: Stationarity & Autocorrelation
    logger.info("Running Unit 1 & 5 stationarity diagnostics (ADF & KPSS)...")
    stat_suite = run_stationarity_suite(prices, returns, display_name)
    price_stat = stat_suite["log_price"]
    return_stat = stat_suite["log_returns"]

    # Autocorrelation tests
    lb_returns = compute_ljung_box(returns, lags=[10, 20])
    lb_sq_returns = compute_ljung_box(returns ** 2, lags=[10, 20])

    acf_pacf_fig_path = report_dir / "fig_acf_pacf.png"
    plot_acf_pacf_diagnostics(returns, display_name, output_path=str(acf_pacf_fig_path))
    plt.close("all")

    # 4. Seasonality & Calendar Effects
    logger.info("Analyzing Unit 1 calendar effects and STL decomposition...")
    calendar_res = analyze_calendar_effects(returns, display_name)
    stl_res = run_stl_decomposition(prices, display_name, period=21)

    fig_season_path = report_dir / "fig_seasonality.png"
    plot_seasonality_and_subseries(returns, calendar_res, display_name, output_path=str(fig_season_path))

    fig_stl_path = report_dir / "fig_stl_decomposition.png"
    plot_stl_components(stl_res, display_name, output_path=str(fig_stl_path))
    plt.close("all")

    # 5. ARMA Conditional Mean Modeling & Predictability
    logger.info("Searching optimal ARMA(p,q) orders via AICc...")
    # d=0 for stationary returns
    arma_search = grid_search_arma(returns, display_name, d=0, max_p=3, max_q=3)
    best_arma = arma_search.best_model

    fig_arma_resid_path = report_dir / "fig_arma_residuals.png"
    plot_arma_residual_diagnostics(best_arma, display_name, output_path=str(fig_arma_resid_path))
    plt.close("all")

    # 6. Realized Volatility & Residual Bootstrap Forecasting
    logger.info("Modeling realized volatility and running residual bootstrap (B=2000)...")
    vol_forecast = forecast_volatility(returns, display_name, horizon=horizon)

    fig_vol_path = report_dir / "fig_volatility_forecast.png"
    plot_volatility_forecast(vol_forecast, display_name, output_path=str(fig_vol_path))
    plt.close("all")

    # 7. Systematic Market Model: Regression with ARMA Errors
    logger.info("Fetching benchmark series '%s' and estimating market model...", benchmark_symbol)
    try:
        bench_raw, bench_meta = fetch_yfinance_ticker(benchmark_symbol, force_refresh=force_refresh, is_primary=False)
        bench_clean = clean_primary_asset(bench_raw, asset_name=benchmark_symbol, min_observations=250)
        market_res = fit_regression_arma(
            returns, bench_clean.log_return_series, asset_name=display_name, benchmark_name=benchmark_symbol
        )

        fig_mkt_path = report_dir / "fig_market_regression.png"
        plot_market_regression(market_res, returns, bench_clean.log_return_series, output_path=str(fig_mkt_path))
        plt.close("all")
    except Exception as exc:
        logger.warning("Market regression failed or skipped: %s", exc)
        market_res = None

    # 8. Macro Linkages & Vector Autoregression (VAR)
    logger.info("Fetching global macro series and estimating VAR system...")
    macro_dict, _ = fetch_all_macro_series(force_refresh=force_refresh)
    aligned_macro, macro_valid, macro_reason = align_macro_series(macro_dict)

    if macro_valid:
        var_res, var_success, var_msg = run_var_analysis(aligned_macro)
        if var_success and var_res is not None:
            fig_var_path = report_dir / "fig_var_irf_fevd.png"
            plot_var_diagnostics_and_irf(var_res, output_path=str(fig_var_path))
            plt.close("all")
        else:
            logger.warning("VAR analysis failed: %s", var_msg)
            var_res = None
    else:
        logger.warning("VAR analysis skipped: %s", macro_reason)
        var_res = None

    # 9. Out-of-Sample Rolling-Origin Evaluation
    logger.info("Executing expanding-window out-of-sample backtesting...")
    # Evaluate volatility forecasting skill
    eval_res = run_rolling_origin_evaluation(
        series=vol_forecast.log_vol_series,
        target_name=f"{display_name} Realized Volatility",
        model_order=vol_forecast.best_order,
        horizon=horizon,
    )

    # 10. Assemble Results Dictionary
    results_data: dict[str, Any] = {
        "asset_metadata": {
            "name": display_name,
            "symbol_or_code": symbol_code,
            "source": asset_metadata.get("source", "Public Source"),
            "last_available_date": asset_metadata.get("last_date", str(prices.index.max().date())),
            "first_available_date": asset_metadata.get("first_date", str(prices.index.min().date())),
            "fetch_timestamp": asset_metadata.get("fetch_timestamp", ""),
            "n_observations": len(returns),
            "is_stale": asset_metadata.get("is_stale", False),
        },
        "warnings": warnings_list,
        "flagged_anomalies": clean_res.flagged_anomalies.to_dict(orient="records"),
        "stationarity": {
            "price_adf_stat": price_stat.adf_stat,
            "price_adf_pvalue": price_stat.adf_pvalue,
            "price_kpss_stat": price_stat.kpss_stat,
            "price_kpss_pvalue": price_stat.kpss_pvalue,
            "log_price_verdict": price_stat.verdict,
            "returns_adf_stat": return_stat.adf_stat,
            "returns_adf_pvalue": return_stat.adf_pvalue,
            "returns_kpss_stat": return_stat.kpss_stat,
            "returns_kpss_pvalue": return_stat.kpss_pvalue,
            "log_returns_verdict": return_stat.verdict,
            "lb_returns_pvalue_lag10": float(lb_returns.iloc[0]["lb_pvalue"]),
            "lb_sq_returns_pvalue_lag10": float(lb_sq_returns.iloc[0]["lb_pvalue"]),
        },
        "seasonality": {
            "ols_f_stat": calendar_res.ols_f_stat,
            "ols_f_pvalue": calendar_res.ols_f_pvalue,
            "kruskal_stat": calendar_res.kruskal_stat,
            "kruskal_pvalue": calendar_res.kruskal_pvalue,
            "day_of_week_significant": calendar_res.is_day_of_week_significant,
            "trend_strength_f_t": stl_res.trend_strength_f_t,
            "seasonal_strength_f_s": stl_res.seasonal_strength_f_s,
            "summary_verdict": calendar_res.summary_verdict,
        },
        "arma_predictability": {
            "best_order": f"({best_arma.order[0]},{best_arma.order[1]},{best_arma.order[2]})",
            "best_aicc": best_arma.aicc,
            "best_bic": best_arma.bic,
            "constant_aicc": arma_search.constant_model.aicc,
            "delta_aicc_vs_constant": best_arma.aicc - arma_search.constant_model.aicc,
            "is_predictable": arma_search.is_predictable,
            "verdict": arma_search.predictability_verdict,
            "residual_lb_pvalue": float(best_arma.lb_df.iloc[0]["lb_pvalue"]),
            "residual_jb_pvalue": best_arma.jb_pvalue,
        },
        "volatility_forecast": {
            "best_order": f"({vol_forecast.best_order[0]},{vol_forecast.best_order[1]},{vol_forecast.best_order[2]})",
            "last_observed_vol_pct": vol_forecast.last_observed_vol_pct,
            "point_forecast_pct": vol_forecast.point_forecast_pct,
            "expected_risk_range_lower_pct": vol_forecast.expected_risk_range_pct[0],
            "expected_risk_range_upper_pct": vol_forecast.expected_risk_range_pct[1],
            "naive_forecast_pct": vol_forecast.naive_forecast_pct,
            "rolling_mean_forecast_pct": vol_forecast.rolling_mean_forecast_pct,
            "bootstrap_fallback_used": vol_forecast.boot_failed_or_fallback,
        },
        "market_model": (
            {
                "benchmark_name": market_res.benchmark_name,
                "n_aligned_obs": market_res.n_aligned_obs,
                "start_date": market_res.start_date,
                "end_date": market_res.end_date,
                "beta": market_res.arma_beta,
                "beta_se": market_res.arma_beta_se,
                "beta_pvalue": market_res.arma_beta_pvalue,
                "beta_ci_lower": market_res.arma_beta_ci[0],
                "beta_ci_upper": market_res.arma_beta_ci[1],
                "alpha_annualized_pct": market_res.arma_alpha_annualized * 100.0,
                "alpha_pvalue": market_res.arma_alpha_pvalue,
                "r_squared": market_res.ols_r_squared,
                "durbin_watson": market_res.ols_dw_stat,
                "ols_lb_pvalue": market_res.ols_lb_pvalue,
                "best_arma_order": f"({market_res.best_arma_order[0]},{market_res.best_arma_order[1]},{market_res.best_arma_order[2]})",
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
                "lag_aic": var_res.lag_order_aic,
                "lag_bic": var_res.lag_order_bic,
                "is_stable": var_res.is_stable,
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
                "var_rmse_1step_pct": var_res.var_rmse_1step * 100.0,
                "baseline_rmse_1step_pct": var_res.baseline_rmse_1step * 100.0,
                "var_beats_baseline": var_res.var_beats_baseline,
                "oos_verdict": var_res.out_of_sample_verdict,
            }
            if var_res
            else {"available": False, "reason": macro_reason if not macro_valid else "VAR unstable"}
        ),
        "evaluation": {
            "n_origins": eval_res.n_origins_evaluated,
            "horizon": eval_res.horizon,
            "model_mase_improvement_pct": eval_res.model_metrics.mase_improvement_pct,
            "table_rows": eval_res.metrics_table.to_dict(orient="records"),
            "verdict": eval_res.summary_verdict,
        },
    }

    # 11. Narrative Synthesis
    logger.info("Generating narrative commentary...")
    narrative = generate_narrative(results_data, use_llm=use_llm and not False)

    # 12. Export Analysis Bundle
    tables_to_export: dict[str, pd.DataFrame] = {
        "table_flagged_anomalies": clean_res.flagged_anomalies,
        "table_arma_model_selection": arma_search.all_models_df,
        "table_calendar_dow": calendar_res.dow_stats,
        "table_calendar_month": calendar_res.month_stats,
        "table_rolling_origin_evaluation": eval_res.metrics_table,
    }
    if var_res:
        tables_to_export["table_var_fevd_nifty"] = var_res.fevd_nifty_horizon10
        tables_to_export["table_var_stationarity"] = var_res.stationarity_table

    md_report_path = export_analysis_bundle(report_dir, results_data, narrative, tables_to_export)

    # 13. Terminal Executive Summary Table
    print_terminal_summary(results_data)

    return md_report_path


def print_terminal_summary(results: dict[str, Any]) -> None:
    """Prints a structured summary table of the key findings to stdout."""
    meta = results.get("asset_metadata", {})
    stat = results.get("stationarity", {})
    cal = results.get("seasonality", {})
    arma = results.get("arma_predictability", {})
    vol = results.get("volatility_forecast", {})
    mkt = results.get("market_model", {})
    var_dict = results.get("var_macro", {})
    ev = results.get("evaluation", {})

    print("\n" + "=" * 80)
    print(f" RISKLENS QUANTITATIVE RISK & MACRO MONITOR: {meta.get('name')} ({meta.get('symbol_or_code')})")
    print("=" * 80)
    print(f"Data Freshness : Source: {meta.get('source')} | Last Date: {meta.get('last_available_date')} | N={meta.get('n_observations')}")
    print("-" * 80)
    print("STATISTICAL ANALYSIS                 EMPIRICAL RESULT                   ANALYTICAL VERDICT")
    print("-" * 80)
    print(f"Unit Root Tests (Log Price)         ADF p={stat.get('price_adf_pvalue', 0):.4f}, KPSS p={stat.get('price_kpss_pvalue', 0):.4f}      {stat.get('log_price_verdict')}")
    print(f"Unit Root Tests (Log Returns)       ADF p={stat.get('returns_adf_pvalue', 0):.4f}, KPSS p={stat.get('returns_kpss_pvalue', 0):.4f}      {stat.get('log_returns_verdict')}")
    print(f"Seasonality & Calendar Effects      OLS F-p={cal.get('ols_f_pvalue', 0):.4f}, Kruskal-p={cal.get('kruskal_pvalue', 0):.4f}    {'Calendar anomaly' if cal.get('day_of_week_significant') else 'No significant calendar anomaly'}")
    print(f"STL Decomposition (Trend / Season)  Trend F_T={cal.get('trend_strength_f_t', 0):.3f}, Season F_S={cal.get('seasonal_strength_f_s', 0):.3f}    {'Trend-dominated series' if cal.get('trend_strength_f_t', 0) > 0.8 else 'Moderate trend'}")
    print(f"Conditional Mean ARMA (Returns)     Best: ARIMA{arma.get('best_order')}, Delta AICc={arma.get('delta_aicc_vs_constant', 0):.2f}    {'Weak-Form EMH Unpredictable' if not arma.get('is_predictable') else 'Statistically Predictable'}")
    print(f"Realized Volatility Forecast (30d)  Current={vol.get('last_observed_vol_pct', 0):.1f}%, 90% Range=[{vol.get('expected_risk_range_lower_pct', 0):.1f}%, {vol.get('expected_risk_range_upper_pct', 0):.1f}%]  Point Forecast: {vol.get('point_forecast_pct', 0):.1f}% p.a.")
    if mkt:
        print(f"Benchmark Linkage ({mkt.get('benchmark_name')})        Beta={mkt.get('beta', 0):.3f}, Alpha={mkt.get('alpha_annualized_pct', 0):+.2f}% p.a. (p={mkt.get('alpha_pvalue', 0):.4f})  R^2={mkt.get('r_squared', 0):.3f}, DW={mkt.get('durbin_watson', 0):.2f}")
    if var_dict.get("available"):
        causal = ", ".join(var_dict.get("granger_causal_factors", [])) or "None"
        print(f"Macro VAR Linkages (5 Variables)    Lag p={var_dict.get('selected_lag')}, Granger Causal: {causal}    Stable VAR System (Cholesky)")
    print(f"Out-of-Sample MASE Evaluation       MASE Change vs Naive: {ev.get('model_mase_improvement_pct', 0):+.2f}%           {ev.get('n_origins')} expanding-window origins")
    print("=" * 80)
    print("Disclaimer: For quantitative evaluation only. Not investment advice.")
    print("=" * 80 + "\n")


def main() -> None:
    """CLI entry point function."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        md_path = run_pipeline(
            fund_query=args.fund_query,
            ticker=args.ticker,
            benchmark_symbol=args.benchmark,
            horizon=args.horizon,
            force_refresh=args.refresh,
            use_llm=not args.no_llm,
            custom_out_dir=args.out,
        )
        print(f"Successfully generated analysis bundle at: {md_path.parent}\nFull report: {md_path}\n")
    except Exception as exc:
        print(f"\n[FATAL ERROR] RiskLens analysis halted: {exc}\n", file=sys.stderr)
        logging.getLogger("risklens").exception("Pipeline crashed with fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
